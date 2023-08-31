import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import *
from scipy.optimize import linear_sum_assignment

def get_loss(no_object_idx, bboxes1, bboxes2, logits, idxs, gamma_iou, gamma_box):
    cost_mat = torch.zeros(len(bboxes1), len(bboxes2))
    for i in range(len(bboxes1)):
        p_c = logits[i][idxs[i]]

        if idxs[i] == no_object_idx:
            ls = - p_c / 10
            cost_mat[i] = ls
        else:
            for j in range(len(bboxes2)):
                ls = - torch.log(p_c) + get_box_loss(bboxes1[i].unsqueeze(0), bboxes2[j].unsqueeze(0), gamma_iou, gamma_box)
                cost_mat[i, j] = ls
    row_idx, col_idx = linear_sum_assignment(cost_mat.detach().numpy())
    return cost_mat[row_idx, col_idx].sum()

def iou(bboxes1, bboxes2):
    bboxes1_x1, bboxes1_x2 = bboxes1[:, 0], bboxes1[:, 0] + bboxes1[:, 2]
    bboxes2_x1, bboxes2_x2 = bboxes2[:, 0], bboxes2[:, 0] + bboxes2[:, 2]

    x1 = torch.stack([bboxes1_x1, bboxes2_x1])
    x2 = torch.stack([bboxes1_x2, bboxes2_x2])

    bboxes1_y1, bboxes1_y2 = bboxes1[:, 1], bboxes1[:, 1] + bboxes1[:, 3]
    bboxes2_y1, bboxes2_y2 = bboxes2[:, 1], bboxes2[:, 1] + bboxes2[:, 3]

    y1 = torch.stack([bboxes1_y1, bboxes2_y1])
    y2 = torch.stack([bboxes1_y2, bboxes2_y2])

    inter_x1 = torch.max(x1, dim=0).values
    inter_x2= torch.min(x2, dim=0).values

    inter_y1 = torch.max(y1, dim=0).values
    inter_y2= torch.min(y2, dim=0).values
    
    if (inter_x2 - inter_x1 > 0) or (inter_y2 - inter_y1 < 0):
        return 0
    
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    bbox1_area = bboxes1[:, 2] * bboxes1[:, 3]
    bbox2_area = bboxes2[:, 2] * bboxes2[:, 3]
    
    union = (bbox1_area + bbox2_area) - inter
    iou = inter / union
    
    b_x1 = torch.min(x1)
    b_x2= torch.max(x2)

    b_y1 = torch.min(y1)
    b_y2= torch.max(y2)
    
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    return 1 - iou - ((b_area - union) / b_area)

def get_box_loss(box1, box2, gamma_iou, gamma_box):

    ls_iou = iou(box1, box2)
    ls_box = gamma_iou * ls_iou + gamma_box * F.l1_loss(box2, box1)

    return ls_box

class DETREncoder(nn.Module):
    def __init__(self, d_model: int, nheads: int, d_ff: int):
        super(DETREncoder, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, nheads)
        self.ln1 = nn.LayerNorm(d_model)
        
        self.ff = FeedForward(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x, pe):

        out = self.ln1(x + self.mha(x + pe, x + pe, x))
        out = self.ln2(out + self.ff(out))
        
        return out
        

class DETRDecoder(nn.Module):
    def __init__(self, d_model: int, nheads: int, d_ff: int):
        super(DETRDecoder, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, nheads)
        self.ln1 = nn.LayerNorm(d_model)
        
        self.mha2 = MultiHeadAttention(d_model, nheads)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.ff = FeedForward(d_model, d_ff)
        self.ln3 = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor, enc_out: Tensor, pe):

        out = self.ln1(x + self.mha1(x, x, x))
        out = self.ln2(out + self.mha2(q=out, k=enc_out + pe, v=enc_out))
        out = self.ln2(out + self.ff(out))
        
        return out
    
class DETRTransformer(nn.Module):
    def __init__(self, enc_depth: int, dec_depth: int, nheads: int, dim: int):
        super(DETRTransformer, self).__init__()
        
        self.encoders = nn.ModuleList(
            [DETREncoder(dim, nheads, dim * 4) for _ in range(enc_depth)]
        )
        self.decoders = nn.ModuleList(
            [DETRDecoder(dim, nheads,  dim * 4) for _ in range(dec_depth)]
        )
        
    def forward(self, x, t, pe):
        enc_out = x
        for encoder in self.encoders:
            enc_out = encoder(enc_out, pe)
        
        out = t
        for decoder in self.decoders:
            out = decoder(out, enc_out, pe)
        return out
    
class DETR(nn.Module):
    def __init__(self, backbone, backbone_dim: int, n_cls: int, n_query: int, dim: int, nheads: int, enc_depth: int, dec_depth):
        super().__init__()

        self.backbone = backbone
        self.conv = nn.Conv2d(backbone_dim, dim, 1)

        self.query_pos = nn.Parameter(torch.rand(n_query, dim))
        
        self.row_embed = nn.Parameter(torch.rand(n_query // 2, dim // 2))
        self.col_embed = nn.Parameter(torch.rand(n_query // 2, dim // 2))
        
        self.transformer = DETRTransformer(enc_depth, dec_depth, nheads, dim)

        self.ln_cls = nn.Linear(dim, n_cls + 1)
        self.ln_bbox = nn.Linear(dim, 4)

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        
        H, W = h.shape[-2:]
        
        pos = torch.cat([
                self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ], dim=-1).flatten(0, 1).unsqueeze(0)
        
        h = self.transformer(h.flatten(2).permute(0, 2, 1),
            self.query_pos.unsqueeze(0).repeat(x.shape[0], 1, 1), pos)
        
        out_cls = F.softmax(self.ln_cls(h), -1)
        out_bbox = F.sigmoid(self.ln_bbox(h))
        
        return out_cls, out_bbox