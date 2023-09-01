import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import *
from scipy.optimize import linear_sum_assignment

def get_loss(no_c, logits, idxs_c, bboxes_tg, bboxes_y, l_iou, l_box):
    idxs_c = idxs_c[idxs_c!=no_c]
    n_cls_bboxes = len(idxs_c)

    logits_c = logits.unsqueeze(0).repeat(n_cls_bboxes, 1, 1)
    logits_c = logits_c[list(range(n_cls_bboxes)), :, idxs_c]
    
    bboxes_tg = bboxes_tg[bboxes_tg != -1].view(-1, 4)
    n_query = bboxes_y.shape[0]
    
    bboxes_tg = bboxes_tg.unsqueeze(1).repeat(1, n_query, 1)
    bboxes_y = bboxes_y.unsqueeze(0).repeat(bboxes_tg.shape[0], 1, 1)
    
    ls_box = get_box_loss(bboxes_tg , bboxes_y, l_iou, l_box)
    ls_match = - logits_c + ls_box
    
    row_idx, col_idx = linear_sum_assignment(ls_match.detach().numpy())
    
    ls = - torch.log(logits_c) + ls_box
    ls_c = ls[row_idx, col_idx].sum()
    
    no_c_idxs = list(set(range(n_query)) - set(col_idx))
    ls_no_c = - logits[no_c_idxs, -1].sum() / 10
    
    return ls_c + ls_no_c

def get_iou(bboxes_tg, bboxes_y):
    
    bboxes_tg_x1, bboxes_tg_x2 = bboxes_tg[:, :, 0], bboxes_tg[:, :, 0] + bboxes_tg[:, :, 2]
    bboxes_y_x1, bboxes_y_x2 = bboxes_y[:, :, 0], bboxes_y[:, :, 0] + bboxes_y[:, :, 2]

    x1 = torch.stack([bboxes_tg_x1, bboxes_y_x1], dim=-1)
    x2 = torch.stack([bboxes_tg_x2, bboxes_y_x2], dim=-1)

    bboxes_tg_y1, bboxes_tg_y2 = bboxes_tg[:, :, 1], bboxes_tg[:, :, 1] + bboxes_tg[:, :, 3]
    bboxes_y_y1, bboxes_y_y2 = bboxes_y[:, :, 1], bboxes_y[:, :, 1] + bboxes_y[:, :, 3]

    y1 = torch.stack([bboxes_tg_y1, bboxes_y_y1], dim=-1)
    y2 = torch.stack([bboxes_tg_y2, bboxes_y_y2], dim=-1)
    
    inter_x1 = torch.max(x1, dim=-1).values
    inter_x2= torch.min(x2, dim=-1).values

    inter_y1 = torch.max(y1, dim=-1).values
    inter_y2= torch.min(y2, dim=-1).values
    
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0
    
    inter = inter_w * inter_h
    
    bboxes_tg_area = bboxes_tg[:, :, 2] * bboxes_tg[:, :, 3]
    bboxes_y_area = bboxes_y[:, :, 2] * bboxes_y[:, :, 3]

    union = (bboxes_tg_area + bboxes_y_area) - inter
    iou = inter / union
    
    b_x1 = torch.min(x1, dim=-1).values
    b_x2= torch.max(x2, dim=-1).values

    b_y1 = torch.min(y1, dim=-1).values
    b_y2= torch.max(y2, dim=-1).values
    
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    
    return 1 - iou - ((b_area - union) / b_area)

def get_box_loss(bboxes_tg, bboxes_y, l_iou, l_box):
    n_cls_bboxes = bboxes_tg.shape[0]
    
    ls_iou = get_iou(bboxes_tg, bboxes_y) / n_cls_bboxes
    ls_l1 = torch.abs(bboxes_tg - bboxes_y).sum(-1) / n_cls_bboxes
    ls_box = (l_iou * ls_iou) + (l_box * ls_l1)
    
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