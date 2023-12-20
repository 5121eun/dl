import torch
from torch import nn
import torch.nn.functional as F

from models.layers import MultiHeadAttention, FeedForward
from scipy.optimize import linear_sum_assignment
from commons.utils import get_giou, cxcywh_to_xyxy

class DETRLoss(nn.Module):
    def __init__(self, n_query: int, l_giou: float, l_box: float, ls_giou_w: float = 20):
        super(DETRLoss, self).__init__()
        
        self.n_query = n_query
        self.l_giou = l_giou
        self.l_box = l_box
        self.ls_giou_w = ls_giou_w
        
    def get_box_loss(self, gt_b, pd_b, n_obj_b):
        gt_b_xyxy, pd_b_xyxy = [cxcywh_to_xyxy(b) for b in [gt_b, pd_b]]

        ls_giou = get_giou(gt_b_xyxy, pd_b_xyxy) / n_obj_b
        
        ls_l1 = torch.abs(gt_b - pd_b).sum(-1) / n_obj_b
        
        ls_box = (self.l_giou * ls_giou) * self.ls_giou_w + (self.l_box * ls_l1)
        return ls_box
        
    def get_p_c(self, idx, logits):
        n_query = self.n_query
        
        idx = idx.unsqueeze(-1).repeat(1, 1, n_query)
        arange = torch.arange(n_query)
        p = logits.unsqueeze(1).repeat(1, n_query, 1, 1)
        p_c = torch.stack([p[b, arange, arange, idx[b]] for b in range(len(idx))], 0)
        
        return p_c
    
    def forward(self, idx, logits, gt_b, pd_b, no_obj_c, n_obj_b):
        n_query = self.n_query
        batch_size = len(idx)
        
        p_c = self.get_p_c(idx, logits)

        gt_b = gt_b.unsqueeze(2).repeat(1, 1, n_query, 1)
        pd_b = pd_b.unsqueeze(1).repeat(1, n_query, 1, 1)
    
        ls_box = self.get_box_loss(gt_b, pd_b, n_obj_b)
        
        ls_match = (1 - p_c) + ls_box
        
        match_idx = [linear_sum_assignment(ls_match[b][:len(idx[b][idx[b]<no_obj_c])].clone().detach().cpu()) for b in range(batch_size)]
        
        ls = - torch.log(p_c) + ls_box
        
        loss = 0
        for i, (row_idx, col_idx) in enumerate(match_idx):
            loss_c = ls[i, row_idx, col_idx]
            col_idx_no_c = list(set(range(n_query)) - set(col_idx))
            row_idx_no_c = list(set(range(n_query)) - set(row_idx))
            loss_no_c = - torch.log(p_c[i, row_idx_no_c, col_idx_no_c]) / 10
            loss += loss_c.sum() + loss_no_c.sum()
        return loss

class DETREncoder(nn.Module):
    def __init__(self, d_model: int, nheads: int, d_ff: int, window_size: int):
        super(DETREncoder, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, nheads, window_size)
        self.ln1 = nn.LayerNorm(d_model)
        
        self.ff = FeedForward(d_model, d_ff, nn.ReLU())
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x, pe, window=False):

        out = self.ln1(x + self.mha(x + pe, x + pe, x, window=window))
        out = self.ln2(out + self.ff(out))
        
        return out
        

class DETRDecoder(nn.Module):
    def __init__(self, d_model: int, nheads: int, d_ff: int):
        super(DETRDecoder, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, nheads)
        self.ln1 = nn.LayerNorm(d_model)
        
        self.mha2 = MultiHeadAttention(d_model, nheads)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.ff = FeedForward(d_model, d_ff, nn.ReLU())
        self.ln3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_out, t, pe):

        out = self.ln1(x + self.mha1(x + t, x + t, x))
        out = self.ln2(out + self.mha2(q=out + t, k=enc_out + pe, v=enc_out))
        out = self.ln3(out + self.ff(out))
        
        return out
    
class DETRTransformer(nn.Module):
    def __init__(self, n_cls: int, enc_depth: int, dec_depth: int, nheads: int, dim: int, window_size: int, global_attn_iter: int):
        super(DETRTransformer, self).__init__()
        
        self.encoders = nn.ModuleList(
            [DETREncoder(dim, nheads, dim * 4, window_size) for _ in range(enc_depth)]
        )
        self.decoders = nn.ModuleList(
            [DETRDecoder(dim, nheads,  dim * 4) for _ in range(dec_depth)]
        )

        self.global_attn_iter = global_attn_iter
                
    def forward(self, x, t, pe):
        enc_out = x
        for i, enc in enumerate(self.encoders):
            enc_out = enc(enc_out, pe, (i + 1) % self.global_attn_iter != 0)
        
        out = t
        for decoder in self.decoders:
            out = decoder(out, enc_out, t, pe)

        return out   
    
class DETR(nn.Module):
    def __init__(self, backbone, backbone_dim: int, n_cls: int, x_len: int, n_query: int, dim: int, nheads: int, enc_depth: int, dec_depth, window_size=0, global_attn_iter=1):
        super().__init__()

        self.backbone = backbone
        self.conv = nn.Conv2d(backbone_dim, dim, 1)

        self.query_pos = nn.Parameter(torch.rand(n_query, dim))
        
        self.row_embed = nn.Parameter(torch.rand(x_len // 2, dim // 2))
        self.col_embed = nn.Parameter(torch.rand(x_len // 2, dim // 2))
        
        self.transformer = DETRTransformer(n_cls, enc_depth, dec_depth, nheads, dim, window_size, global_attn_iter)
        
        self.ff_cls = nn.Linear(dim, n_cls + 1)
        self.ff_cls_ln = nn.LayerNorm(n_cls + 1)
        
        self.ff_bbox = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, 4),
        )
        self.ff_bbox_ln = nn.LayerNorm(4)

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        
        H, W = h.shape[-2:]
        
        pos = torch.cat([
                self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ], dim=-1).flatten(0, 1).unsqueeze(0)

        out = self.transformer(h.flatten(2).permute(0, 2, 1),
            self.query_pos.unsqueeze(0).repeat(x.shape[0], 1, 1), pos)
        
        cls_out = F.softmax(self.ff_cls_ln(self.ff_cls(out)), -1)
        bbox_out = F.sigmoid(self.ff_bbox_ln(self.ff_bbox(out)))
        
        return cls_out, bbox_out