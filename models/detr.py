import sys
sys.path.append('..')

import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import *
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
        
        idxs = idx.unsqueeze(1).repeat(1, n_query)
        arange = torch.arange(n_query)
        
        p_c = logits.unsqueeze(0).repeat(n_query, 1, 1)[arange, arange, idxs]
        
        return p_c
    
    def forward(self, idx, logits, gt_b, pd_b, n_obj_b):
        n_query = self.n_query
        
        gt_b = gt_b.unsqueeze(1).repeat(1, n_query, 1)
        pd_b = pd_b.unsqueeze(0).repeat(n_query, 1, 1)
    
        p_c = self.get_p_c(idx, logits)
        ls_box = self.get_box_loss(gt_b, pd_b, n_obj_b)
        
        ls_match = (1 - p_c) + ls_box
        
        row_idx, col_idx = linear_sum_assignment(ls_match.clone().detach().cpu())
        
        ls = - torch.log(p_c) + ls_box
        ls = ls[row_idx, col_idx]

        return ls

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
        
    def forward(self, x: Tensor, enc_out: Tensor, t, pe):

        out = self.ln1(x + self.mha1(x + t, x + t, x))
        out = self.ln2(out + self.mha2(q=out + t, k=enc_out + pe, v=enc_out))
        out = self.ln3(out + self.ff(out))
        
        return out
    
class DETRTransformer(nn.Module):
    def __init__(self, n_cls: int, enc_depth: int, dec_depth: int, nheads: int, dim: int):
        super(DETRTransformer, self).__init__()
        
        self.encoders = nn.ModuleList(
            [DETREncoder(dim, nheads, dim * 4) for _ in range(enc_depth)]
        )
        self.decoders = nn.ModuleList(
            [DETRDecoder(dim, nheads,  dim * 4) for _ in range(dec_depth)]
        )
        
        self.ff_cls = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, n_cls + 1),
        )
        self.ff_cls_ln = nn.LayerNorm(n_cls + 1)
        
        self.ff_bbox = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 4),
        )
        self.ff_bbox_ln = nn.LayerNorm(4)
        
    def forward(self, x, t, pe):
        enc_out = x
        for encoder in self.encoders:
            enc_out = encoder(enc_out, pe)
        
        outs = []
        out = t
        for decoder in self.decoders:
            out = decoder(out, enc_out, t, pe)
            cls_out = F.softmax(self.ff_cls_ln(self.ff_cls(out)), -1)
            bbox_out = F.sigmoid(self.ff_bbox_ln(self.ff_bbox(out)))
            outs.append((cls_out, bbox_out))
                    
        return outs    
    
class DETR(nn.Module):
    def __init__(self, backbone, backbone_dim: int, n_cls: int, n_query: int, dim: int, nheads: int, enc_depth: int, dec_depth):
        super().__init__()

        self.backbone = backbone
        self.conv = nn.Conv2d(backbone_dim, dim, 1)

        self.query_pos = nn.Parameter(torch.rand(n_query, dim))
        
        self.row_embed = nn.Parameter(torch.rand(n_query // 2, dim // 2))
        self.col_embed = nn.Parameter(torch.rand(n_query // 2, dim // 2))
        
        self.transformer = DETRTransformer(n_cls, enc_depth, dec_depth, nheads, dim)


    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        
        H, W = h.shape[-2:]
        
        pos = torch.cat([
                self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ], dim=-1).flatten(0, 1).unsqueeze(0)
        
        outs = self.transformer(h.flatten(2).permute(0, 2, 1),
            self.query_pos.unsqueeze(0).repeat(x.shape[0], 1, 1), pos)
        
        return outs