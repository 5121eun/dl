#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn, Tensor

from models.layers import TransformerBlock

class Bert(nn.Module):
    def __init__(self, ntokens: int, ntimes: int, dim: int, depth: int, emb_mat: Tensor = None, **kwargs):
        super(Bert, self).__init__()
        
        if emb_mat is not None:
            self.tok_embed = nn.Embedding.from_pretrained(emb_mat, freeze=False)
        else:
            self.tok_embed = nn.Embedding(ntokens, dim)
            
        self.seg_embed = nn.Embedding(2, dim)
        self.pos_embed = nn.Embedding(2 * ntimes, dim)
        
        self.encoders = nn.ModuleList(
            [TransformerBlock(dim=dim, **kwargs) for _ in range(depth)]
        )
        
    def forward(self, x: Tensor, seg_ids: Tensor, **kwargs):
        b, n = x.shape
        
        out = self.tok_embed(x)
        pos = torch.arange(0, n, dtype=torch.long, device=x.device).unsqueeze(0)
        out = out + self.pos_embed(pos) + self.seg_embed(seg_ids)
        
        for encoder in self.encoders:
            out = encoder(out)
            
        return out

