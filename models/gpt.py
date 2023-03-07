#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
from torch import nn, Tensor

from models.layers import TransformerBlock

class GPT(nn.Module):
    def __init__(self, ntokens: int, ntimes: int, dim: int, nheads: int, hid_dim: int, depth: int, dropout = 0., **kwargs):
        super(GPT, self).__init__()
        
        self.embed = nn.Embedding(ntokens, dim)
        self.pos_embed = nn.Embedding(ntimes, dim)
        
        self.decoders = nn.ModuleList(
            [TransformerBlock(dim, nheads, hid_dim, dropout, **kwargs) for _ in range(depth)]
        )
                
    def forward(self, x: Tensor, **kwargs):
        b, n = x.shape
        pos = torch.arange(0, n, dtype=torch.long, device=x.device).unsqueeze(0)
        
        out = self.embed(x)
        pos_out = self.pos_embed(pos)
        out = out + pos_out
        
        sparse_masks = kwargs['sparse_masks']
        del kwargs['sparse_masks']
        
        for i, decoder in enumerate(self.decoders):
            if i % 2 == 0:
                sps_mask = None
            else:
                sps_mask = sparse_masks
            
            if i != len(self.decoders) - 1:
                out = decoder(out, sparse_masks=sps_mask, **kwargs)
            else:
                out = decoder(out, is_final=True, sparse_masks=sps_mask, **kwargs)
                        
        return out
