#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
from torch import nn, Tensor

from models.layers import TransformerBlock

class GPT(nn.Module):
    def __init__(self, ntokens: int, ntimes: int, dim: int, nheads: int, hid_dim: int, depth: int, emb_mat: Tensor = None, dropout = 0.):
        super(GPT, self).__init__()
        
        if emb_mat is not None:
            self.embed = nn.Embedding.from_pretrained(emb_mat, freeze=False)
        else:
            self.embed = nn.Embedding(ntokens, dim)

        self.pos_embed = nn.Embedding(ntimes, dim)
        
        self.decoders = nn.ModuleList(
            [TransformerBlock(dim, nheads, hid_dim, dropout) for _ in range(depth)]
        )
                
    def forward(self, x: Tensor, **kwargs):
        b, n = x.shape
        pos = torch.arange(0, n, dtype=torch.long, device=x.device).unsqueeze(0)
        
        out = self.embed(x)
        pos_out = self.pos_embed(pos)
        out = out + pos_out
        
        for decoder in self.decoders:
            out = decoder(out, **kwargs)
        
        return out
    
#     @torch.no_grad()
#     def generate(self, token, ntimes: int):
#         for ntimes in range(ntimes):
#             out = self(token)
#             out = F.softmax(out, dim=-1)
#             out = out.argmax(dim=-1)
            
#             token = torch.cat((token[:, :1], out), dim=1)
#         return out
