#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn, Tensor

from models.layers import TransformerBlock

class PatchEmbedding(nn.Module):
    def __init__(self, nchannels: int, patch_size: int, dim: int):
        super(PatchEmbedding, self).__init__()
        
        patch_dim = (patch_size ** 2) * nchannels
        self.patch_size = patch_size
        self.out = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x: Tensor):
        p = self.patch_size
        b, c, h, w  = x.shape
        n = (h * w) // (p ** 2)
        assert (h * w) % (p ** 2) == 0, '(h * w) % (p ** 2) != 0'
        
        x = x.transpose(1, 3).transpose(1, 2)
        x = x.reshape(b, h//p, p, w//p, p,  c).transpose(2, 3).reshape(b, n, (p ** 2) * c)
        
        return self.out(x)

class ViT(nn.Module):
    def __init__(self, nchannels: int, patch_size: int, img_size: int, dim: int, depth: int, **kwargs):
        super(ViT, self).__init__()
        
        n = (img_size ** 2) // (patch_size ** 2)
        
        self.patch_embed = PatchEmbedding(nchannels, patch_size, dim)
        self.cls = nn.Parameter(torch.randn(1, dim))
        self.pos_embed = nn.Parameter(torch.randn(n + 1, dim))
        
        self.encoders = nn.ModuleList(
            [TransformerBlock(dim=dim, **kwargs) for _ in range(depth)]
        )
                
    def forward(self, x: Tensor, **kwargs):
        b, h, w, c = x.shape
        
        out = self.patch_embed(x)
        cls = self.cls.repeat(b, 1, 1)
        out = torch.cat((cls, out), axis=1)
        out = out + self.pos_embed
        
        for encoder in self.encoders:
            out = encoder(out, **kwargs)

        return out
