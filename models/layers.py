#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F

class FeedForward(nn.Sequential):
    def __init__(self, dim: int, hid_dim: int, func: nn.Module, dropout = 0.):
        super(FeedForward, self).__init__(
            nn.Linear(dim, hid_dim),
            func,
            nn.Dropout(dropout),
            nn.Linear(hid_dim, dim),
            nn.Dropout(dropout)
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, nheads: int, dropout = 0., final_ln=False):
        super(MultiHeadAttention, self).__init__()
        
        assert dim % nheads == 0, 'dim % nheads != 0'
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        if final_ln:
            self.ln = nn.LayerNorm(dim // nheads)
        
        self.out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )
        
        self.nheads = nheads
        self.final_ln = final_ln
        
    def forward(self, x, seq_mask=None, sparse_masks=None, is_final=False):
        h = self.nheads
        b, n, d = x.shape
        dk = d // h
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        q, k, v = map(lambda x : x.view(b, n, h, dk), (q, k, v))
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(2, 3)) / (dk ** 0.5)
        
        if seq_mask is not None:
            attn = attn.masked_fill(seq_mask == 0, -1e9)
        
        if sparse_masks is not None:
            attn[:, 0::2] = attn[:, 0::2].masked_fill(sparse_masks[0] == 0, -1e9)
            attn[:, 1::2] = attn[:, 1::2].masked_fill(sparse_masks[1] == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        attn = torch.matmul(attn, v)
        
        if self.final_ln & is_final:
            attn = self.ln(attn)
        
        out = attn.transpose(1, 2).contiguous().view(b, n, d)
        out = self.out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, nheads: int, hid_dim: int, dropout = 0., **kwargs):
        super(TransformerBlock, self).__init__()
        
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, nheads, dropout, **kwargs)
        
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hid_dim, nn.GELU(), dropout)
        
    def forward(self, x, **kwargs):
        
        out = self.attn(self.ln1(x), **kwargs) + x
        out = self.ff(self.ln2(out)) + out
        
        return out

class WindowMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nheads: int, window_size = 0):
        super(WindowMultiHeadAttention, self).__init__()
        
        assert d_model % nheads == 0, 'd_model % nheads != 0'
        
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        
        self.out = nn.Linear(d_model, d_model, bias=False)
        
        self.nheads = nheads
        self.window_size = window_size
        
    def forward(self, q, k, v, mask=None, window=False):
        h = self.nheads
        w = self.window_size
        b, qn, d = q.shape
        b, n, d = k.shape
        dk = d // h
        
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.view(b, qn, h, dk) # reshape b, qn, d -> b, qn, h, dk
        k, v = map(lambda x : x.view(b, n, h, dk), (k, v)) # reshape b, n, d -> b, n, h, dk
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # transpose b, n, h, dk -> b, h, n, dk
        
        if w > 0 and window is not False:
            q, k, v = q.view(b, h, n // w, n // w, dk), k.view(b, h, n // w, n // w, dk), v.view(b, h, n // w, n // w, dk)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = torch.matmul(attn, v)
        
        out = attn.transpose(1, -2).contiguous().view(b, qn, d)
        out = self.out(out)
        return out
