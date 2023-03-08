#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros((max_len, d_model))
        
        pos = torch.arange(0, max_len).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2)
        
        pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.shape[1]]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nheads: int):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % nheads == 0, 'd_model % nheads != 0'
        
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        
        self.out = nn.Linear(d_model, d_model, bias=False)
        
        self.out = nn.Linear(d_model, d_model, bias=False)
        
        self.nheads = nheads
        
    def forward(self, q:Tensor, k:Tensor, v: Tensor, mask=None):
        h = self.nheads
        b, qn, d = q.shape
        b, n, d = k.shape
        dk = d // h
        
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        
        q = q.view(b, qn, h, dk) # reshape b, qn, d -> b, qn, h, dk
        k, v = map(lambda x : x.view(b, n, h, dk), (k, v)) # reshape b, n, d -> b, n, h, dk
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # transpose b, n, h, dk -> b, h, n, dk
        
        attn = torch.matmul(q, k.transpose(2, 3)) / (dk ** 0.5)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = torch.matmul(attn, v)
        
        out = attn.transpose(1, 2).contiguous().view(b, qn, d)
        out = self.out(out)
        return out

class FeedForward(nn.Sequential):
    def __init__(self, d_model: int, d_ff: int):
        super(FeedForward, self).__init__(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

class TransformerEncoder(nn.Module):
    def __init__(self, d_model:int, nheads: int, d_ff: int):
        super(TransformerEncoder, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, nheads)
        self.ln1 = nn.LayerNorm(d_model)
        
        self.ff = FeedForward(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor):
        out = self.ln1(x + self.mha(x, x, x))
        out = self.ln2(out + self.ff(out))
        
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, nheads: int, d_ff: int):
        super(TransformerDecoder, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, nheads)
        self.ln1 = nn.LayerNorm(d_model)
        
        self.mha2 = MultiHeadAttention(d_model, nheads)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.ff = FeedForward(d_model, d_ff)
        self.ln3 = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor, enc_x: Tensor, mask=None):
        
        out = self.ln1(x + self.mha1(x, x, x, mask))
        out = self.ln2(out + self.mha2(q=out, k=enc_x, v=enc_x))
        out = self.ln2(out + self.ff(out))
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nheads: int, d_ff: int):
        super(TransformerBlock, self).__init__()
        
        self.encoder = TransformerEncoder(d_model, nheads, d_ff)
        self.decoder = TransformerDecoder(d_model, nheads, d_ff)
        
    def forward(self, x: Tensor, t: Tensor, mask=None):
        enc_out = self.encoder(x)
        out = self.decoder(t, enc_out, mask)
        return out

class Transformer(nn.Module):
    def __init__(self, max_len: int, ntokens: int, d_model: int, nheads: int, d_ff: int, nlayers: int):
        super(Transformer, self).__init__()
        
        self.emb = nn.Embedding(ntokens, d_model)
        self.pe = PositionalEncoding(max_len, d_model)
        
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, nheads, d_ff) for _ in range(nlayers)]
        )
        self.linear = nn.Linear(d_model, ntokens)
        
        self.init_weight()
        
        self.d_model = d_model
    
    def init_weight(self):
        self.linear.weight = self.emb.weight
        
    def forward(self, x: Tensor, t: Tensor, mask=None):
        x = self.emb(x) * (self.d_model ** 0.5)
        x = self.pe(x)
        
        t = self.emb(t) * (self.d_model ** 0.5)
        t = self.pe(t)
        
        for layer in self.layers:
            out = layer(x, t, mask)
        
        out = self.linear(out)
        
        return out
    
    @torch.no_grad()
    def generate(self, x: Tensor, token: Tensor, seq_len: int, mask=None):      
        for i in range(seq_len):
            out = self(x, token, mask[:i + 1, :i + 1] if mask is not None else None)
            out = F.softmax(out, dim=-1)
            out = out.argmax(dim=-1)
            token = torch.cat((token[:, :1], out), dim=1)
        return out
    
    def get_seq_mask(self, seq_len: int):
        return torch.ones((seq_len, seq_len)).tril()

