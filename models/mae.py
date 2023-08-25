import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from models.vit import *

def patch(x, p):
    b, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(b, h//p, p, w//p, p,  c).transpose(2, 3).reshape(b, -1, p, p, c)
    return x

def unpatch(x, h, w):
    b, n, p, _, c = x.shape
    x = x.reshape(b, h//p, w//p, p, p, c).transpose(2, 3).reshape(b, h, w, c)
    return x
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

    def forward(self, x: Tensor, unmask_idx: np.ndarray):
        b, c, h, w = x.shape
        p = self.patch_size

        x = patch(x, p)
        x = x[:, unmask_idx].reshape(b, -1, (p ** 2) * c)
        x = self.out(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, input_length: int, dim: int, depth: int, **kwargs):
        super(Encoder, self).__init__()

        self.cls = nn.Parameter(torch.randn(1, dim))
        self.pos_embed = nn.Parameter(torch.randn(input_length+1, dim))

        self.encoders = nn.ModuleList(
            [TransformerBlock(dim=dim, **kwargs) for _ in range(depth)]
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        cls = self.cls.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls, x), axis=1)
        x = x + self.pos_embed
        for encoder in self.encoders:
            x = encoder(x, **kwargs)

        return self.ln(x)

class Decoder(nn.Module):
    def __init__(self, input_length: int, dim: int, depth: int, patch_size: int, n_channels: int, **kwargs):
        super(Decoder, self).__init__()

        self.pos_embed = nn.Parameter(torch.randn(input_length, dim))

        self.decoders = nn.ModuleList(
            [TransformerBlock(dim=dim, **kwargs) for _ in range(depth)]
        )

        self.out = nn.Linear(dim, (patch_size ** 2) * n_channels)

    def forward(self, x):

        x = x + self.pos_embed

        for decoder in self.decoders:
            x = decoder(x)

        return F.sigmoid(self.out(x))

    
class MAE(nn.Module):
    def __init__(self, n_channels: int, patch_size: int, n_tokens: int, enc_dim: int, dec_dim: int, enc_depth: int, dec_depth: int, enc_nheads: int, dec_nheads: int, masking_ratio: float = 0.75, **kwargs):
        super(MAE, self).__init__()

        self.patch = PatchEmbedding(n_channels, patch_size, enc_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

        self.enc = Encoder(int(n_tokens * (1 - masking_ratio)), dim=enc_dim, depth=enc_depth, nheads=enc_nheads, hid_dim=enc_dim*4)
        self.enc_out = nn.Linear(enc_dim, dec_dim)

        self.dec = Decoder(n_tokens + 1, dec_dim, dec_depth, patch_size, n_channels, nheads=dec_nheads, hid_dim=dec_dim*4)

        self.patch_size = patch_size
        self.n_tokens = n_tokens


    def forward(self, x, unmask_idx:np.ndarray, **kwargs):
        b, c, _, _ = x.shape

        out = self.patch(x, unmask_idx)
        out = self.enc(out)
        out = self.enc_out(out)

        x = self.mask_token.repeat(b, self.n_tokens + 1, 1)
        x[:, [0, *unmask_idx]] = out
        out = self.dec(x)[:, 1:].reshape(b, self.n_tokens, self.patch_size, self.patch_size, c)

        return out
        