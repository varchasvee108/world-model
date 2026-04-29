import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class FiLM(nn.Module):
    def __init__(self, emb_dim, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels * 2))

    def forward(self, x, emb):
        gamma_beta = self.mlp(emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = rearrange(gamma, "b c -> b c 1 1")
        beta = rearrange(beta, "b c -> b c 1 1")
        return x * (1 + gamma) + beta


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.film = FiLM(emb_dim, out_ch)
        self.res_conv = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.film(h, emb)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels**-0.5

    def forward(self, x):
        b, c, h, w = x.shape
        h_norm = self.norm(x)
        qkv = self.qkv(h_norm)
        q, k, v = rearrange(qkv, "b (qkv c) h w -> qkv b (h w) c", qkv=3)
        attn = torch.einsum("b n c, b m c -> b n m", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("b n m, b m c -> b n c", attn, v)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
