# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: SwinV2 块（cosine attention + res-post-norm + 连续相对位置偏置）。
"""SwinV2 块（cosine attention + res-post-norm + 连续相对位置偏置）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import LAYERS
from .basic import DropPath
from .relative_position_bias import ContinuousRelativePositionBias
from .window_attention import window_partition, window_reverse


@LAYERS.register()
@LAYERS.register(name="swinv2_block")
class SwinV2Block(nn.Module):
    """SwinV2 块：cosine window 注意力 + 连续相对位置偏置 + res-post-norm + MLP。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.rpb = ContinuousRelativePositionBias(num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwinV2 块前向：x (B, L, C) -> 同形状输出。"""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        ws = min(self.window_size, H, W)
        shift = min(self.shift_size, ws) if self.shift_size > 0 else 0
        if shift > 0:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        x = window_partition(x, ws)  # (B*nW, N, C)
        x = x + self.drop_path(self.attn(self.norm1(x), ws))
        x = window_reverse(x, ws, H + pad_h, W + pad_w)
        if pad_h or pad_w:
            x = x[:, :H, :W, :]
        if shift > 0:
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x.permute(0, 3, 1, 2)

    def attn(self, t: torch.Tensor, ws: int) -> torch.Tensor:
        """窗口注意力前向：q/k/v -> 注意力输出。"""
        B, N, C = t.shape
        qkv = self.qkv(t).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # cosine attention
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)).clamp(min=-1.0, max=1.0)
        bias = self.rpb(ws)  # (H, N, N)
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, H, N, hd)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


__all__ = ["SwinV2Block"]
