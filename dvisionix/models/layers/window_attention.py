# -*- coding: utf-8 -*-
"""Swin 窗口注意力：WindowAttention + 窗口划分/还原工具。"""

import torch
import torch.nn as nn

from ...registry import LAYERS


@LAYERS.register()
@LAYERS.register(name="window_attention")
class WindowAttention(nn.Module):
    """窗口多头注意力（Swin 家族共享）。输入 (B, N, C)（N = 窗口内 token 数）。"""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, H, N, hd)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, ws*ws, C)。"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return x


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws*ws, C) -> (B, H, W, C)。"""
    B = int(windows.shape[0] // (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


__all__ = ["WindowAttention", "window_partition", "window_reverse"]
