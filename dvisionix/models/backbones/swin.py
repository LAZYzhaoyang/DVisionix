# -*- coding: utf-8 -*-
"""Swin Transformer 骨干（window attention + shifted window + patch merging，compact）。"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import BACKBONES
from .feature import FeatureBackboneBase


@BACKBONES.register()
@BACKBONES.register(name="swin_backbone")
class SwinBackbone(FeatureBackboneBase):
    """Swin-lite 骨干：patch embed（4x4 stride4）+ 4 个 stage（PatchMerging + N 个 SwinBlock）。

    features_only=True 输出 stride 4/8/16/32 四个多尺度特征。
    说明：shifted window 采用 cyclic shift + 反向 shift 的 compact 近似（未加严格 attention mask）。
    """

    def __init__(
        self,
        embed_dim: int = 64,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (2, 4, 8, 16),
        window_size: int = 4,
        in_channels: int = 3,
        features_only: bool = False,
        out_indices: Optional[Sequence[int]] = None,
        drop_path_rate: float = 0.0,
    ):
        stages: List[nn.Module] = []
        stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4),
            _LayerNorm2d(embed_dim, eps=1e-6),
        )
        stages.append(nn.Sequential(stem))
        prev = embed_dim
        n_blocks = sum(depths)
        block_idx = 0
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            layers: List[nn.Module] = []
            if i > 0:
                layers.append(_PatchMerging(prev, prev * 2))
                prev = prev * 2
            for j in range(depth):
                dpr = drop_path_rate * block_idx / max(n_blocks - 1, 1)
                shift = 0 if j % 2 == 0 else window_size // 2
                layers.append(_SwinBlock(prev, heads, window_size, shift_size=shift, drop_path=dpr))
                block_idx += 1
            stages.append(nn.Sequential(*layers))
        super().__init__(
            stages, in_channels=in_channels, features_only=features_only, out_indices=out_indices
        )


class _PatchMerging(nn.Module):
    """2x2 patch 合并：通道 x4 -> Linear -> 2x，空间减半。"""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))
        if W % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0))
        _, _, H, W = x.shape
        x = (
            x.view(B, C, H // 2, 2, W // 2, 2)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(B, H // 2, W // 2, 4 * C)
        )
        x = self.norm(x)
        x = self.reduction(x)
        return x.permute(0, 3, 1, 2)  # (B, out_dim, H/2, W/2)


class _SwinBlock(nn.Module):
    """Swin 块：LayerNorm -> window/shifted-window 注意力 -> LayerNorm -> MLP。"""

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
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.drop_path = nn.Identity() if drop_path == 0 else _DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        shortcut = x
        y = self.norm1(x)
        # 自适应窗口：特征图小于配置窗口时按实际尺寸
        ws = min(self.window_size, H, W)
        shift = min(self.shift_size, ws) if self.shift_size > 0 else 0
        if shift > 0:
            y = torch.roll(y, shifts=(-shift, -shift), dims=(1, 2))
        # 尺寸不能整除窗口时补零，处理后裁剪
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            y = F.pad(y, (0, 0, 0, pad_w, 0, pad_h))
        y = _window_partition(y, ws)  # (B*nW, ws*ws, C)
        y = self.attn(y)
        y = _window_reverse(y, ws, H + pad_h, W + pad_w)
        if pad_h or pad_w:
            y = y[:, :H, :W, :]
        if shift > 0:
            y = torch.roll(y, shifts=(shift, shift), dims=(1, 2))
        x = shortcut + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.permute(0, 3, 1, 2)


class _WindowAttention(nn.Module):
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


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return x


def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] // (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


class _DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x / keep * mask


__all__ = ["SwinBackbone"]


class _LayerNorm2d(nn.Module):
    """逐通道 LayerNorm（channels_first，(B, C, H, W) 上对每个通道做 LN）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


__all__ = ["SwinBackbone"]
