# -*- coding: utf-8 -*-
"""Swin Transformer 骨干（window attention + shifted window + patch merging，compact）。"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import BACKBONES
from ..layers import DropPath, LayerNorm2d, PatchMerging, WindowAttention
from ..layers.window_attention import window_partition, window_reverse
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
            LayerNorm2d(embed_dim, eps=1e-6),
        )
        stages.append(nn.Sequential(stem))
        prev = embed_dim
        n_blocks = sum(depths)
        block_idx = 0
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            layers: List[nn.Module] = []
            if i > 0:
                layers.append(PatchMerging(prev, prev * 2))
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
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)

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
        y = window_partition(y, ws)  # (B*nW, ws*ws, C)
        y = self.attn(y)
        y = window_reverse(y, ws, H + pad_h, W + pad_w)
        if pad_h or pad_w:
            y = y[:, :H, :W, :]
        if shift > 0:
            y = torch.roll(y, shifts=(shift, shift), dims=(1, 2))
        x = shortcut + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.permute(0, 3, 1, 2)


__all__ = ["SwinBackbone"]


__all__ = ["SwinBackbone"]
