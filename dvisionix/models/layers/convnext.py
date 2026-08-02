# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: ConvNeXt 块（LN + 深度可分离 + 层缩放 + DropPath）。
"""ConvNeXt 块（LN + 深度可分离 + 层缩放 + DropPath）。"""

import torch
import torch.nn as nn

from ...registry import LAYERS
from .basic import DropPath


@LAYERS.register()
@LAYERS.register(name="convnext_block")
class ConvNeXtBlock(nn.Module):
    """ConvNeXt 块：7x7 深度可分离 -> 逐通道 LN -> 1x1 扩展 -> GELU -> 1x1 收缩 -> 层缩放。

    Args:
        dim: 通道数。
        drop_path: DropPath 概率。
        layer_scale_init: 层缩放初值（<=0 禁用）。
    """

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ConvNeXt 块前向：x (B,C,H,W) -> 同形状输出。"""
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


__all__ = ["ConvNeXtBlock"]
