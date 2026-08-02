# -*- coding: utf-8 -*-
"""ConvNeXtV2 块（GRN 全局响应归一化，替代 LayerScale）。"""

import torch
import torch.nn as nn

from ...registry import LAYERS
from .basic import DropPath
from .grn import GRN


@LAYERS.register()
@LAYERS.register(name="convnextv2_block")
class ConvNeXtV2Block(nn.Module):
    """ConvNeXtV2 块：7x7 DWConv -> LayerNorm2d -> 1x1 扩展 -> GELU -> 1x1 收缩 -> GRN -> DropPath。"""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)  # channels_last token LN
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.grn = GRN(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        B, H, W, C = x.shape
        x = self.grn(x.reshape(B, H * W, C)).reshape(B, H, W, C)  # GRN 在 (B, N, C) 上计算
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


__all__ = ["ConvNeXtV2Block"]
