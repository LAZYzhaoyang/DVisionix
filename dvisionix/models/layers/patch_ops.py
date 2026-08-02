# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: Patch 操作：PatchMerging（Swin 降采样）与 PatchExpand（Swin-UNet 上采样）。
"""Patch 操作：PatchMerging（Swin 降采样）与 PatchExpand（Swin-UNet 上采样）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import LAYERS


@LAYERS.register()
@LAYERS.register(name="patch_merging")
class PatchMerging(nn.Module):
    """2x2 patch 合并：通道 x4 -> LayerNorm -> Linear -> 2x，空间减半。"""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PatchMerging：x (B,C,H,W) -> (B,2C,H/2,W/2)。"""
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


@LAYERS.register()
@LAYERS.register(name="patch_expand")
class PatchExpand(nn.Module):
    """Swin-UNet PatchExpand：LayerNorm -> Linear(2x) -> PixelShuffle(2)（通道减半、空间翻倍）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PatchExpand：x (B,C,H,W) -> (B,C/2,2H,2W)。"""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.expand(x)  # (B, H, W, 2C)
        x = x.permute(0, 3, 1, 2)  # (B, 2C, H, W)
        x = F.pixel_shuffle(x, 2)  # (B, C//2, 2H, 2W)
        return x


__all__ = ["PatchMerging", "PatchExpand"]
