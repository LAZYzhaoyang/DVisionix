# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: ViT 骨干网络（patch embed + Transformer encoder，单尺度输出）。
"""ViT 骨干网络（patch embed + Transformer encoder，单尺度输出）。"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ...registry import BACKBONES
from ..layers import PositionEmbeddingSine
from .feature import FeatureBackboneBase


@BACKBONES.register()
@BACKBONES.register(name="vit_backbone")
class ViTBackbone(FeatureBackboneBase):
    """ViT 骨干：patch embed（4x4 stride4）-> Transformer encoder -> LayerNorm -> 网格特征。

    features_only=True 输出单尺度 (B, embed_dim, H/p, W/p)；正弦位置编码支持任意输入尺寸。
    单尺度输出配合 FPN/PANet 或单尺度检测/分割头即插即用。
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
        features_only: bool = False,
        out_indices: Optional[Sequence[int]] = None,
        drop_path_rate: float = 0.0,
    ):
        stages: List[nn.Module] = []
        blocks = [
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_path_rate if drop_path_rate < 0.1 else 0.0,
                batch_first=True,
            )
            for _ in range(depth)
        ]
        stage = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            _AddPosEmbed(embed_dim),
            _PatchTokens(embed_dim),
            *blocks,
            nn.LayerNorm(embed_dim),
            _TokensToGrid(embed_dim),
        )
        stages.append(stage)
        super().__init__(
            stages, in_channels=in_channels, features_only=features_only, out_indices=out_indices
        )


class _AddPosEmbed(nn.Module):
    """在 4D 特征图上叠加正弦位置编码（支持任意输入尺寸）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.pe = PositionEmbeddingSine(dim // 2, normalize=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """在 4D 特征图上叠加正弦位置编码。"""
        return x + self.pe(x)


class _PatchTokens(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, HW, C) 的 token 序列。"""
        return x.flatten(2).permute(0, 2, 1)  # (B, HW, C)


class _TokensToGrid(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, C) token 序列 -> (B, C, H, W) 特征图。"""
        B, N, C = x.shape
        h = w = int(N**0.5)
        return x.permute(0, 2, 1).reshape(B, C, h, w)


__all__ = ["ViTBackbone"]
