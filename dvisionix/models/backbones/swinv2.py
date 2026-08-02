# -*- coding: utf-8 -*-
"""SwinV2 骨干（cosine attention + 连续相对位置偏置 + res-post-norm）。"""

from typing import List, Optional, Sequence

import torch.nn as nn

from ...registry import BACKBONES
from ..layers import LayerNorm2d, PatchMerging, SwinV2Block
from .feature import FeatureBackboneBase


@BACKBONES.register()
@BACKBONES.register(name="swinv2_backbone")
class SwinV2Backbone(FeatureBackboneBase):
    """SwinV2-lite：patch embed（4x4 stride4）+ 4 个 stage（PatchMerging + SwinV2Block）。

    SwinV2 改进：cosine attention、log-spaced 连续相对位置偏置、res-post-norm。
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
                layers.append(
                    SwinV2Block(prev, heads, window_size, shift_size=shift, drop_path=dpr)
                )
                block_idx += 1
            stages.append(nn.Sequential(*layers))
        super().__init__(
            stages, in_channels=in_channels, features_only=features_only, out_indices=out_indices
        )


__all__ = ["SwinV2Backbone"]
