# -*- coding: utf-8 -*-
"""ConvNeXtV2 骨干（GRN 全局响应归一化）。"""

from typing import List, Optional, Sequence

import torch.nn as nn

from ...registry import BACKBONES
from ..layers import ConvNeXtV2Block, LayerNorm2d
from .feature import FeatureBackboneBase


@BACKBONES.register()
@BACKBONES.register(name="convnextv2_backbone")
class ConvNeXtV2Backbone(FeatureBackboneBase):
    """ConvNeXtV2 骨干：stem（4x4 stride4）+ 4 个 stage（2x2 stride2 + ConvNeXtV2Block，GRN 替代 LayerScale）。"""

    def __init__(
        self,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        in_channels: int = 3,
        features_only: bool = False,
        out_indices: Optional[Sequence[int]] = None,
        drop_path_rate: float = 0.0,
    ):
        stages: List[nn.Module] = []
        prev = dims[0]
        n_blocks = sum(depths)
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            layers = []
            if i == 0:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, dim, kernel_size=4, stride=4, padding=1),
                        LayerNorm2d(dim, eps=1e-6),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        LayerNorm2d(prev, eps=1e-6),
                        nn.Conv2d(prev, dim, kernel_size=2, stride=2),
                    )
                )
            for j in range(depth):
                dpr = drop_path_rate * (sum(depths[:i]) + j) / max(n_blocks - 1, 1)
                layers.append(ConvNeXtV2Block(dim, drop_path=dpr))
            stages.append(nn.Sequential(*layers))
            prev = dim
        super().__init__(
            stages, in_channels=in_channels, features_only=features_only, out_indices=out_indices
        )


__all__ = ["ConvNeXtV2Backbone"]
