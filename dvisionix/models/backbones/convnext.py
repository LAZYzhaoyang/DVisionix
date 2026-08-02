# -*- coding: utf-8 -*-
"""ConvNeXt 骨干网络（现代 CNN，LN + 深度可分离 + 层缩放）。"""

from typing import List, Optional, Sequence

import torch.nn as nn

from ...registry import BACKBONES
from ..layers import ConvNeXtBlock, LayerNorm2d
from .feature import FeatureBackboneBase


@BACKBONES.register()
@BACKBONES.register(name="convnext_backbone")
class ConvNeXtBackbone(FeatureBackboneBase):
    """ConvNeXt 骨干：stem（4x4 stride4）+ 4 个 stage（2x2 stride2 下采样 + N 个 ConvNeXtBlock）。

    features_only=True 时输出 stride 4/8/16/32 四个多尺度特征。
    """

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
                layers.append(ConvNeXtBlock(dim, drop_path=dpr))
            stages.append(nn.Sequential(*layers))
            prev = dim
        super().__init__(
            stages, in_channels=in_channels, features_only=features_only, out_indices=out_indices
        )


__all__ = ["ConvNeXtBackbone"]


__all__ = ["ConvNeXtBackbone"]
