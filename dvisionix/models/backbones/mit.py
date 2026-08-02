# -*- coding: utf-8 -*-
"""SegFormer 编码器（MiTBackbone）：overlap patch embed + MixFFN 多尺度输出。"""

from typing import List, Optional, Sequence

import torch.nn as nn

from ...registry import BACKBONES
from ..layers import LayerNorm2d, MixFFN
from .feature import FeatureBackboneBase


@BACKBONES.register()
@BACKBONES.register(name="mit_backbone")
class MiTBackbone(FeatureBackboneBase):
    """SegFormer MiT 编码器：overlap patch embed（首层 7x7 stride4，后续 3x3 stride2）+ MixFFN 块。

    features_only=True 输出 stride 4/8/16/32 四个多尺度特征（配合 SegFormer / 任意分割头即插即用）。
    """

    def __init__(
        self,
        embed_dims: Sequence[int] = (32, 64, 160, 256),
        depths: Sequence[int] = (2, 2, 2, 2),
        in_channels: int = 3,
        features_only: bool = False,
        out_indices: Optional[Sequence[int]] = None,
        mlp_ratios: Sequence[int] = (4, 4, 4, 4),
    ):
        stages: List[nn.Module] = []
        prev = in_channels
        for i, (dim, depth, ratio) in enumerate(zip(embed_dims, depths, mlp_ratios)):
            layers: List[nn.Module] = []
            if i == 0:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(prev, dim, kernel_size=7, stride=4, padding=3),
                        LayerNorm2d(dim),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(prev, dim, kernel_size=3, stride=2, padding=1),
                        LayerNorm2d(dim),
                    )
                )
            for _ in range(depth):
                layers.append(MixFFN(dim, expand=ratio))
            stages.append(nn.Sequential(*layers))
            prev = dim
        super().__init__(
            stages, in_channels=in_channels, features_only=features_only, out_indices=out_indices
        )


__all__ = ["MiTBackbone"]
