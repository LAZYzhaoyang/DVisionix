# -*- coding: utf-8 -*-
"""ConvNeXt 骨干网络（现代 CNN，LN + 深度可分离 + 层缩放）。"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ...registry import BACKBONES
from ..layers import ConvNeXtBlock
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
                        _LayerNorm2d(dim, eps=1e-6),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        _LayerNorm2d(prev, eps=1e-6),
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


__all__ = ["ConvNeXtBackbone"]
