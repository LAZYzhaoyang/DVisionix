# -*- coding: utf-8 -*-
"""CSPDarknet 骨干（YOLOv5/v8 官方结构：Conv stem + CSP 阶段）。"""

from typing import List, Optional, Sequence

import torch.nn as nn

from ...registry import BACKBONES
from ..layers import ConvNormAct, CSPLayer
from .feature import FeatureBackboneBase


@BACKBONES.register()
@BACKBONES.register(name="cspdarknet_backbone")
class CSPDarknetBackbone(FeatureBackboneBase):
    """CSPDarknet（YOLOv5 风格）骨干：stem（6x6 stride2）+ 4 个 stage（3x3 stride2 + CSP）。

    features_only=True 时输出 stride 8/16/32/64 四个多尺度特征。
    """

    def __init__(
        self,
        depths: Sequence[int] = (3, 6, 9, 3),
        channels: Sequence[int] = (64, 128, 256, 512),
        in_channels: int = 3,
        features_only: bool = False,
        out_indices: Optional[Sequence[int]] = None,
    ):
        stages: List[nn.Module] = []
        stem = ConvNormAct(in_channels, 32, kernel_size=6, stride=2, padding=2, act="silu")
        stages.append(nn.Sequential(stem))
        prev = 32
        for depth, ch in zip(depths, channels):
            stage_layers = [ConvNormAct(prev, ch, kernel_size=3, stride=2, act="silu")]
            stage_layers.append(CSPLayer(ch, ch, num_blocks=depth, hidden_ratio=0.5, act="silu"))
            stages.append(nn.Sequential(*stage_layers))
            prev = ch
        super().__init__(
            stages, in_channels=in_channels, features_only=features_only, out_indices=out_indices
        )


__all__ = ["CSPDarknetBackbone"]
