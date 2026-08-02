# -*- coding: utf-8 -*-
"""CSPDarknet 骨干（YOLOv5/v8 官方结构：Conv stem + CSP 阶段）。"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ...registry import BACKBONES
from ..layers import ConvNormAct
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
            stage_layers.append(_CSP(ch, ch, depth))
            stages.append(nn.Sequential(*stage_layers))
            prev = ch
        super().__init__(
            stages, in_channels=in_channels, features_only=features_only, out_indices=out_indices
        )


class _CSP(nn.Module):
    """CSPDarknet 专用 CSP：1x1 分流 + N 个 Bottleneck(3x3, SiLU, 残差)。"""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3):
        super().__init__()
        hidden = max(1, out_channels // 2)
        self.main = ConvNormAct(in_channels, hidden, 1, act="silu")
        self.short = ConvNormAct(in_channels, hidden, 1, act="silu")
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    ConvNormAct(hidden, hidden, 1, act="silu"),
                    ConvNormAct(hidden, hidden, 3, stride=1, act="silu"),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.out = ConvNormAct(hidden * 2, out_channels, 1, act="silu")

    def forward(self, x):
        y1 = self.blocks(self.main(x))
        y2 = self.short(x)
        return self.out(torch.cat([y1, y2], dim=1))


__all__ = ["CSPDarknetBackbone"]
