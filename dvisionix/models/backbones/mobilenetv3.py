# -*- coding: utf-8 -*-
"""MobileNetV3 轻量骨干（MBConv 倒残差 + SE，适合移动/实时）。"""

from typing import List, Optional, Sequence, Tuple

import torch.nn as nn

from ...registry import BACKBONES
from ..layers import ConvNormAct, MBConvBlock
from .feature import FeatureBackboneBase


@BACKBONES.register()
@BACKBONES.register(name="mobilenetv3_backbone")
class MobileNetV3Backbone(FeatureBackboneBase):
    """MobileNetV3-lite 骨干：stem（3x3 stride2）+ 5 个 MBConv 阶段。

    features_only=True 时输出 stride 2/4/8/16/32 五个多尺度特征。
    """

    def __init__(
        self,
        in_channels: int = 3,
        features_only: bool = False,
        out_indices: Optional[Sequence[int]] = None,
        width_mult: float = 1.0,
    ):
        def _ch(c: int) -> int:
            return max(8, int(c * width_mult))

        stages: List[nn.Module] = []
        stem = ConvNormAct(in_channels, _ch(16), kernel_size=3, stride=2, act="relu6")
        stages.append(nn.Sequential(stem))
        prev = _ch(16)

        # (expand, out, kernel, stride, se)
        cfg: List[Tuple[int, int, int, int, bool]] = [
            (1, 16, 3, 1, True),  # stride 2
            (4, 24, 3, 2, False),  # stride 4
            (3, 24, 3, 1, False),
            (3, 40, 5, 2, True),  # stride 8
            (3, 40, 5, 1, True),
            (3, 40, 5, 1, True),
            (6, 80, 3, 2, False),  # stride 16
            (2.5, 80, 3, 1, False),
            (2.3, 80, 3, 1, False),
            (2.3, 80, 3, 1, False),
            (6, 112, 3, 1, True),
            (6, 112, 3, 1, True),
            (6, 160, 5, 2, True),  # stride 32
            (6, 160, 5, 1, True),
            (6, 160, 5, 1, True),
        ]
        for expand, out, k, s, use_se in cfg:
            block = MBConvBlock(
                prev,
                _ch(out),
                kernel_size=k,
                stride=s,
                expand_ratio=int(expand),
                se_ratio=0.25 if use_se else None,
                act="relu6",
            )
            stages.append(nn.Sequential(block))
            prev = _ch(out)
        super().__init__(
            stages, in_channels=in_channels, features_only=features_only, out_indices=out_indices
        )


__all__ = ["MobileNetV3Backbone"]
