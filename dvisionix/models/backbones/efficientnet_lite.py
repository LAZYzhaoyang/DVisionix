# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: EfficientNetLite 轻量骨干（MBConv + SE，B0-lite 配置）。
"""EfficientNetLite 轻量骨干（MBConv + SE，B0-lite 配置）。"""

from typing import List, Optional, Sequence, Tuple

import torch.nn as nn

from ...registry import BACKBONES
from ..layers import ConvNormAct, MBConvBlock
from .feature import FeatureBackboneBase


@BACKBONES.register()
@BACKBONES.register(name="efficientnet_lite_backbone")
class EfficientNetLiteBackbone(FeatureBackboneBase):
    """EfficientNet-lite（B0 缩放）骨干：stem（3x3 stride2）+ 7 个 MBConv 阶段（stride 2/4/8/16/32/32/32）。

    width_mult / depth_mult 控制缩放。
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
        stem = ConvNormAct(in_channels, _ch(32), kernel_size=3, stride=2, act="silu")
        stages.append(nn.Sequential(stem))
        prev = _ch(32)

        # (expand, out, kernel, stride, se)
        cfg: List[Tuple[int, int, int, int, bool]] = [
            (1, 16, 3, 1, True),
            (6, 24, 3, 2, True),
            (6, 24, 3, 1, True),
            (6, 40, 5, 2, True),
            (6, 40, 5, 1, True),
            (6, 80, 3, 2, True),
            (6, 80, 3, 1, True),
            (6, 80, 3, 1, True),
            (6, 112, 3, 1, True),
            (6, 112, 3, 1, True),
            (6, 112, 5, 1, True),
            (6, 192, 5, 2, True),
            (6, 192, 5, 1, True),
            (6, 192, 5, 1, True),
            (6, 192, 5, 1, True),
            (6, 320, 3, 1, True),
        ]
        for expand, out, k, s, use_se in cfg:
            block = MBConvBlock(
                prev,
                _ch(out),
                kernel_size=k,
                stride=s,
                expand_ratio=int(expand),
                se_ratio=0.25 if use_se else None,
                act="silu",
            )
            stages.append(nn.Sequential(block))
            prev = _ch(out)
        super().__init__(
            stages, in_channels=in_channels, features_only=features_only, out_indices=out_indices
        )


__all__ = ["EfficientNetLiteBackbone"]
