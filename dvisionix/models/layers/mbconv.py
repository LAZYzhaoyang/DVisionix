# -*- coding: utf-8 -*-
"""MobileNetV3 倒残差块（MBConv：1x1 扩展 + 深度可分离 + SE + 线性投影）。"""

import torch
import torch.nn as nn

from ...registry import LAYERS
from .basic import ConvNormAct, SEBlock


@LAYERS.register()
@LAYERS.register(name="mbconv_block")
class MBConvBlock(nn.Module):
    """MBConv（MobileNetV2/V3 倒残差）块。

    Args:
        in_channels / out_channels: 输入 / 输出通道。
        kernel_size: 深度卷积核（3 或 5）。
        stride: 深度卷积步长（1 或 2）。
        expand_ratio: 扩展倍率（1 表示无 1x1 扩展）。
        se_ratio: SE 通道压缩比（None 禁用）。
        act: 激活类型（如 "relu6" / "silu"）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 4,
        se_ratio: float = 0.25,
        act: str = "relu6",
    ):
        super().__init__()
        hidden = int(in_channels * expand_ratio)
        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_channels, hidden, 1, act=act))
        layers.append(
            ConvNormAct(
                hidden,
                hidden,
                kernel_size,
                stride=stride,
                groups=hidden,
                act=act,
                padding=kernel_size // 2,
            )
        )
        if se_ratio is not None and se_ratio > 0:
            layers.append(SEBlock(hidden, reduction=max(1, int(1 / se_ratio))))
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.blocks = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks(x)
        return x + out if self.use_residual else out


__all__ = ["MBConvBlock"]
