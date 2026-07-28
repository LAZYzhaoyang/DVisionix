# -*- coding: utf-8 -*-
"""内置自定义层。

这些层都注册到全局 ``LAYERS`` 注册表，可通过 ``build_layer`` 或配置字典构建，
也可以直接 import 使用。用户可参考此文件的写法定义自己的层并注册。
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from ...registry import LAYERS
from .builder import build_norm_layer, build_activation_layer


@LAYERS.register()
@LAYERS.register(name="conv_norm_act")
class ConvNormAct(nn.Module):
    """Conv2d + Norm + Activation 的常用组合块。

    Args:
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        kernel_size: 卷积核大小。
        stride: 步长。
        padding: 填充；``None`` 时自动按 ``kernel_size // 2`` 计算（same padding）。
        dilation: 空洞率。
        groups: 分组卷积组数。
        bias: 是否使用偏置；``None`` 时在有 norm 时自动关闭。
        norm: 归一化类型（见 build_norm_layer），``None`` 表示不使用。
        act: 激活类型（见 build_activation_layer），``None`` 表示不使用。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        norm: Union[str, Dict[str, Any], None] = "bn",
        act: Union[str, Dict[str, Any], None] = "relu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        if bias is None:
            bias = norm is None

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.norm = build_norm_layer(norm, out_channels)
        self.act = build_activation_layer(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


@LAYERS.register()
@LAYERS.register(name="mlp")
class MLP(nn.Module):
    """两层 MLP（Linear -> Act -> Dropout -> Linear -> Dropout）。

    常用于分类头、Transformer FFN 等场景。
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act: Union[str, Dict[str, Any], None] = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


@LAYERS.register()
@LAYERS.register(name="se")
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    Args:
        channels: 输入通道数。
        reduction: 中间维度的压缩比例。
        act: 压缩后的激活。
        gate: 门控激活（输出到 [0,1]）。
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        act: Union[str, Dict[str, Any], None] = "relu",
        gate: Union[str, Dict[str, Any], None] = "sigmoid",
    ) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = build_activation_layer(act)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gate = build_activation_layer(gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.gate(self.fc2(self.act(self.fc1(scale))))
        return x * scale


@LAYERS.register()
@LAYERS.register(name="drop_path")
class DropPath(nn.Module):
    """随机深度（Stochastic Depth）——按样本随机丢弃残差分支。

    训练时以概率 ``drop_prob`` 将整条残差路径置零并做尺度补偿；推理时为恒等。
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # (B, 1, 1, ...) 形状的伯努利掩码，适配任意维度
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


__all__ = ["ConvNormAct", "MLP", "SEBlock", "DropPath"]