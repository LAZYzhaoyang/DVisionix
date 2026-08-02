# -*- coding: utf-8 -*-
"""归一化层：channels-first 的 LayerNorm2d。"""

import torch
import torch.nn as nn

from ...registry import LAYERS


@LAYERS.register()
@LAYERS.register(name="layer_norm2d")
class LayerNorm2d(nn.Module):
    """逐通道 LayerNorm（channels_first，(B, C, H, W) 上对每个通道做 LN）。

    供 ConvNeXt / Swin / SegFormerV2 等需要 channels-first LN 的模型共享。
    """

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


__all__ = ["LayerNorm2d"]
