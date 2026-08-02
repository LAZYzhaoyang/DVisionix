# -*- coding: utf-8 -*-
"""GRN（全局响应归一化，ConvNeXtV2）层。"""

import torch
import torch.nn as nn

from ...registry import LAYERS


@LAYERS.register()
@LAYERS.register(name="grn")
class GRN(nn.Module):
    """全局响应归一化（ConvNeXtV2）：对每个 token 的特征响应做归一化后经可学习门控缩放。

    Input: (B, H, W, C) channels_last 或 (B, C, H, W) channels_first。
    """

    def __init__(self, dim: int, channels_first: bool = False, eps: float = 1e-6):
        super().__init__()
        self.channels_first = channels_first
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_first:
            gx = x.norm(p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            w = self.gamma.view(1, -1, 1, 1) * nx + self.beta.view(1, -1, 1, 1)
            return x * w
        gx = x.norm(p=2, dim=-1, keepdim=True)
        nx = gx / (gx.mean(dim=-2, keepdim=True) + self.eps)
        w = self.gamma * nx + self.beta
        return x * w


__all__ = ["GRN"]
