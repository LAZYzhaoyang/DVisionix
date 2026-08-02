# -*- coding: utf-8 -*-
"""YOLOv11 PSA 位置自注意力层（分组 + 自注意力 + 融合）。"""

import torch
import torch.nn as nn

from ...registry import LAYERS
from .basic import ConvNormAct


@LAYERS.register()
@LAYERS.register(name="psa_block")
class PSABlock(nn.Module):
    """PSA（Position-Sensitive Attention，YOLOv11）：1x1 分流 -> 分组自注意力 -> concat -> 1x1。

    compact 实现：自注意力用通道级 MHSA（B, N, C）在空间 token 上计算。
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = None,
        groups: int = 2,
        num_heads: int = 4,
        act: str = "silu",
    ):
        super().__init__()
        hidden = hidden or max(1, in_channels // 2)
        self.conv_in = ConvNormAct(in_channels, hidden, 1, act=act)
        self.groups = groups
        self.attn = nn.MultiheadAttention(
            hidden // groups, num_heads=min(num_heads, hidden // groups), batch_first=True
        )
        self.norm = nn.LayerNorm(hidden // groups)
        self.conv_out = ConvNormAct(hidden, in_channels, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv_in(x)
        B, C, H, W = h.shape
        g = self.groups
        hg = C // g
        hs = list(h.chunk(g, dim=1))  # 每块 (B, hg, H, W)
        outs = []
        for part in hs:
            t = part.flatten(2).permute(0, 2, 1)  # (B, HW, hg)
            a = self.attn(self.norm(t), self.norm(t), self.norm(t))[0]
            a = a.permute(0, 2, 1).reshape(B, hg, H, W)
            outs.append(a)
        fused = torch.cat(outs, dim=1)
        return residual + self.conv_out(fused)


__all__ = ["PSABlock"]
