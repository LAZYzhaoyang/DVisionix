# -*- coding: utf-8 -*-
"""SwinV2 连续相对位置偏置（log-spaced，MLP 映射）。"""

import torch
import torch.nn as nn

from ...registry import LAYERS


@LAYERS.register()
@LAYERS.register(name="continuous_relative_position_bias")
class ContinuousRelativePositionBias(nn.Module):
    """SwinV2 的 log-spaced 连续相对位置偏置：位置差经 log 编码后由 MLP 映射为每头偏置。

    forward(window_size) -> (num_heads, ws*ws, ws*ws) 偏置表。
    """

    def __init__(self, num_heads: int, hidden_dim: int = 64):
        super().__init__()
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )

    def forward(self, window_size: int) -> torch.Tensor:
        coords_h = torch.arange(window_size, device=self.cpb_mlp[0].weight.device)
        coords_w = torch.arange(window_size, device=self.cpb_mlp[0].weight.device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flat = coords.flatten(1).unsqueeze(2) - coords.flatten(1).unsqueeze(1)  # (2, N, N)
        rel = coords_flat.permute(1, 2, 0).float()  # (N, N, 2)
        # log 空间连续编码（SwinV2）：sign * log(1 + |rel|)
        rel_log = torch.sign(rel) * torch.log1p(rel.abs())
        bias = self.cpb_mlp(rel_log)  # (N, N, num_heads)
        return bias.permute(2, 0, 1)  # (num_heads, N, N)


__all__ = ["ContinuousRelativePositionBias"]
