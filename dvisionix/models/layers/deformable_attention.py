# -*- coding: utf-8 -*-
"""多尺度可变形注意力（纯 PyTorch compact 实现，无 C++ 算子）。

对每个 query，在每层特征上预测 num_points 个采样偏移，用双线性采样聚合多尺度上下文，
注意力权重对（level, point）做 softmax。参考原版 Deformable DETR，但做了 compact 化
（单参考点、head 间取平均），便于即插即用与教学。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import LAYERS


@LAYERS.register()
@LAYERS.register(name="multi_scale_deformable_attention")
class MultiScaleDeformableAttention(nn.Module):
    """多尺度可变形注意力。

    Args:
        embed_dim: 特征维度。
        num_heads: 注意力头数（输出对各头取平均，保持 compact）。
        num_levels: 特征层级数。
        num_points: 每层采样点数。
        dropout: 输出 dropout。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.level_embed = nn.Parameter(torch.zeros(num_levels, embed_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.level_embed, std=0.02)

    def forward(self, query, value_list, reference_points):
        """query: (B, N, C)；value_list: List[(B, C, H, W)]（各层特征）；reference_points: (B, N, 2) 归一化。"""
        B, N, C = query.shape
        L = len(value_list)
        assert L == self.num_levels

        offset = self.sampling_offsets(query).reshape(B, N, self.num_heads, L, self.num_points, 2)
        offset = offset * 0.1  # 归一化坐标下的小位移，保持稳定
        ref = reference_points.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (B, N, 1, 1, 1, 2)
        sample_locations = ref + offset  # (B, N, H, L, P, 2) 归一化 [0,1]

        weights = self.attention_weights(query).reshape(B, N, self.num_heads, L, self.num_points)
        weights = weights.softmax(dim=3).softmax(dim=4)  # 对 level 与 point 分别 softmax
        values = [self.value_proj(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for v in value_list]

        sampled_list = []
        for lvl in range(L):
            grid = sample_locations[:, :, :, lvl]  # (B, N, H, P, 2)
            grid = grid.permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, N, self.num_points, 2)
            # grid_sample: grid 取值 [-1,1]
            grid_s = grid * 2.0 - 1.0
            vals = F.grid_sample(
                values[lvl].repeat_interleave(self.num_heads, dim=0),
                grid_s,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # (B*H, C, N, P)
            vals = vals.reshape(B, self.num_heads, C, N, self.num_points)
            sampled_list.append(vals)
        sampled = torch.stack(sampled_list, dim=4)  # (B, H, C, N, L, P)
        # 注意力权重 (B, N, H, L, P) -> (B, H, 1, N, L, P)
        w = weights.permute(0, 2, 1, 3, 4).unsqueeze(2)  # (B, H, 1, N, L, P)
        out = (sampled * w).sum(dim=(4, 5))  # (B, H, C, N)
        out = out.mean(dim=1)  # 各头取平均 -> (B, C, N)
        out = out.permute(0, 2, 1)  # (B, N, C)
        return self.dropout(self.output_proj(out))


@LAYERS.register()
@LAYERS.register(name="multi_scale_deformable_attention_v2")
class MultiScaleDeformableAttentionV2(MultiScaleDeformableAttention):
    """多尺度可变形注意力 V2：在 compact 版基础上增强——分层参考点 + 按层尺度归一采样偏移。

    - 分层参考点：每层参考点 = 基础参考点 + 每层可学习偏移（level_offset）。
    - 尺度归一偏移：采样偏移按各层空间尺寸归一化（offset * 2 / (H_l, W_l)），跨层采样更均衡。
    输出契约与 compact 版一致，可替换 DeformableDETR / RT-DETR 的注意力层。
    """

    def __init__(self, embed_dim, num_heads=8, num_levels=4, num_points=4, dropout=0.1):
        super().__init__(embed_dim, num_heads, num_levels, num_points, dropout)
        self.level_offset = nn.Parameter(torch.zeros(num_levels, 1, 1, 2))
        nn.init.uniform_(self.level_offset, -0.02, 0.02)

    def forward(self, query, value_list, reference_points):
        B, N, C = query.shape
        L = len(value_list)
        assert L == self.num_levels

        offset = self.sampling_offsets(query).reshape(B, N, self.num_heads, L, self.num_points, 2)
        ref = reference_points.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (B, N, 1, 1, 1, 2)
        # 分层参考点：基础参考点 + 每层可学习偏移
        ref = ref + self.level_offset.view(1, 1, 1, L, 1, 2)
        # 采样偏移按各层空间尺寸归一化（非原地，保持 autograd）
        scales = torch.zeros(L, 2, device=query.device)
        for lvl in range(L):
            h_l, w_l = value_list[lvl].shape[-2:]
            scales[lvl] = torch.tensor([2.0 / w_l, 2.0 / h_l], device=query.device)
        offset = offset * scales.view(1, 1, 1, L, 1, 2)
        sample_locations = ref + offset

        weights = self.attention_weights(query).reshape(B, N, self.num_heads, L, self.num_points)
        weights = weights.softmax(dim=3).softmax(dim=4)
        values = [self.value_proj(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for v in value_list]

        sampled_list = []
        for lvl in range(L):
            grid = sample_locations[:, :, :, lvl]  # (B, N, H, P, 2)
            grid = grid.permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, N, self.num_points, 2)
            grid_s = grid * 2.0 - 1.0
            vals = F.grid_sample(
                values[lvl].repeat_interleave(self.num_heads, dim=0),
                grid_s,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            vals = vals.reshape(B, self.num_heads, C, N, self.num_points)
            sampled_list.append(vals)
        sampled = torch.stack(sampled_list, dim=4)  # (B, H, C, N, L, P)
        w = weights.permute(0, 2, 1, 3, 4).unsqueeze(2)
        out = (sampled * w).sum(dim=(4, 5))
        out = out.mean(dim=1).permute(0, 2, 1)
        return self.dropout(self.output_proj(out))


__all__ = ["MultiScaleDeformableAttention", "MultiScaleDeformableAttentionV2"]
