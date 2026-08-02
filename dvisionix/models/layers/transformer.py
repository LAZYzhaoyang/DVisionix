# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: Transformer 共享层：可变形编解码层（DETR 家族）、MixFFN（SegFormer 家族）。
"""Transformer 共享层：可变形编解码层（DETR 家族）、MixFFN（SegFormer 家族）。"""

import torch.nn as nn

from ...registry import LAYERS
from .deformable_attention import MultiScaleDeformableAttention


@LAYERS.register()
@LAYERS.register(name="deformable_encoder_layer")
class DeformableEncoderLayer(nn.Module):
    """可变形编码器层：多尺度可变形自注意力 + FFN（DeformableDETR / RT-DETR 共用）。"""

    def __init__(self, d_model, num_heads, num_levels, num_points, dropout):
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(
            d_model,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(inplace=True), nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, proj, ref):
        """可变形编码器层前向：token 序列 + 多尺度特征 -> 同形状输出。"""
        x = x + self.dropout(self.self_attn(self.norm1(x), proj, ref))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


@LAYERS.register()
@LAYERS.register(name="deformable_decoder_layer")
class DeformableDecoderLayer(nn.Module):
    """可变形解码器层：自注意力 + 可变形交叉注意力 + FFN（DeformableDETR / RT-DETR 共用）。"""

    def __init__(self, d_model, num_heads, num_levels, num_points, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = MultiScaleDeformableAttention(
            d_model,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(inplace=True), nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, query, proj, ref_q):
        """可变形解码器层前向：tgt query 序列 -> 同形状输出。"""
        q = self.norm1(tgt + query)
        tgt = tgt + self.dropout(self.self_attn(q, q, q)[0])
        tgt = tgt + self.dropout(self.cross_attn(self.norm2(tgt + query), proj, ref_q))
        tgt = tgt + self.dropout(self.ffn(self.norm3(tgt)))
        return tgt


@LAYERS.register()
@LAYERS.register(name="mix_ffn")
class MixFFN(nn.Module):
    """MixFFN（SegFormer）：LN -> 3x3 深度卷积 -> MLP 扩展/收缩 -> 残差。"""

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.mlp1 = nn.Linear(dim, dim * expand)
        self.act = nn.GELU()
        self.mlp2 = nn.Linear(dim * expand, dim)

    def forward(self, x):
        """MixFFN 前向：x (B,C,H,W) -> 同形状输出。"""
        residual = x
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


__all__ = ["DeformableEncoderLayer", "DeformableDecoderLayer", "MixFFN"]
