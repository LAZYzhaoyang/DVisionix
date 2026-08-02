# -*- coding: utf-8 -*-
"""Deformable DETR 风格检测头（紧凑版：纯 PyTorch 可变形注意力）。"""

import torch
import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel
from ...layers import (
    DeformableDecoderLayer,
    DeformableEncoderLayer,
    PositionEmbeddingSine,
)


@HEADS.register()
@HEADS.register(name="deformable_detr_head")
class DeformableDETRHead(BaseModel):
    """Deformable DETR-lite：多尺度可变形编码器 + query 解码器（自注意力 + 可变形交叉注意力）。

    输出契约与 DETRHead 一致（{"logits": (B,Q,C+1), "boxes": (B,Q,4) 归一化 cxcywh}），
    可复用 DETRLoss 与 detr_decode。
    """

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

    def __init__(
        self,
        in_channels_list,
        num_classes,
        d_model: int = 256,
        num_queries: int = 300,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1,
        in_channels=None,
        **kwargs,
    ):
        super().__init__(task_type="detection")
        self.in_channels_list = list(in_channels_list)
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_levels = len(in_channels_list)

        self.input_projs = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in in_channels_list])
        self.pos_embed = PositionEmbeddingSine(d_model // 2)
        self.level_embed = nn.Parameter(torch.zeros(self.num_levels, d_model))

        self.encoder_layers = nn.ModuleList(
            [
                DeformableEncoderLayer(d_model, num_heads, self.num_levels, num_points, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DeformableDecoderLayer(d_model, num_heads, self.num_levels, num_points, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.ref_point_embed = nn.Embedding(num_queries, 2)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 4)
        )

    def _make_ref_points(self, shapes, device):
        refs = []
        for h, w in shapes:
            ys = (torch.arange(h, device=device) + 0.5) / h
            xs = (torch.arange(w, device=device) + 0.5) / w
            grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2)
            refs.append(grid)
        return torch.cat(refs, dim=0)  # (T, 2)

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        B = feats[0].shape[0]
        device = feats[0].device
        proj = [self.input_projs[i](f) for i, f in enumerate(feats)]
        shapes = [(p.shape[2], p.shape[3]) for p in proj]
        proj = [p + self.level_embed[i].view(1, -1, 1, 1) for i, p in enumerate(proj)]
        pos = [self.pos_embed(p) for p in proj]

        tokens = []
        for i, p in enumerate(proj):
            t = p.flatten(2).permute(0, 2, 1) + pos[i].flatten(2).permute(0, 2, 1)
            tokens.append(t)
        memory = torch.cat(tokens, dim=1)  # (B, T, d)
        ref = self._make_ref_points(shapes, device).unsqueeze(0).expand(B, -1, -1)

        for layer in self.encoder_layers:
            memory = layer(memory, proj, ref)

        tgt = torch.zeros(B, self.num_queries, self.d_model, device=device)
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        ref_q = self.ref_point_embed.weight.sigmoid().unsqueeze(0).expand(B, -1, -1)
        for layer in self.decoder_layers:
            tgt = layer(tgt, query, proj, ref_q)

        logits = self.class_embed(tgt)
        boxes = self.bbox_embed(tgt).sigmoid()
        return {"logits": logits, "boxes": boxes}


__all__ = ["DeformableDETRHead"]
