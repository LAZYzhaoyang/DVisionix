# -*- coding: utf-8 -*-
"""RT-DETR 增强版检测头（IoU-aware query selection + 多尺度可变形编码器，compact）。"""

import torch
import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel
from ...layers import MultiScaleDeformableAttention, PositionEmbeddingSine


@HEADS.register()
@HEADS.register(name="rtdetr_full_head")
class RTDETRFullHead(BaseModel):
    """RT-DETR 增强版（compact）：多尺度可变形编码器 -> IoU-aware query selection -> transformer 解码器。

    与 compact 版（rtdetr_head）的区别：
    - 编码器用多尺度可变形注意力（而非简单卷积融合）；
    - query 选择头同时预测类别与框，score = class_score * iou，选择分数最高的 top-k；
    - 解码器以选择头的预测框中心为参考点做可变形交叉注意力，框经细化后输出。

    输出契约与 DETR 一致（{"logits": (B,Q,C+1), "boxes": (B,Q,4) 归一化 cxcywh}），
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
        topk: int = 100,
        dropout: float = 0.1,
        in_channels=None,
        **kwargs,
    ):
        super().__init__(task_type="detection")
        self.in_channels_list = list(in_channels_list)
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_queries = num_queries
        self.topk = min(int(num_queries), int(topk))
        self.num_levels = len(in_channels_list)

        self.input_projs = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in in_channels_list])
        self.pos_embed = PositionEmbeddingSine(d_model // 2)
        self.level_embed = nn.Parameter(torch.zeros(self.num_levels, d_model))

        self.encoder_layers = nn.ModuleList(
            [
                _DeformableEncoderLayer(d_model, num_heads, self.num_levels, num_points, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        # IoU-aware selection head（作用于拼接后的扁平 token）
        self.sel_class_embed = nn.Linear(d_model, num_classes + 1)
        self.sel_box_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 4)
        )
        self.sel_iou_embed = nn.Linear(d_model, 1)

        self.decoder_layers = nn.ModuleList(
            [
                _DeformableDecoderLayer(d_model, num_heads, self.num_levels, num_points, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.query_embed = nn.Embedding(num_queries, d_model)
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
        return torch.cat(refs, dim=0)

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

        # IoU-aware query selection
        sel_logits = self.sel_class_embed(memory)  # (B, T, C+1)
        sel_boxes = self.sel_box_embed(memory).sigmoid()  # (B, T, 4) cxcywh
        sel_iou = self.sel_iou_embed(memory).sigmoid()  # (B, T, 1)
        cls_score = torch.softmax(sel_logits, dim=-1)[..., :-1].max(dim=-1).values  # (B, T)
        score = cls_score * sel_iou.squeeze(-1)  # (B, T)
        k = min(self.topk, memory.shape[1])
        topk_idx = score.topk(k, dim=1).indices  # (B, k)

        queries = memory.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        ref_q = sel_boxes.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))[..., :2]  # cxcy
        init_boxes = sel_boxes.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))

        query_pos = self.query_embed.weight[:k].unsqueeze(0).expand(B, -1, -1)
        tgt = queries  # 以选择出的 token 内容作为解码器初始状态
        for layer in self.decoder_layers:
            tgt = layer(tgt, query_pos, proj, ref_q)

        logits = self.class_embed(tgt)
        delta = self.bbox_embed(tgt)
        boxes = (init_boxes + delta).sigmoid()
        return {"logits": logits, "boxes": boxes}


class _DeformableEncoderLayer(nn.Module):
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
        x = x + self.dropout(self.self_attn(self.norm1(x), proj, ref))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class _DeformableDecoderLayer(nn.Module):
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
        q = self.norm1(tgt + query)
        tgt = tgt + self.dropout(self.self_attn(q, q, q)[0])
        tgt = tgt + self.dropout(self.cross_attn(self.norm2(tgt + query), proj, ref_q))
        tgt = tgt + self.dropout(self.ffn(self.norm3(tgt)))
        return tgt


__all__ = ["RTDETRFullHead"]
