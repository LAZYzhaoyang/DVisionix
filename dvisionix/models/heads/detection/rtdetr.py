# -*- coding: utf-8 -*-
"""RT-DETR 风格检测头（混合编码器 + query 选择 + transformer 解码器，compact）。"""

import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel
from ...layers import PositionEmbeddingSine


@HEADS.register()
@HEADS.register(name="rtdetr_head")
class RTDETRHead(BaseModel):
    """RT-DETR-lite：多尺度特征经混合编码器融合 -> 按类别分数选 top-k query ->
    transformer 解码器 -> (logits, boxes)。输出契约与 DETRHead 一致，可复用 DETRLoss / detr_decode。
    """

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

    def __init__(
        self,
        in_channels_list,
        num_classes,
        d_model: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        topk: int = 100,
        dropout: float = 0.1,
        in_channels=None,
        **kwargs,
    ):
        super().__init__(task_type="detection")
        self.in_channels_list = list(in_channels_list)
        self.num_classes = num_classes
        self.d_model = d_model
        self.topk = min(int(num_queries), int(topk))

        self.input_projs = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in in_channels_list])
        self.score_conv = nn.Conv2d(d_model, num_classes + 1, 1)
        self.pos_embed = PositionEmbeddingSine(d_model // 2)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward=1024, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        target = feats[0].shape[-2:]
        fused = None
        for i, f in enumerate(feats):
            p = self.input_projs[i](f)
            if p.shape[-2:] != target:
                p = F.interpolate(p, size=target, mode="bilinear", align_corners=False)
            fused = p if fused is None else fused + p

        b, d, h, w = fused.shape
        flat = fused.flatten(2).permute(0, 2, 1)  # (B, HW, d)
        pos = self.pos_embed(fused).flatten(2).permute(0, 2, 1)
        memory = flat + pos

        scores = self.score_conv(fused).flatten(2).permute(0, 2, 1)  # (B, HW, C+1)
        max_score = scores.max(dim=-1).values  # (B, HW)
        k = min(self.topk, int(h * w))
        topk_idx = max_score.topk(k, dim=1).indices  # (B, k)
        queries = flat.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, d))  # (B, k, d)

        hs = self.decoder(queries, memory)
        logits = self.class_embed(hs)
        boxes = self.bbox_embed(hs).sigmoid()
        return {"logits": logits, "boxes": boxes}


__all__ = ["RTDETRHead"]
