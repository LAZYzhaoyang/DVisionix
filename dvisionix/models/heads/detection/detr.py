# -*- coding: utf-8 -*-
"""DETR 检测头（transformer encoder-decoder + 类别/框 FFN）。"""

import torch
import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel
from ...layers import PositionEmbeddingSine


@HEADS.register()
@HEADS.register(name="detr_head")
class DETRHead(BaseModel):
    """DETR head：单层特征 -> transformer -> (logits, boxes)。

    输出：
    - logits: (B, num_queries, num_classes + 1)（含背景）
    - boxes: (B, num_queries, 4) 归一化 cxcywh
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        d_model: int = 256,
        num_queries: int = 100,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(task_type="detection")
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.pos_embed = PositionEmbeddingSine(d_model // 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(self, feats):
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        x = self.input_proj(feats)  # (B, d, H, W)
        b, d, h, w = x.shape
        src = x.flatten(2).permute(0, 2, 1)  # (B, HW, d)
        pos = self.pos_embed(x).flatten(2).permute(0, 2, 1)  # (B, HW, d)
        memory = self.encoder(src + pos)  # (B, HW, d)

        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        tgt = torch.zeros_like(queries)
        hs = self.decoder(tgt + queries, memory)  # (B, Q, d)

        logits = self.class_embed(hs)
        boxes = self.bbox_embed(hs).sigmoid()
        return {"logits": logits, "boxes": boxes}


__all__ = ["DETRHead"]
