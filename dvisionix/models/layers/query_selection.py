# -*- coding: utf-8 -*-
"""DINO 混合 query 选择（Hybrid Query Selection）。"""

import torch
import torch.nn as nn

from ...registry import LAYERS


@LAYERS.register()
@LAYERS.register(name="query_selection")
class QuerySelection(nn.Module):
    """混合 query 选择：按类别分数取 top-k encoder token 作为 decoder query，并输出 anchor 框。

    输入 class_logits (B, T, C+1)、box_deltas (B, T, 4)、memory (B, T, d)；
    返回 (queries, init_boxes, indices)。
    """

    def __init__(self, topk: int = 100):
        super().__init__()
        self.topk = int(topk)

    def forward(self, class_logits, box_deltas, memory):
        B, T, _ = memory.shape
        scores = torch.softmax(class_logits, dim=-1)[..., :-1].max(dim=-1).values  # (B, T)
        k = min(self.topk, T)
        idx = scores.topk(k, dim=1).indices  # (B, k)
        queries = memory.gather(1, idx.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
        init_boxes = box_deltas.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4)).sigmoid()
        return queries, init_boxes, idx


__all__ = ["QuerySelection"]
