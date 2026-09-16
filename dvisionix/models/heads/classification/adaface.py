# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: AdaFace 度量学习头（自适应 margin，随特征范数调节）。
"""AdaFace 度量学习头（自适应 margin，随特征范数调节）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="adaface")
@HEADS.register(name="adaface_head")
class AdaFaceHead(BaseModel):
    """AdaFace：margin 随特征范数自适应（低质量样本用小 margin）。"""

    def __init__(self, in_channels, num_classes, s: float = 30.0, m: float = 0.4, t: float = 1.0):
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        self.s = float(s)
        self.m = float(m)
        self.t = float(t)
        self.weight = nn.Parameter(torch.empty(num_classes, in_channels))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, labels=None):
        """AdaFace 前向：特征 -> 加自适应间隔的余弦 logits (B, num_classes)。"""
        w_norm = F.normalize(self.weight, dim=1)
        x_norm = F.normalize(x, dim=1)
        # fp16 下 1.0-1e-7 舍入成 1.0，clamp 保护失效 → acos 反传 -inf → 梯度 NaN。
        cos = torch.mm(x_norm, w_norm.t())
        if cos.dtype in (torch.float16, torch.bfloat16):
            cos = cos.float()  # 半精度下 1e-4 级余量才可表示
        cos = cos.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        if labels is not None and self.training:
            # 特征范数 -> 自适应 margin 系数（相对 batch 均值）
            norms = torch.norm(x, dim=1).clamp(min=1e-6)
            theta = torch.acos(cos)
            m_adapt = self.m * torch.sigmoid((norms - norms.mean()) / (norms.std() + 1e-6) * self.t)
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            target_cos = torch.cos(theta + m_adapt[:, None] * one_hot)
            cos = cos * (1 - one_hot) + target_cos * one_hot
        return cos * self.s


__all__ = ["AdaFaceHead"]
