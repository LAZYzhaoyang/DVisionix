# -*- coding: utf-8 -*-
"""CurricularFace 度量学习头（课程式自适应 margin，compact）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="curricularface")
@HEADS.register(name="curricularface_head")
class CurricularFaceHead(BaseModel):
    """CurricularFace（compact）：目标类施加随难度自适应的角度 margin。

    与论文"课程式"思路一致但做了简化：目标相似度越低（样本越难）margin 越大，
    促使模型优先学好困难样本。训练时传 ``labels`` 生效；推理时退化为 s*cos(theta)。
    """

    def __init__(self, in_channels, num_classes, s: float = 30.0, m: float = 0.5):
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_channels))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, labels=None):
        w_norm = F.normalize(self.weight, dim=1)
        x_norm = F.normalize(x, dim=1)
        cos = torch.mm(x_norm, w_norm.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is not None and self.training:
            n = cos.shape[0]
            target = cos[torch.arange(n, device=cos.device), labels]  # (N,)
            # 自适应 margin：困难样本（目标相似度低）margin 更大
            margin = self.m * (1.0 - target) / (1.0 + target).clamp(min=1e-4)
            theta = torch.acos(target)
            cos_target = torch.cos(theta + margin)
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            cos = cos * (1 - one_hot) + cos_target.unsqueeze(1) * one_hot

        return cos * self.s


__all__ = ["CurricularFaceHead"]
