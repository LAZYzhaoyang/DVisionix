# -*- coding: utf-8 -*-
"""RetinaNet anchor-based 检测头（RetinaNetHead）。"""

import torch
import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="retinanet_head")
class RetinaNetHead(BaseModel):
    """RetinaNet anchor-based 检测头（多尺度，每位置 num_anchors 个 anchor）。

    输出 dict：
    - cls: 每层 (B, num_anchors * num_classes, H, W)
    - reg: 每层 (B, num_anchors * 4, H, W)
    """

    def __init__(self, in_channels, num_classes, num_anchors=9, num_convs=2):
        super().__init__(task_type="detection")
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        def _convs(out):
            layers = []
            cin = in_channels
            for _ in range(num_convs):
                layers += [nn.Conv2d(cin, in_channels, 3, padding=1), nn.ReLU(inplace=True)]
                cin = in_channels
            return nn.Sequential(*layers)

        self.cls_convs = _convs(in_channels)
        self.reg_convs = _convs(in_channels)
        self.cls_out = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.reg_out = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(self, feats):
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        cls_outs, reg_outs = [], []
        for feat in feats:
            cls_outs.append(self.cls_out(self.cls_convs(feat)))
            reg_outs.append(self.reg_out(self.reg_convs(feat)))
        return {"cls": cls_outs, "reg": reg_outs}


__all__ = ["RetinaNetHead"]
