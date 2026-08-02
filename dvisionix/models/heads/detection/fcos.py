# -*- coding: utf-8 -*-
"""FCOS anchor-free 检测头（FCOSHead）。"""

import torch
import torch.nn as nn

from ...base import BaseModel
from ....registry import HEADS


@HEADS.register()
@HEADS.register(name="fcos_head")
class FCOSHead(BaseModel):
    """FCOS anchor-free 检测头（多尺度）。

    输入：FPN 特征列表；输出 dict：
    - cls: 每层 (B, num_classes, H, W)
    - reg: 每层 (B, 4, H, W)，log(距离/stride)
    - center: 每层 (B, 1, H, W)，center-ness
    """

    def __init__(self, in_channels, num_classes, strides=(8, 16, 32, 64, 128), num_convs=2):
        super().__init__(task_type="detection")
        self.num_classes = num_classes
        self.strides = list(strides)

        def _convs(out):
            layers = []
            cin = in_channels
            for _ in range(num_convs):
                layers += [nn.Conv2d(cin, in_channels, 3, padding=1), nn.GroupNorm(8, in_channels), nn.ReLU(inplace=True)]
                cin = in_channels
            return nn.Sequential(*layers)

        self.cls_convs = _convs(in_channels)
        self.reg_convs = _convs(in_channels)
        self.cls_out = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_out = nn.Conv2d(in_channels, 4, 1)
        self.center_out = nn.Conv2d(in_channels, 1, 1)

    def forward(self, feats):
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        cls_outs, reg_outs, center_outs = [], [], []
        for feat in feats:
            cls_outs.append(self.cls_out(self.cls_convs(feat)))
            reg_feat = self.reg_convs(feat)
            reg_outs.append(self.reg_out(reg_feat))
            center_outs.append(self.center_out(reg_feat))
        return {"cls": cls_outs, "reg": reg_outs, "center": center_outs}


__all__ = ["FCOSHead"]