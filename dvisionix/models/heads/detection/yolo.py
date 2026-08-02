# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: YOLOv8 风格解耦检测头（YOLOHead，anchor-free）。
"""YOLOv8 风格解耦检测头（YOLOHead，anchor-free）。"""

import torch
import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="yolo_head")
class YOLOHead(BaseModel):
    """YOLOv8 风格解耦头：cls 分支 + reg 分支（ltrb 距离，除以 stride 归一化）。

    输入：FPN/PANet 特征列表；输出 dict：
    - cls: 每层 (B, num_classes, H, W)
    - reg: 每层 (B, 4, H, W)  # ltrb / stride
    """

    def __init__(self, in_channels, num_classes, strides=(8, 16, 32), num_convs=2):
        super().__init__(task_type="detection")
        self.num_classes = num_classes
        self.strides = list(strides)

        def _convs():
            layers = []
            cin = in_channels
            for _ in range(num_convs):
                layers += [
                    nn.Conv2d(cin, in_channels, 3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.SiLU(inplace=True),
                ]
                cin = in_channels
            return nn.Sequential(*layers)

        self.cls_convs = _convs()
        self.reg_convs = _convs()
        self.cls_out = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_out = nn.Conv2d(in_channels, 4, 1)

    def forward(self, feats):
        """YOLOHead 前向：特征列表 -> {cls, reg} 每层预测。"""
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        cls_outs, reg_outs = [], []
        for feat in feats:
            cls_outs.append(self.cls_out(self.cls_convs(feat)))
            reg_outs.append(self.reg_out(self.reg_convs(feat)))
        return {"cls": cls_outs, "reg": reg_outs}


__all__ = ["YOLOHead"]
