# -*- coding: utf-8 -*-
"""检测头（demo 级）。

输出每像素 (objectness + box(4) + num_classes) 的原始预测张量，
后处理（decode/NMS）由 dvisionix.models.postprocess 或独立解码器完成。
"""

import torch
import torch.nn as nn

from ..base import BaseModel
from ...registry import HEADS


@HEADS.register()
@HEADS.register(name="det_head")
class DetHead(BaseModel):
    """单阶段检测头（grid 风格）。

    输入: 特征图 (B, in_channels, H, W)；
    输出: raw 张量 (B, 5 + num_classes, H, W)。
    """

    def __init__(self, in_channels, num_classes):
        super().__init__(task_type="detection")
        self.num_classes = num_classes
        self.out_channels = 5 + num_classes
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
