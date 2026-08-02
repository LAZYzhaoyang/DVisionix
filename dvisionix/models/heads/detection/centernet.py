# -*- coding: utf-8 -*-
"""CenterNet 检测头（关键点热图 + 尺寸 + 偏移）。"""

import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="centernet_head")
class CenterNetHead(BaseModel):
    """CenterNet 风格单阶段头：中心点热图 + 宽高 + 偏移。

    单尺度输入（装配器注入 in_channels）；输出 dict：
    {"heatmap": (B, C, H, W), "wh": (B, 2, H, W), "offset": (B, 2, H, W)}。
    """

    def __init__(self, in_channels, num_classes, hidden: int = 64, num_convs: int = 3):
        super().__init__(task_type="detection")
        self.num_classes = num_classes
        self.hidden = hidden

        stem = []
        cin = in_channels
        for _ in range(num_convs):
            stem += [
                nn.Conv2d(cin, hidden, 3, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ]
            cin = hidden
        self.stem = nn.Sequential(*stem)
        self.heatmap = nn.Conv2d(hidden, num_classes, 1)
        self.wh = nn.Conv2d(hidden, 2, 1)
        self.offset = nn.Conv2d(hidden, 2, 1)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[-1]  # 单尺度头：取多尺度特征最后一层
        feat = self.stem(x)
        return {"heatmap": self.heatmap(feat), "wh": self.wh(feat), "offset": self.offset(feat)}


__all__ = ["CenterNetHead"]
