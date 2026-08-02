# -*- coding: utf-8 -*-
"""教学级分割模型（SimpleSegmentationModel）。

仅用于演示与快速验证，生产请使用 SegmentationModel + DeepLabV3Head / UNetDecoder 等组件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class SimpleSegmentationModel(BaseModel):
    """简单的全卷积分割模型（输出与输入同尺寸）。"""

    def __init__(self, num_classes: int = 21, in_channels: int = 3, **kwargs):
        super().__init__()
        self.task_type = "segmentation"
        self.num_classes = num_classes

        self.encoder1 = self._make_block(in_channels, 64)
        self.encoder2 = self._make_block(64, 128)
        self.encoder3 = self._make_block(128, 256)

        self.decoder3 = self._make_block(256, 128)
        self.decoder2 = self._make_block(128, 64)
        self.decoder1 = self._make_block(64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        input_size = x.shape[2:]

        x1 = F.max_pool2d(self.encoder1(x), 2)
        x2 = F.max_pool2d(self.encoder2(x1), 2)
        x3 = F.max_pool2d(self.encoder3(x2), 2)

        x = F.interpolate(self.decoder3(x3), scale_factor=2, mode="bilinear", align_corners=True)
        x = F.interpolate(self.decoder2(x), scale_factor=2, mode="bilinear", align_corners=True)
        x = F.interpolate(self.decoder1(x), scale_factor=2, mode="bilinear", align_corners=True)

        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        return self.final_conv(x)


__all__ = ["SimpleSegmentationModel"]