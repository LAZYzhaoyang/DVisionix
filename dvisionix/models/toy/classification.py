# -*- coding: utf-8 -*-
"""教学级分类模型（SimpleCNN）。

仅用于演示与快速验证数据/训练流程，生产请使用组件化模型（如 LinearClassifier + timm 骨干）。
"""

import torch
import torch.nn as nn

from ..base import BaseModel


class SimpleCNN(BaseModel):
    """简单的 CNN 分类模型（适用于 CIFAR-10 等小尺寸图像）。"""

    def __init__(self, num_classes: int = 10, in_channels: int = 3, **kwargs):
        super().__init__()
        self.task_type = "classification"
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


__all__ = ["SimpleCNN"]
