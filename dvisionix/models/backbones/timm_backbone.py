# -*- coding: utf-8 -*-
"""
基于 timm 的骨干网络封装

- TimmBackbone: 特征提取器，输出全局池化后的特征向量，暴露 num_features。
- TimmClassifier: 骨干 + 线性分类头，遵循统一的 BaseModel 接口。

支持 timm 中的 ResNet、ViT、Swin 等大量预训练模型，也支持自定义骨干名称。
"""

from typing import List, Optional

import torch
import torch.nn as nn

from ..base import BaseModel


def _require_timm():
    try:
        import timm  # noqa: F401
        return timm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "timm is not installed. Please install it with: pip install timm"
        ) from exc


def list_timm_models(filter: str = "") -> List[str]:
    """列出 timm 中可用的模型名称（可用通配符过滤，如 'resnet*'）。"""
    timm = _require_timm()
    return timm.list_models(filter) if filter else timm.list_models()


class TimmBackbone(BaseModel):
    """
    timm 骨干网络封装（特征提取器）

    通过 timm.create_model(num_classes=0) 得到全局池化后的特征向量，
    可作为分类头/检测头/分割头的通用特征来源。

    Examples:
        >>> backbone = TimmBackbone("resnet50", pretrained=False)
        >>> feats = backbone(torch.randn(2, 3, 224, 224))
        >>> feats.shape  # (2, backbone.num_features)
    """

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = False,
        in_channels: int = 3,
        global_pool: str = "avg",
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        timm = _require_timm()
        self.name = name
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            global_pool=global_pool,
            drop_rate=drop_rate,
        )
        # timm 模型在 num_classes=0 时 forward 返回池化特征
        self.num_features: int = int(self.model.num_features)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """返回全局池化后的特征向量 (batch_size, num_features)。"""
        return self.model(x)


class TimmClassifier(BaseModel):
    """
    基于 timm 骨干的分类模型（骨干 + 线性分类头）

    骨干与分类头解耦，便于替换骨干或复用特征。

    Examples:
        >>> model = TimmClassifier("resnet50", num_classes=10, pretrained=False)
        >>> logits = model(torch.randn(2, 3, 224, 224))
        >>> logits.shape  # (2, 10)
    """

    def __init__(
        self,
        name: str = "resnet50",
        num_classes: int = 1000,
        pretrained: bool = False,
        in_channels: int = 3,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.task_type = "classification"
        self.name = name
        self.num_classes = num_classes

        self.backbone = TimmBackbone(
            name=name,
            pretrained=pretrained,
            in_channels=in_channels,
            global_pool="avg",
            drop_rate=drop_rate,
        )
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播，返回分类 logits (batch_size, num_classes)。"""
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def freeze_backbone(self) -> None:
        """冻结骨干参数，仅训练分类头（用于迁移学习/线性探测）。"""
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        """解冻骨干参数。"""
        self.backbone.unfreeze()
