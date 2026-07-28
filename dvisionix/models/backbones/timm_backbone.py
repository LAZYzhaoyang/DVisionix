# -*- coding: utf-8 -*-
"""基于 timm 的骨干网络封装。

- TimmBackbone: 特征提取器。
    - features_only=False（默认）：输出全局池化后的特征向量 (B, num_features)，适用于分类头。
    - features_only=True：输出多尺度特征图列表 [C1, C2, ...]，并暴露 out_channels，可供 FPN / 检测头 / 分割头复用。
- TimmClassifier: 骨干 + 线性分类头，遵循统一的 BaseModel 接口。

支持 timm 中的 ResNet、ViT、Swin 等大量预训练模型，也支持自定义骨干名称。
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..base import BaseModel
from ...registry import BACKBONES


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


@BACKBONES.register()
@BACKBONES.register(name="timm_backbone")
class TimmBackbone(BaseModel):
    """timm 骨干网络封装（特征提取器）。

    Args:
        name: timm 模型名称。
        pretrained: 是否加载预训练权重。
        in_channels: 输入通道数。
        features_only: 为 True 时输出多尺度特征图列表，否则输出全局池化特征向量。
        out_indices: features_only 模式下返回的特征层级索引（None 表示默认）。
        global_pool: 非 features_only 模式的池化方式。
        drop_rate: 非 features_only 模式的 dropout。

    Examples:
        >>> backbone = TimmBackbone("resnet50", features_only=True, out_indices=(1, 2, 3, 4))
        >>> feats = backbone(torch.randn(2, 3, 224, 224))  # List[Tensor]
        >>> backbone.out_channels  # [C1, C2, C3, C4]
    """

    def __init__(self, name="resnet50", pretrained=False, in_channels=3,
                 features_only=False, out_indices=None, global_pool="avg", drop_rate=0.0):
        super().__init__()
        timm = _require_timm()
        self.name = name
        self.features_only = features_only
        if features_only:
            self.model = timm.create_model(
                name, pretrained=pretrained, in_chans=in_channels,
                features_only=True, out_indices=out_indices,
            )
            try:
                self.out_channels = list(self.model.feature_info.channels())
            except Exception:
                self.out_channels = []
            self.num_features = self.out_channels[-1] if self.out_channels else 0
        else:
            self.model = timm.create_model(
                name, pretrained=pretrained, num_classes=0, in_chans=in_channels,
                global_pool=global_pool, drop_rate=drop_rate,
            )
            self.num_features = int(self.model.num_features)
            self.out_channels = [self.num_features]

    def forward(self, x, **kwargs):
        """返回全局池化特征向量 (B, num_features) 或多尺度特征图列表。"""
        return self.model(x)

    def freeze_backbone(self):
        self.freeze()

    def unfreeze_backbone(self):
        self.unfreeze()


@BACKBONES.register()
@BACKBONES.register(name="timm_classifier")
class TimmClassifier(BaseModel):
    """基于 timm 骨干的分类模型（骨干 + 线性分类头），骨干与分类头解耦。

    Examples:
        >>> model = TimmClassifier("resnet50", num_classes=10, pretrained=False)
        >>> logits = model(torch.randn(2, 3, 224, 224))
        >>> logits.shape  # (2, 10)
    """

    def __init__(self, name="resnet50", num_classes=1000, pretrained=False, in_channels=3, drop_rate=0.0):
        super().__init__(task_type="classification")
        self.name = name
        self.num_classes = num_classes
        self.backbone = TimmBackbone(
            name=name, pretrained=pretrained, in_channels=in_channels,
            global_pool="avg", drop_rate=drop_rate,
        )
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x, **kwargs):
        """前向传播，返回分类 logits (batch_size, num_classes)。"""
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def freeze_backbone(self):
        """冻结骨干参数，仅训练分类头（用于迁移学习/线性探测）。"""
        self.backbone.freeze()

    def unfreeze_backbone(self):
        """解冻骨干参数。"""
        self.backbone.unfreeze()
