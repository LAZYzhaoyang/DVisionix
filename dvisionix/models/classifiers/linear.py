# -*- coding: utf-8 -*-
"""LinearClassifier（backbone + 分类头，配置驱动、即插即用）。"""

from typing import Any, Dict, Optional

import torch

from ...registry import BACKBONES, HEADS, MODELS
from ..base import BaseModel


@MODELS.register()
@MODELS.register(name="linear_classifier")
class LinearClassifier(BaseModel):
    """骨干 + 分类头组合模型（替代旧的 generalized 分类分支）。"""

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(task_type="classification")
        bb_cfg = dict(backbone)
        self.backbone = BACKBONES.build(bb_cfg)
        self.num_features = int(
            getattr(self.backbone, "num_features", 0)
            or getattr(self.backbone, "out_channels", [0])[-1]
        )
        if head is None:
            if num_classes is None:
                raise ValueError("head 未提供时必须给出 num_classes")
            self.head = HEADS.build(
                {"type": "cls_head", "in_channels": self.num_features, "num_classes": num_classes}
            )
        else:
            head_cfg = dict(head)
            head_cfg.setdefault("in_channels", self.num_features)
            if "num_classes" not in head_cfg and num_classes is not None:
                head_cfg["num_classes"] = num_classes
            self.head = HEADS.build(head_cfg)
        self.num_classes = getattr(self.head, "num_classes", num_classes)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.head(self.backbone(x))


__all__ = ["LinearClassifier"]
