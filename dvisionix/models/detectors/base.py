# -*- coding: utf-8 -*-
"""检测器装配脚手架（SingleStageDetector）。

统一「backbone(features_only) + neck(可选) + head」装配逻辑，具体检测器（FCOS / RetinaNet）
继承本类并实现 forward / decode / 具体损失接入。backbone / neck / head 均可配置驱动、即插即用。
"""

from typing import Any, Dict, Optional

import torch

from ...registry import BACKBONES, HEADS, NECKS
from ..base import BaseModel


class SingleStageDetector(BaseModel):
    """单阶段检测器基类。

    Args:
        backbone: 骨干配置（自动 features_only=True）。
        head: 检测头配置（自动注入 in_channels）。
        neck: 颈部配置（可选，自动注入 in_channels）。
        out_indices: 骨干输出层级（可选）。
    """

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        out_indices: Optional[list] = None,
    ):
        super().__init__(task_type="detection")
        bb_cfg = dict(backbone)
        bb_cfg.setdefault("features_only", True)
        if out_indices is not None:
            bb_cfg["out_indices"] = out_indices
        self.backbone = BACKBONES.build(bb_cfg)

        if neck is not None:
            neck_cfg = dict(neck)
            neck_cfg.setdefault("in_channels", self.backbone.out_channels)
            self.neck = NECKS.build(neck_cfg)
            neck_out = getattr(self.neck, "out_channels", None)
            self.in_channels = (
                neck_out
                if isinstance(neck_out, int)
                else (
                    neck_out[-1]
                    if isinstance(neck_out, (list, tuple))
                    else self.backbone.out_channels[-1]
                )
            )
        else:
            self.neck = None
            self.in_channels = self.backbone.out_channels[-1]

        head_cfg = dict(head)
        if head.get("type") in ("rtdetr_head", "RTDETRHead"):
            head_cfg.setdefault("in_channels_list", list(self.backbone.out_channels))
        else:
            head_cfg.setdefault("in_channels", self.in_channels)
        self.head = HEADS.build(head_cfg)
        self.num_classes = getattr(self.head, "num_classes", head_cfg.get("num_classes"))

    def extract_features(self, x: torch.Tensor):
        """backbone -> neck 特征提取，返回多尺度特征列表。"""
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward(self, x: torch.Tensor, **kwargs):
        """原始预测（多尺度 dict），后处理由 decode 完成。"""
        return self.head(self.extract_features(x))

    def decode(self, preds, image_hw, score_threshold=0.3, iou_threshold=0.5, max_detections=100):
        """子类实现：预测 -> (boxes_list, scores_list, labels_list)。"""
        raise NotImplementedError


__all__ = ["SingleStageDetector"]
