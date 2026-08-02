# -*- coding: utf-8 -*-
"""Deformable DETR 检测器（多尺度可变形注意力，输出契约与 DETR 一致）。"""

from typing import Any, Dict, Optional

from ...registry import MODELS
from .base import SingleStageDetector, detr_decode


@MODELS.register()
@MODELS.register(name="deformable_detr")
class DeformableDETRDetector(SingleStageDetector):
    """Deformable DETR-lite：多尺度骨干 -> DeformableDETRHead -> 复用 detr_decode。"""

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        out_indices: Optional[list] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        head_cfg = dict(head)
        if num_classes is not None and "num_classes" not in head_cfg:
            head_cfg["num_classes"] = num_classes
        super().__init__(backbone, head_cfg, neck, out_indices)

    def decode(
        self, preds, image_hw, score_threshold=0.05, iou_threshold=0.5, max_detections=100, topk=300
    ):
        return detr_decode(
            preds,
            image_hw,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            topk=topk,
        )


__all__ = ["DeformableDETRDetector"]
