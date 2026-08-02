# -*- coding: utf-8 -*-
"""YOLOv8 风格检测器（anchor-free，TaskAlignedAssigner）。"""

from typing import Any, Dict, Optional

from ..postprocess import yolo_decode
from .base import SingleStageDetector
from ...registry import MODELS


@MODELS.register()
@MODELS.register(name="yolo")
class YOLODetector(SingleStageDetector):
    """YOLOv8 风格单阶段检测器（复用 backbone + neck + YOLOHead）。"""

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
        self.strides = list(getattr(self.head, "strides", (8, 16, 32)))

    def decode(self, preds, image_hw, score_threshold=0.05, iou_threshold=0.5,
               max_detections=100, topk_per_level=1000):
        return yolo_decode(
            preds, image_hw, self.strides,
            score_threshold=score_threshold, iou_threshold=iou_threshold,
            max_detections=max_detections, topk_per_level=topk_per_level,
        )


__all__ = ["YOLODetector"]