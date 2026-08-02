# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: RT-DETR 增强版检测器（IoU-aware query selection，输出契约与 DETR 一致）。
"""RT-DETR 增强版检测器（IoU-aware query selection，输出契约与 DETR 一致）。"""

from typing import Any, Dict, Optional

from ...registry import MODELS
from .base import SingleStageDetector, detr_decode


@MODELS.register()
@MODELS.register(name="rtdetr_full")
class RTDETRFullDetector(SingleStageDetector):
    """RT-DETR 增强版：多尺度骨干 -> RTDETRFullHead（可变形编码器 + IoU-aware 选择）-> 复用 detr_decode。"""

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
        """推理解码：preds + image_hw -> (boxes_list, scores_list, labels_list)。"""
        return detr_decode(
            preds,
            image_hw,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            topk=topk,
        )


__all__ = ["RTDETRFullDetector"]
