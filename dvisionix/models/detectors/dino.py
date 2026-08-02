# -*- coding: utf-8 -*-
"""DINO 检测器（hybrid query selection + 去噪训练，输出契约与 DETR 一致）。"""

from typing import Any, Dict, Optional

from ...registry import MODELS
from .base import SingleStageDetector, detr_decode


@MODELS.register()
@MODELS.register(name="dinodetr")
class DINODetrDetector(SingleStageDetector):
    """DINO-lite：多尺度骨干 -> DINODetrHead（去噪训练需要 batch，needs_batch=True）。"""

    needs_batch = True  # 训练 forward 需 batch（GT 生成去噪 query）

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

    def forward(self, x, **kwargs):
        feats = self.extract_features(x)
        return self.head(feats, batch=kwargs.get("batch"))

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


__all__ = ["DINODetrDetector"]
