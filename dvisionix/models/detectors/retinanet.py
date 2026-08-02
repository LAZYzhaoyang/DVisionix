# -*- coding: utf-8 -*-
"""RetinaNet 检测器（anchor-based 单阶段）。

配置示例::

    model:
      type: retinanet
      num_classes: 3
      backbone: {type: timm_backbone, name: resnet18, features_only: true, out_indices: [1,2,3,4]}
      neck: {type: fpn, out_channels: 128}
      head: {type: retinanet_head, num_classes: 3, num_anchors: 9}
    # 对应 loss 可用 retinanet_detection（assigner: max_iou 或 atss）
"""

from typing import Any, Dict, Optional

from ...registry import MODELS
from ..postprocess import retinanet_decode
from .anchors import AnchorGenerator
from .base import SingleStageDetector


@MODELS.register()
@MODELS.register(name="retinanet")
class RetinaNetDetector(SingleStageDetector):
    """RetinaNet anchor-based 单阶段检测器。"""

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        out_indices: Optional[list] = None,
        num_classes: Optional[int] = None,
        strides=(8, 16, 32, 64, 128),
        base_sizes=(32, 64, 128, 256, 512),
        **kwargs,
    ):
        head_cfg = dict(head)
        if num_classes is not None and "num_classes" not in head_cfg:
            head_cfg["num_classes"] = num_classes
        super().__init__(backbone, head_cfg, neck, out_indices)
        self.anchor_gen = AnchorGenerator(strides=strides, base_sizes=base_sizes)
        if getattr(self.head, "num_anchors", None) != self.anchor_gen.num_anchors:
            raise ValueError(
                f"head.num_anchors ({getattr(self.head, 'num_anchors', None)}) 与 "
                f"anchor 生成器 ({self.anchor_gen.num_anchors}) 不一致"
            )

    def decode(
        self,
        preds,
        image_hw,
        score_threshold=0.05,
        iou_threshold=0.5,
        max_detections=100,
        topk_per_level=1000,
    ):
        return retinanet_decode(
            preds,
            image_hw,
            self.anchor_gen,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            topk_per_level=topk_per_level,
        )


__all__ = ["RetinaNetDetector"]
