# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: YOLOv9-lite 检测器（PGI：主头 + 浅层辅助头，可逆块骨干可选）。
"""YOLOv9-lite 检测器（PGI：主头 + 浅层辅助头，可逆块骨干可选）。"""

from typing import Any, Dict, Optional

from ...registry import BACKBONES, HEADS, MODELS, NECKS
from ..base import BaseModel
from .yolo import yolo_decode


@MODELS.register()
@MODELS.register(name="yolo_v9")
class YOLOv9Detector(BaseModel):
    """YOLOv9-lite：E-ELAN/可逆骨干 + neck + 主 YOLOHead + 浅层 PGI 辅助头（仅训练）。

    forward 训练时输出主头 cls/reg 与辅助头 aux_cls/aux_reg；
    推理时辅助头不参与，decode 走主头（yolo_decode，含 NMS）。
    """

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        aux_head: Optional[Dict[str, Any]] = None,
        aux_stage_index: int = 1,
        out_indices: Optional[list] = None,
        num_classes: Optional[int] = None,
        **kwargs,
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
        head_cfg.setdefault("in_channels", self.in_channels)
        if num_classes is not None:
            head_cfg.setdefault("num_classes", num_classes)
        self.head = HEADS.build(head_cfg)
        self.strides = list(getattr(self.head, "strides", (8, 16, 32)))
        self.num_classes = getattr(self.head, "num_classes", num_classes)

        self.aux_stage_index = int(aux_stage_index)
        self.aux_head = None
        if aux_head is not None:
            aux_cfg = dict(aux_head)
            aux_cfg.setdefault("in_channels", self.backbone.out_channels[self.aux_stage_index])
            aux_cfg.setdefault("num_classes", self.num_classes)
            self.aux_head = HEADS.build(aux_cfg)

    def forward(self, x, **kwargs):
        """YOLOv9Detector 前向：x -> 训练时含主/辅头输出，推理仅主头预测。"""
        feats = self.backbone(x)
        main_feats = self.neck(feats) if self.neck is not None else feats
        out = self.head(main_feats)
        if self.aux_head is not None and self.training:
            aux = self.aux_head([feats[self.aux_stage_index]])
            out["aux_cls"] = aux["cls"]
            out["aux_reg"] = aux["reg"]
        return out

    def decode(
        self,
        preds,
        image_hw,
        score_threshold=0.05,
        iou_threshold=0.5,
        max_detections=100,
        topk_per_level=1000,
    ):
        """推理解码：preds + image_hw -> (boxes_list, scores_list, labels_list)。"""
        return yolo_decode(
            preds,
            image_hw,
            self.strides,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            topk_per_level=topk_per_level,
        )


__all__ = ["YOLOv9Detector"]
