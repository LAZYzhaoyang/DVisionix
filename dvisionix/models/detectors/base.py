# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 检测器装配脚手架（SingleStageDetector）。
"""检测器装配脚手架（SingleStageDetector）。

统一「backbone(features_only) + neck(可选) + head」装配逻辑，具体检测器（FCOS / RetinaNet）
继承本类并实现 forward / decode / 具体损失接入。backbone / neck / head 均可配置驱动、即插即用。
"""

from typing import Any, Dict, Optional

import torch

from ...registry import BACKBONES, HEADS, NECKS
from ..base import BaseModel
from ..postprocess import batched_nms


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

        head_type = head.get("type") if isinstance(head, dict) else str(head)
        head_cls = (
            HEADS.get(head_type) if isinstance(head_type, str) and head_type in HEADS else None
        )
        is_multi = getattr(head_cls, "input_style", "single_scale") == "multi_scale"
        head_cfg = dict(head)
        if is_multi:
            head_cfg.setdefault("in_channels_list", self._head_input_channels())
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

    def _head_input_channels(self):
        """多尺度头输入通道：有 neck 时取 neck 输出通道列表，否则取 backbone 多尺度。"""
        if self.neck is not None:
            neck_out = getattr(self.neck, "out_channels", None)
            if isinstance(neck_out, (list, tuple)):
                return list(neck_out)
            if neck_out is not None:
                num_outs = getattr(self.neck, "num_outs", None) or len(self.backbone.out_channels)
                return [neck_out] * int(num_outs)
        return list(self.backbone.out_channels)

    def forward(self, x: torch.Tensor, **kwargs):
        """原始预测（多尺度 dict），后处理由 decode 完成。"""
        return self.head(self.extract_features(x))

    def decode(self, preds, image_hw, score_threshold=0.3, iou_threshold=0.5, max_detections=100):
        """子类实现：预测 -> (boxes_list, scores_list, labels_list)。"""
        raise NotImplementedError


def detr_decode(
    preds,
    image_hw,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
    topk: int = 300,
):
    """DETR 输出解码 -> (boxes_list, scores_list, labels_list)（含 NMS）。

    preds: {"logits": (B, Q, C+1), "boxes": (B, Q, 4) 归一化 cxcywh}。
    """
    logits, boxes = preds["logits"], preds["boxes"]
    img_h, img_w = image_hw
    prob = torch.softmax(logits, dim=-1)[..., :-1]  # 去掉背景
    scores, labels = prob.max(dim=-1)  # (B, Q)

    x, y, w, h = boxes.unbind(dim=-1)
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    boxes_px = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, Q, 4)

    boxes_list, scores_list, labels_list = [], [], []
    for b in range(scores.shape[0]):
        keep = scores[b] >= score_threshold
        bboxes = boxes_px[b][keep]
        sc = scores[b][keep]
        lb = labels[b][keep]
        if bboxes.numel() > 0:
            if bboxes.shape[0] > topk:
                _, idx = sc.topk(topk)
                bboxes, sc, lb = bboxes[idx], sc[idx], lb[idx]
            keep2 = batched_nms(bboxes, sc, lb, iou_threshold)[:max_detections]
            bboxes, sc, lb = bboxes[keep2], sc[keep2], lb[keep2]
        boxes_list.append(bboxes)
        scores_list.append(sc)
        labels_list.append(lb)
    return boxes_list, scores_list, labels_list


__all__ = ["SingleStageDetector", "detr_decode"]
