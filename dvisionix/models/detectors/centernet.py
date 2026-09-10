# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: CenterNet 检测器（关键点热图峰值解码）。
"""CenterNet 检测器（关键点热图峰值解码）。"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ...registry import MODELS
from .base import SingleStageDetector


@MODELS.register()
@MODELS.register(name="centernet")
class CenterNetDetector(SingleStageDetector):
    """CenterNet：骨干单尺度特征 -> CenterNetHead（heatmap/wh/offset）-> 峰值解码。"""

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        out_indices: Optional[list] = None,
        num_classes: Optional[int] = None,
        stride: int = 4,
        topk: int = 100,
        **kwargs,
    ):
        head_cfg = dict(head)
        if num_classes is not None and "num_classes" not in head_cfg:
            head_cfg["num_classes"] = num_classes
        super().__init__(backbone, head_cfg, neck, out_indices)
        self.stride = int(stride)
        self.topk = int(topk)

    def decode(
        self,
        preds,
        image_hw,
        score_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
    ):
        """从热图取局部峰值 -> 中心（含偏移）+ 宽高 -> 像素框。

        Args:
            preds: ``CenterNetHead`` 输出 dict（``heatmap`` / ``wh`` / ``offset``）。
            image_hw: 原图 ``(H, W)``，用于把特征图坐标映射回像素坐标。
            score_threshold: 峰值分数阈值，低于该值的中心点被丢弃。
            iou_threshold: NMS 的 IoU 阈值。CenterNet 用 3x3 max-pool 做峰值抑制，
                天然保证每个目标只有一个峰，不需要额外 NMS；该参数只为满足
                ``DetectionTask.validation_step`` 的统一 decode 契约（R4）而接收，
                **不参与计算**。
            max_detections: 每张图最多保留的检测框数。
        """
        hm = preds["heatmap"].sigmoid()
        wh = preds["wh"]
        offset = preds["offset"]
        img_h, img_w = image_hw
        B = hm.shape[0]
        device = hm.device
        boxes_list, scores_list, labels_list = [], [], []

        for b in range(B):
            # 3x3 max-pool 过滤非峰值
            hm_pool = F.max_pool2d(hm[b], kernel_size=3, stride=1, padding=1)
            peak_mask = (hm[b] >= hm_pool) & (hm[b] > score_threshold)
            scores = hm[b][peak_mask]
            labels = peak_mask.nonzero(as_tuple=False)[:, 0]
            ys, xs = (
                peak_mask.nonzero(as_tuple=False)[:, 1],
                peak_mask.nonzero(as_tuple=False)[:, 2],
            )
            if scores.numel() == 0:
                boxes_list.append(torch.zeros((0, 4), device=device))
                scores_list.append(torch.zeros((0,), device=device))
                labels_list.append(torch.zeros((0,), dtype=torch.long, device=device))
                continue
            k = min(max_detections, scores.numel())
            top = scores.topk(k).indices
            scores, labels, ys, xs = scores[top], labels[top], ys[top], xs[top]
            off = offset[b][:, ys, xs].t()  # (K, 2)
            size = wh[b][:, ys, xs].t()  # (K, 2)
            cx = (xs.float() + off[:, 0]) * self.stride
            cy = (ys.float() + off[:, 1]) * self.stride
            w = size[:, 0]
            h = size[:, 1]
            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
        return boxes_list, scores_list, labels_list


__all__ = ["CenterNetDetector"]
