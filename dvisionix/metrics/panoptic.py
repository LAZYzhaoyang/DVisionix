# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 全景分割质量（Panoptic Quality，PQ / SQ / RQ）指标。
"""全景分割质量（Panoptic Quality，PQ / SQ / RQ）指标。

遵循 Panoptic Segmentation (Kirillov et al.) 定义：按类别逐类匹配预测/真值片段
（IoU >= 0.5 贪心一一匹配），聚合 TP/FP/FN 后计算：
    PQ = sum(TP IoU) / (TP + 0.5 FP + 0.5 FN)
    SQ = sum(TP IoU) / TP
    RQ = TP / (TP + 0.5 FP + 0.5 FN)
支持 id 编码约定：segment_id = category_id * id_scale + instance_id。
"""

from typing import Dict

import numpy as np
import torch

from ..registry import METRICS
from .base import BaseMetric


@METRICS.register()
@METRICS.register(name="panoptic_quality")
class PanopticQuality(BaseMetric):
    """全景分割 PQ / SQ / RQ 指标。

    Args:
        num_categories: 类别总数（含 stuff，0 为背景）。
        id_scale: segment_id = category_id * id_scale + instance_id。
        name: 指标名。
    """

    def __init__(self, num_categories: int, id_scale: int = 1000, name: str = "PQ"):
        super().__init__(name)
        self.num_categories = int(num_categories)
        self.id_scale = int(id_scale)

    def reset(self) -> None:
        """重置全景评估累积状态。"""
        self._tp_iou: Dict[int, float] = {}
        self._tp: Dict[int, int] = {}
        self._fp: Dict[int, int] = {}
        self._fn: Dict[int, int] = {}

    def update(self, pred_ids: torch.Tensor, gt_ids: torch.Tensor) -> None:
        """喂入一张全景图（或 batch 拆开逐张调用）。pred_ids / gt_ids: (H, W) int64。"""
        pred = pred_ids.detach().cpu().numpy().astype(np.int64)
        gt = gt_ids.detach().cpu().numpy().astype(np.int64)
        pred_cat = pred // self.id_scale
        gt_cat = gt // self.id_scale
        categories = set(np.unique(pred_cat)) | set(np.unique(gt_cat))
        categories = {c for c in categories if 0 <= c < self.num_categories}

        for cat in categories:
            pm = pred[pred_cat == cat]
            gm = gt[gt_cat == cat]
            pred_inst = np.unique(pm)
            gt_inst = np.unique(gm)
            p_masks = [pred == i for i in pred_inst]
            g_masks = [gt == j for j in gt_inst]
            if not p_masks or not g_masks:
                self._fp[cat] = self._fp.get(cat, 0) + len(p_masks)
                self._fn[cat] = self._fn.get(cat, 0) + len(g_masks)
                continue

            p_flat = np.stack([m.reshape(-1) for m in p_masks])  # (P, HW)
            g_flat = np.stack([m.reshape(-1) for m in g_masks])  # (G, HW)
            inter = (p_flat[:, None, :] & g_flat[None, :, :]).sum(axis=2)  # (P, G)
            union = (p_flat[:, None, :].sum(axis=2) + g_flat[None, :, :].sum(axis=2) - inter).clip(
                min=1
            )
            iou = inter / union  # (P, G)

            matched_gt = set()
            tp_iou = 0.0
            for pi in range(len(p_masks)):
                candidates = [
                    gi for gi in range(len(g_masks)) if gi not in matched_gt and iou[pi, gi] >= 0.5
                ]
                if not candidates:
                    continue
                gi = max(candidates, key=lambda g: iou[pi, g])
                matched_gt.add(gi)
                tp_iou += float(iou[pi, gi])

            self._tp[cat] = self._tp.get(cat, 0) + len(matched_gt)
            self._fp[cat] = self._fp.get(cat, 0) + len(p_masks) - len(matched_gt)
            self._fn[cat] = self._fn.get(cat, 0) + len(g_masks) - len(matched_gt)
            self._tp_iou[cat] = self._tp_iou.get(cat, 0.0) + tp_iou

    def compute(self) -> Dict[str, float]:
        """计算 PQ / SQ / RQ 三项全景指标。"""
        cats = [
            c
            for c in set(self._tp) | set(self._fp) | set(self._fn)
            if (self._tp.get(c, 0) + self._fp.get(c, 0) + self._fn.get(c, 0)) > 0
        ]
        if not cats:
            return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0}
        pq_sum = 0.0
        rq_sum = 0.0
        sq_num = 0.0
        sq_den = 0.0
        for c in cats:
            tp = self._tp.get(c, 0)
            fp = self._fp.get(c, 0)
            fn = self._fn.get(c, 0)
            denom = tp + 0.5 * fp + 0.5 * fn
            pq_sum += self._tp_iou.get(c, 0.0) / denom if denom > 0 else 0.0
            rq_sum += tp / denom if denom > 0 else 0.0
            if tp > 0:
                sq_num += self._tp_iou.get(c, 0.0) / tp
                sq_den += 1.0
        return {
            "PQ": float(pq_sum / len(cats)),
            "SQ": float(sq_num / sq_den) if sq_den > 0 else 0.0,
            "RQ": float(rq_sum / len(cats)),
        }


__all__ = ["PanopticQuality"]
