# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 匈牙利匹配器（DETR set-based 损失用）。
"""匈牙利匹配器（DETR set-based 损失用）。"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """最小代价二分匹配（Kuhn-Munkres / 匈牙利算法，numpy 实现）。

    实现要点（性能关键）：

    - 该实现把每一**行**分配给一个**列**，因此要求 ``行数 <= 列数``；
      否则先转置、求解、再把索引换回来。
    - 因此**不需要把矩阵补成方阵**。v1.0.0 无论形状如何都补齐到
      ``max(n, m) × max(n, m)``，于是 DETR 的 ``300 queries × 30 GT``
      会变成一个 300×300 的问题，而算法是纯 Python 三重循环的 O(n²m)：
      实测单次匹配需要 **13.6 秒**。转置后是 30×300，规模下降两个数量级。

    Args:
        cost: (n, m) 代价矩阵。

    Returns:
        (row_ind, col_ind)：匹配的行/列索引（长度为 min(n, m)）。
    """
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = np.ascontiguousarray(cost.T)
    n, m = cost.shape  # 此时必然 n <= m

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_idx, col_idx = [], []
    for j in range(1, m + 1):
        if p[j] != 0:
            row_idx.append(p[j] - 1)
            col_idx.append(j - 1)
    rows = np.array(row_idx, dtype=np.int64)
    cols = np.array(col_idx, dtype=np.int64)
    if transposed:
        return cols, rows
    return rows, cols


class HungarianMatcher:
    """DETR 匈牙利匹配器：按分类代价 + L1 + GIoU 代价最小化匹配 query 与 gt。"""

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def __call__(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ):
        """单张图：pred_logits (Q, C+1)，pred_boxes (Q, 4) 归一化 cxcywh。

        Returns:
            (pred_idx, gt_idx)：匹配索引（Tensor）。
        """
        if gt_boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=pred_logits.device), torch.empty(
                (0,), dtype=torch.long, device=pred_logits.device
            )

        q, c = pred_logits.shape

        # 分类代价：-log_softmax 中对应 gt 类别
        out_prob = F.log_softmax(pred_logits, dim=-1)  # (Q, C+1)
        cost_cls = -out_prob[:, gt_labels]  # (Q, M)

        # L1 代价（归一化 cxcywh）
        cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)  # (Q, M)

        # GIoU 代价
        def _xywh_to_xyxy(b):
            x, y, w, h = b.unbind(dim=-1)
            return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)

        pb = _xywh_to_xyxy(pred_boxes)
        gb = _xywh_to_xyxy(gt_boxes)
        # 简单 IoU 矩阵（归一化坐标下）
        inter = torch.clamp(
            torch.min(pb[:, None, 2:], gb[None, :, 2:])
            - torch.max(pb[:, None, :2], gb[None, :, :2]),
            min=0,
        )
        iw = inter[..., 0]
        ih = inter[..., 1]
        inter_area = iw * ih
        area_p = torch.clamp(pb[:, 2] - pb[:, 0], min=0) * torch.clamp(pb[:, 3] - pb[:, 1], min=0)
        area_g = torch.clamp(gb[:, 2] - gb[:, 0], min=0) * torch.clamp(gb[:, 3] - gb[:, 1], min=0)
        union = area_p[:, None] + area_g[None, :] - inter_area
        iou = inter_area / (union + 1e-8)
        cost_giou = 1 - iou

        cost = self.cost_class * cost_cls + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        row, col = _hungarian(cost.detach().cpu().numpy())
        return (
            torch.as_tensor(row, dtype=torch.long, device=pred_logits.device),
            torch.as_tensor(col, dtype=torch.long, device=pred_logits.device),
        )


__all__ = ["HungarianMatcher"]
