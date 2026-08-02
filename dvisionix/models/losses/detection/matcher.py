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

    先将代价矩阵补齐为方阵（缺失行列用大常数），保证存在完备匹配，避免 n>m
    时无增广路径的死循环；最后丢弃填充的伪匹配。

    Args:
        cost: (n, m) 代价矩阵。

    Returns:
        (row_ind, col_ind)：匹配的行/列索引（长度为 min(n, m)）。
    """
    orig_n, orig_m = cost.shape
    n, m = cost.shape
    size = max(n, m)
    if size > n or size > m:
        padded = np.full((size, size), 1e9, dtype=np.float64)
        padded[:n, :m] = cost
    else:
        padded = cost
    n = size
    m = size
    cost = padded

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
        if p[j] != 0 and (p[j] - 1) < orig_n and (j - 1) < orig_m:
            row_idx.append(p[j] - 1)
            col_idx.append(j - 1)
    return np.array(row_idx, dtype=np.int64), np.array(col_idx, dtype=np.int64)


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
