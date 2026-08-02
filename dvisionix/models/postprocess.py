# -*- coding: utf-8 -*-
"""
检测后处理：NMS（非极大值抑制）

提供纯 PyTorch 实现的单类与多类 NMS，避免依赖 torchvision.ops。
"""

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算两组框的 IoU 矩阵。

    Args:
        boxes1: (N, 4) [x1, y1, x2, y2]
        boxes2: (M, 4) [x1, y1, x2, y2]

    Returns:
        IoU 矩阵 (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter + 1e-8
    return inter / union


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    单类 NMS。

    Args:
        boxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,)
        iou_threshold: IoU 抑制阈值

    Returns:
        保留框的索引（按分数降序）(K,)
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        remain = (ious <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
        order = order[remain + 1]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    多类 NMS：不同类别互不抑制（通过坐标偏移实现）。

    Args:
        boxes: (N, 4)
        scores: (N,)
        labels: (N,)
        iou_threshold: IoU 阈值

    Returns:
        保留框的索引 (K,)
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    max_coord = boxes.max() if boxes.numel() > 0 else torch.tensor(0.0)
    offsets = labels.to(boxes) * (max_coord + 1)
    boxes_offset = boxes + offsets[:, None]
    return nms(boxes_offset, scores, iou_threshold)
