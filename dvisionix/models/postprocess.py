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


def fcos_decode(
    preds,
    image_hw,
    strides,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
    topk_per_level: int = 1000,
):
    """FCOS 原始输出解码 -> (boxes_list, scores_list, labels_list)（含 NMS）。

    preds: {"cls": [...], "reg": [...], "center": [...]} 各层张量。
    """
    cls_outs, reg_outs, center_outs = preds["cls"], preds["reg"], preds["center"]
    img_h, img_w = image_hw
    B = cls_outs[0].shape[0]
    boxes_list, scores_list, labels_list = [], [], []
    for b in range(B):
        per_image_boxes, per_image_scores, per_image_labels = [], [], []
        for lvl, stride in enumerate(strides):
            cls = cls_outs[lvl][b]
            reg = reg_outs[lvl][b]
            center = center_outs[lvl][b]
            num_classes, h, w = cls.shape
            device = cls.device

            ys = (torch.arange(h, device=device) + 0.5) * stride
            xs = (torch.arange(w, device=device) + 0.5) * stride
            cx, cy = torch.meshgrid(xs, ys, indexing="xy")
            dist = torch.exp(reg) * stride
            boxes = torch.stack(
                [
                    cx - dist[0],
                    cy - dist[1],
                    cx + dist[2],
                    cy + dist[3],
                ],
                dim=-1,
            ).reshape(
                -1, 4
            )  # (H*W, 4)

            cls_prob = torch.sigmoid(cls).reshape(num_classes, -1).t()  # (N, C)
            center_prob = torch.sigmoid(center).reshape(-1)
            scores = (cls_prob * center_prob[:, None]).reshape(-1)  # (N*C,)
            n_loc = boxes.shape[0]
            topk = min(topk_per_level, scores.numel())
            top_scores, top_idx = scores.topk(topk)
            keep = top_scores >= score_threshold
            top_scores, top_idx = top_scores[keep], top_idx[keep]
            if top_idx.numel() == 0:
                continue
            labels = top_idx // n_loc
            loc_idx = top_idx % n_loc
            per_image_boxes.append(boxes[loc_idx])
            per_image_scores.append(top_scores)
            per_image_labels.append(labels)

        if not per_image_boxes:
            boxes_list.append(torch.zeros((0, 4), device=cls_outs[0].device))
            scores_list.append(torch.zeros((0,), device=cls_outs[0].device))
            labels_list.append(torch.zeros((0,), dtype=torch.long, device=cls_outs[0].device))
            continue

        boxes = torch.cat(per_image_boxes, dim=0)
        scores = torch.cat(per_image_scores, dim=0)
        labels = torch.cat(per_image_labels, dim=0)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
        keep = batched_nms(boxes, scores, labels, iou_threshold)[:max_detections]
        boxes_list.append(boxes[keep])
        scores_list.append(scores[keep])
        labels_list.append(labels[keep])
    return boxes_list, scores_list, labels_list


def retinanet_decode(
    preds,
    image_hw,
    anchor_gen,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
    topk_per_level: int = 1000,
):
    """RetinaNet 原始输出解码 -> (boxes_list, scores_list, labels_list)（含 NMS）。"""
    from .detectors.anchors import delta2bbox  # 惰性导入避免循环依赖

    cls_outs, reg_outs = preds["cls"], preds["reg"]
    img_h, img_w = image_hw
    B = cls_outs[0].shape[0]
    anchors_per_level = anchor_gen.grid_anchors(cls_outs)
    boxes_list, scores_list, labels_list = [], [], []
    for b in range(B):
        per_image_boxes, per_image_scores, per_image_labels = [], [], []
        for lvl in range(len(cls_outs)):
            cls = cls_outs[lvl][b]
            reg = reg_outs[lvl][b]
            A = anchor_gen.num_anchors
            num_classes = cls.shape[0] // A
            _, h, w = cls.shape
            n_loc = h * w
            anchors = anchors_per_level[lvl]  # (n_loc*A, 4)

            cls_flat = (
                cls.reshape(A, num_classes, h, w).permute(2, 3, 0, 1).reshape(-1, num_classes)
            )  # (n_loc*A, C)
            reg_flat = reg.reshape(A, 4, h, w).permute(2, 3, 0, 1).reshape(-1, 4)
            boxes = delta2bbox(reg_flat, anchors)

            scores = torch.sigmoid(cls_flat).reshape(-1)  # (n_loc*A*C,)
            topk = min(topk_per_level, scores.numel())
            top_scores, top_idx = scores.topk(topk)
            keep = top_scores >= score_threshold
            top_scores, top_idx = top_scores[keep], top_idx[keep]
            if top_idx.numel() == 0:
                continue
            n_anchor = n_loc * A
            labels = top_idx // n_anchor
            anchor_idx = top_idx % n_anchor
            per_image_boxes.append(boxes[anchor_idx])
            per_image_scores.append(top_scores)
            per_image_labels.append(labels)

        if not per_image_boxes:
            boxes_list.append(torch.zeros((0, 4), device=cls_outs[0].device))
            scores_list.append(torch.zeros((0,), device=cls_outs[0].device))
            labels_list.append(torch.zeros((0,), dtype=torch.long, device=cls_outs[0].device))
            continue

        boxes = torch.cat(per_image_boxes, dim=0)
        scores = torch.cat(per_image_scores, dim=0)
        labels = torch.cat(per_image_labels, dim=0)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
        keep = batched_nms(boxes, scores, labels, iou_threshold)[:max_detections]
        boxes_list.append(boxes[keep])
        scores_list.append(scores[keep])
        labels_list.append(labels[keep])
    return boxes_list, scores_list, labels_list


def yolo_decode(
    preds,
    image_hw,
    strides,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
    topk_per_level: int = 1000,
):
    """YOLO 原始输出解码 -> (boxes_list, scores_list, labels_list)（含 NMS）。"""
    cls_outs, reg_outs = preds["cls"], preds["reg"]
    img_h, img_w = image_hw
    B = cls_outs[0].shape[0]
    boxes_list, scores_list, labels_list = [], [], []
    for b in range(B):
        per_image_boxes, per_image_scores, per_image_labels = [], [], []
        for lvl, stride in enumerate(strides):
            cls = cls_outs[lvl][b]
            reg = reg_outs[lvl][b]
            num_classes, h, w = cls.shape
            device = cls.device

            ys = (torch.arange(h, device=device) + 0.5) * stride
            xs = (torch.arange(w, device=device) + 0.5) * stride
            cx, cy = torch.meshgrid(xs, ys, indexing="xy")
            ltrb = reg * stride
            boxes = torch.stack(
                [
                    cx - ltrb[0],
                    cy - ltrb[1],
                    cx + ltrb[2],
                    cy + ltrb[3],
                ],
                dim=-1,
            ).reshape(-1, 4)

            cls_prob = torch.sigmoid(cls).reshape(num_classes, -1).t()  # (N, C)
            scores = cls_prob.reshape(-1)
            n_loc = boxes.shape[0]
            topk = min(topk_per_level, scores.numel())
            top_scores, top_idx = scores.topk(topk)
            keep = top_scores >= score_threshold
            top_scores, top_idx = top_scores[keep], top_idx[keep]
            if top_idx.numel() == 0:
                continue
            labels = top_idx // n_loc
            loc_idx = top_idx % n_loc
            per_image_boxes.append(boxes[loc_idx])
            per_image_scores.append(top_scores)
            per_image_labels.append(labels)

        if not per_image_boxes:
            boxes_list.append(torch.zeros((0, 4), device=cls_outs[0].device))
            scores_list.append(torch.zeros((0,), device=cls_outs[0].device))
            labels_list.append(torch.zeros((0,), dtype=torch.long, device=cls_outs[0].device))
            continue

        boxes = torch.cat(per_image_boxes, dim=0)
        scores = torch.cat(per_image_scores, dim=0)
        labels = torch.cat(per_image_labels, dim=0)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
        keep = batched_nms(boxes, scores, labels, iou_threshold)[:max_detections]
        boxes_list.append(boxes[keep])
        scores_list.append(scores[keep])
        labels_list.append(labels[keep])
    return boxes_list, scores_list, labels_list


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


__all__ = [
    "nms",
    "batched_nms",
    "box_iou",
    "fcos_decode",
    "retinanet_decode",
    "yolo_decode",
    "detr_decode",
]


def maskformer_decode(
    preds,
    image_hw,
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    max_detections: int = 100,
):
    """MaskFormerHead full 模式解码：逐 query mask + 类别 + 分数。

    Returns:
        (masks_list, scores_list, labels_list)：每张图 (K, H, W) bool / (K,) / (K,)。
    """
    logits, masks = preds["pred_logits"], preds["pred_masks"]  # (B,Q,C+1), (B,Q,H,W)
    img_h, img_w = image_hw
    masks = masks.sigmoid()
    probs = torch.softmax(logits, dim=-1)[..., :-1]  # (B, Q, C)
    scores, labels = probs.max(dim=-1)
    masks_list, scores_list, labels_list = [], [], []
    for b in range(scores.shape[0]):
        keep = scores[b] >= score_threshold
        m = masks[b][keep] > mask_threshold
        s = scores[b][keep]
        lb = labels[b][keep]
        # 限制检测数
        if m.shape[0] > max_detections:
            _, idx = s.topk(max_detections)
            m, s, lb = m[idx], s[idx], lb[idx]
        masks_list.append(m)
        scores_list.append(s)
        labels_list.append(lb)
    return masks_list, scores_list, labels_list
