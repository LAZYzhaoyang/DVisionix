# -*- coding: utf-8 -*-
"""检测目标分配器（assigner）。

正负样本匹配逻辑与 Loss 解耦：assigner 只负责把 GT 框分配到网格单元，
产出 objectness / box / class 目标张量；损失组件消费这些目标。
"""

from typing import List, Tuple

import torch


class GridAssigner:
    """中心点网格分配器（配合 GridDetectionModel / GridDetectionLoss 使用）。

    规则：GT 框中心落入的网格单元为正样本；一个单元允许多个 GT（后写覆盖）。
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(
        self,
        pred_shape: Tuple[int, int, int, int],
        boxes_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        image_hw: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """生成网格目标。

        Returns:
            (obj_target, box_target, cls_target, num_pos)
            - obj_target: (B, GH, GW) float，1 表示正样本
            - box_target: (B, 4, GH, GW) float，归一化 [cx, cy, w, h]
            - cls_target: (B, GH, GW) long，-1 表示负样本
        """
        _, _, grid_h, grid_w = pred_shape
        img_h, img_w = image_hw

        obj_target = torch.zeros((len(boxes_list), grid_h, grid_w), device=device)
        box_target = torch.zeros((len(boxes_list), 4, grid_h, grid_w), device=device)
        cls_target = torch.full(
            (len(boxes_list), grid_h, grid_w), -1, dtype=torch.long, device=device
        )

        num_pos = 0
        for b in range(len(boxes_list)):
            boxes = boxes_list[b].to(device).float()
            labels = labels_list[b].to(device).long()
            for k in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[k]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = (x2 - x1).clamp(min=1e-6)
                bh = (y2 - y1).clamp(min=1e-6)
                gx = int((cx / img_w * grid_w).clamp(0, grid_w - 1).item())
                gy = int((cy / img_h * grid_h).clamp(0, grid_h - 1).item())
                obj_target[b, gy, gx] = 1.0
                box_target[b, 0, gy, gx] = cx / img_w
                box_target[b, 1, gy, gx] = cy / img_h
                box_target[b, 2, gy, gx] = bw / img_w
                box_target[b, 3, gy, gx] = bh / img_h
                cls_target[b, gy, gx] = labels[k]
                num_pos += 1

        return obj_target, box_target, cls_target, num_pos


def _box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """两组框的 IoU 矩阵：boxes1 (N,4) x boxes2 (M,4) -> (N,M)。"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter + 1e-8
    return inter / union


class FCOSAssigner:
    """FCOS 目标分配器（中心采样 + 尺度约束 + 至少一个正样本回退）。

    标签语义：-1 忽略，0 背景，1..num_classes 为类别（class c -> c+1）。
    """

    def __init__(
        self,
        num_classes: int,
        strides=(8, 16, 32, 64, 128),
        scales=(0.0, 64.0, 128.0, 256.0, 512.0, 1e10),
        center_sampling: bool = True,
        center_sample_radius: float = 1.5,
        min_pos: int = 1,
    ):
        self.num_classes = num_classes
        self.strides = list(strides)
        self.scales = list(scales)
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.min_pos = min_pos

    def assign(self, feature_shapes, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, image_hw):
        """为单张图生成各层目标。

        Args:
            feature_shapes: 每层 (H, W) 列表。
            gt_boxes: (M, 4) 像素 xyxy。
            gt_labels: (M,) 类别索引。

        Returns:
            每层 (labels, bbox_targets, reg_targets, centerness)：
            - labels: (N,) long
            - bbox_targets: (N, 4) 像素 xyxy
            - reg_targets: (N, 4) log(距离/stride)
            - centerness: (N,)
        """
        img_h, img_w = image_hw
        results = []
        for lvl, (h, w) in enumerate(feature_shapes):
            stride = self.strides[lvl]
            device = gt_boxes.device if gt_boxes.numel() else gt_labels.device
            ys = (torch.arange(h, device=device) + 0.5) * stride
            xs = (torch.arange(w, device=device) + 0.5) * stride
            cx, cy = torch.meshgrid(xs, ys, indexing="xy")
            centers = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)

            labels = torch.full((h * w,), -1, dtype=torch.long, device=device)
            bbox_targets = torch.zeros((h * w, 4), device=device)
            reg_targets = torch.zeros((h * w, 4), device=device)
            centerness = torch.zeros((h * w,), device=device)

            if gt_boxes.numel() == 0:
                labels[:] = 0
                results.append((labels, bbox_targets, reg_targets, centerness))
                continue

            gt_boxes = gt_boxes.to(device).float()
            gt_labels = gt_labels.to(device).long()
            area_min, area_max = self.scales[lvl], self.scales[lvl + 1]

            for j in range(gt_boxes.shape[0]):
                x1, y1, x2, y2 = gt_boxes[j].unbind()
                gw, gh = x2 - x1, y2 - y1
                if gw <= 0 or gh <= 0:
                    continue
                area = gw * gh
                if not (area_min <= area < area_max):
                    continue

                left = centers[:, 0] - x1
                top = centers[:, 1] - y1
                right = x2 - centers[:, 0]
                bottom = y2 - centers[:, 1]
                dist = torch.stack([left, top, right, bottom], dim=1)  # (N,4)
                inside = (dist > 0).all(dim=1)

                if self.center_sampling:
                    radius = self.center_sample_radius * stride
                    gcx, gcy = (x1 + x2) / 2, (y1 + y2) / 2
                    center_ok = (
                        (centers[:, 0] >= gcx - radius)
                        & (centers[:, 0] <= gcx + radius)
                        & (centers[:, 1] >= gcy - radius)
                        & (centers[:, 1] <= gcy + radius)
                    )
                    candidate = inside & center_ok
                else:
                    candidate = inside

                if candidate.any():
                    labels[candidate] = gt_labels[j] + 1
                    bbox_targets[candidate] = gt_boxes[j]
                    reg_targets[candidate] = torch.log(dist[candidate] / stride + 1e-8)
                    lrtb = dist[candidate].clamp(min=0)
                    min_lr = torch.min(lrtb[:, 0], lrtb[:, 2])
                    max_lr = torch.max(lrtb[:, 0], lrtb[:, 2])
                    min_tb = torch.min(lrtb[:, 1], lrtb[:, 3])
                    max_tb = torch.max(lrtb[:, 1], lrtb[:, 3])
                    centerness[candidate] = torch.sqrt(min_lr * min_tb / (max_lr * max_tb + 1e-8))

            # min_pos 回退：保证每个 gt 至少一个正样本
            if self.min_pos > 0:
                for j in range(gt_boxes.shape[0]):
                    if not (labels == (gt_labels[j] + 1)).any():
                        gcx = (gt_boxes[j, 0] + gt_boxes[j, 2]) / 2
                        gcy = (gt_boxes[j, 1] + gt_boxes[j, 3]) / 2
                        d2 = (centers[:, 0] - gcx) ** 2 + (centers[:, 1] - gcy) ** 2
                        best = d2.argmin()
                        labels[best] = gt_labels[j] + 1
                        bbox_targets[best] = gt_boxes[j]
                        x1, y1, x2, y2 = gt_boxes[j].unbind()
                        dist = torch.stack(
                            [
                                centers[best, 0] - x1,
                                centers[best, 1] - y1,
                                x2 - centers[best, 0],
                                y2 - centers[best, 1],
                            ]
                        ).clamp(
                            min=1e-6
                        )  # 框外回退位置需保证距离非负
                        reg_targets[best] = torch.log(dist / stride)
                        centerness[best] = 1.0

            labels[labels == -1] = 0
            results.append((labels, bbox_targets, reg_targets, centerness))
        return results


class MaxIoUAssigner:
    """RetinaNet 默认分配器：按最大 IoU 分配 anchor（>=pos 正样本，<neg 负样本，中间忽略）。"""

    def __init__(self, num_classes: int, pos_iou_thr: float = 0.5, neg_iou_thr: float = 0.4):
        self.num_classes = num_classes
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr

    def assign(
        self, anchors: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, image_hw
    ):
        """anchors: (N, 4) 全部层展平。

        Returns:
            (labels (N,) [-1,0..C], bbox_targets (N,4) 像素 xyxy)
        """
        img_h, img_w = image_hw
        device = anchors.device
        n = anchors.shape[0]
        labels = torch.full((n,), -1, dtype=torch.long, device=device)
        bbox_targets = torch.zeros_like(anchors)

        if gt_boxes.numel() == 0:
            labels[:] = 0
            return labels, bbox_targets

        gt_boxes = gt_boxes.to(device).float()
        gt_labels = gt_labels.to(device).long()
        iou = _box_iou_matrix(anchors, gt_boxes)
        max_iou, argmax_gt = iou.max(dim=1)
        inside = (
            (anchors[:, 0] >= 0)
            & (anchors[:, 1] >= 0)
            & (anchors[:, 2] <= img_w)
            & (anchors[:, 3] <= img_h)
        )
        labels[max_iou < self.neg_iou_thr] = 0
        pos = (max_iou >= self.pos_iou_thr) & inside
        labels[pos] = gt_labels[argmax_gt[pos]] + 1
        bbox_targets[pos] = gt_boxes[argmax_gt[pos]]

        # 每个 gt 至少分配一个最佳 anchor
        best_per_gt = iou.max(dim=0).indices
        labels[best_per_gt] = gt_labels + 1
        bbox_targets[best_per_gt] = gt_boxes

        labels[labels == -1] = 0
        return labels, bbox_targets


class ATSSAssigner:
    """自适应训练样本选择（ATSS）：按中心距离每层取 top-k，IoU mean+std 阈值选正样本。"""

    def __init__(self, num_classes: int, num_anchors: int = 9, topk: int = 9):
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.topk = topk

    def assign(
        self, anchors_per_level, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, strides, image_hw
    ):
        """anchors_per_level: 每层 (Ni, 4)。

        Returns:
            (labels (N,) [-1,0..C], bbox_targets (N,4) 像素 xyxy)
        """
        device = anchors_per_level[0].device
        all_anchors = torch.cat(anchors_per_level, dim=0)
        n = all_anchors.shape[0]
        labels = torch.zeros((n,), dtype=torch.long, device=device)
        bbox_targets = torch.zeros_like(all_anchors)

        if gt_boxes.numel() == 0:
            return labels, bbox_targets

        gt_boxes = gt_boxes.to(device).float()
        gt_labels = gt_labels.to(device).long()
        acx = (all_anchors[:, 0] + all_anchors[:, 2]) / 2
        acy = (all_anchors[:, 1] + all_anchors[:, 3]) / 2
        iou_all = _box_iou_matrix(all_anchors, gt_boxes)  # (N, M)

        cand = torch.zeros((n,), dtype=torch.bool, device=device)
        starts = [0]
        for a in anchors_per_level:
            starts.append(starts[-1] + a.shape[0])

        for j in range(gt_boxes.shape[0]):
            gcx = (gt_boxes[j, 0] + gt_boxes[j, 2]) / 2
            gcy = (gt_boxes[j, 1] + gt_boxes[j, 3]) / 2
            dist = (acx - gcx) ** 2 + (acy - gcy) ** 2
            level_cand = torch.zeros((n,), dtype=torch.bool, device=device)
            for lvl, stride in enumerate(strides):
                s, e = starts[lvl], starts[lvl + 1]
                cnt = e - s
                k = min(self.topk, cnt)
                if k <= 0:
                    continue
                topk_idx = dist[s:e].topk(k, largest=False).indices + s
                level_cand[topk_idx] = True
            if not level_cand.any():
                continue
            iou_cand = iou_all[level_cand, j]
            thr = iou_cand.mean() + iou_cand.std()
            cand |= level_cand & (iou_all[:, j] >= thr)

        if cand.any():
            best_gt = iou_all[cand].argmax(dim=1)
            labels[cand] = gt_labels[best_gt] + 1
            bbox_targets[cand] = gt_boxes[best_gt]

        labels[labels == 0] = 0  # 未选中的为背景
        return labels, bbox_targets


class TaskAlignedAssigner:
    """YOLOv8 风格任务对齐分配器。

    对齐度量 = sigmoid(cls)^alpha * IoU(pred_box, gt)^beta；按每个 gt 取 top-k 正样本，
    冲突时保留对齐度量最大的 gt。
    """

    def __init__(self, num_classes: int, topk: int = 13, alpha: float = 0.5, beta: float = 6.0):
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    def assign(self, pred_boxes, pred_scores, centers, strides, gt_boxes, gt_labels):
        """每层预测框/分数/位置中心 -> (labels_per_level, bbox_targets_per_level)。

        pred_boxes / pred_scores / centers: 每层 list。
        labels 语义：0 背景，1..num_classes 为类别（class c -> c+1）。
        """
        device = pred_boxes[0].device
        starts = [0]
        for pb in pred_boxes:
            starts.append(starts[-1] + pb.shape[0])
        n = starts[-1]
        labels_all = torch.zeros((n,), dtype=torch.long, device=device)
        bbox_t_all = torch.zeros((n, 4), device=device)
        ranges = list(zip(starts, starts[1:]))

        if gt_boxes.numel() == 0:
            return (
                [labels_all[s:e] for s, e in ranges],
                [bbox_t_all[s:e] for s, e in ranges],
            )

        all_boxes = torch.cat(pred_boxes, dim=0)
        all_scores = torch.cat(pred_scores, dim=0)
        all_centers = torch.cat(centers, dim=0)
        best_align = torch.zeros((n,), device=device)

        for j in range(gt_boxes.shape[0]):
            box = gt_boxes[j]
            inside = (
                (all_centers[:, 0] >= box[0])
                & (all_centers[:, 0] <= box[2])
                & (all_centers[:, 1] >= box[1])
                & (all_centers[:, 1] <= box[3])
            )
            iou = _box_iou_matrix(all_boxes, box[None])[:, 0]
            cls_score = torch.sigmoid(all_scores[:, gt_labels[j]])
            align = (cls_score**self.alpha) * (iou**self.beta)
            align = torch.where(inside & (iou > 0), align, torch.full_like(align, -1e9))

            k = min(self.topk, int((inside & (iou > 0)).sum().item()))
            if k <= 0:
                continue
            topk_idx = align.topk(k).indices
            top_align = align[topk_idx]
            update = top_align > best_align[topk_idx]
            for idx, flag in zip(topk_idx.tolist(), update.tolist()):
                if flag:
                    best_align[idx] = align[idx]
                    labels_all[idx] = gt_labels[j] + 1
                    bbox_t_all[idx] = box

        return (
            [labels_all[s:e] for s, e in ranges],
            [bbox_t_all[s:e] for s, e in ranges],
        )


__all__ = ["GridAssigner", "FCOSAssigner", "MaxIoUAssigner", "ATSSAssigner", "TaskAlignedAssigner"]
