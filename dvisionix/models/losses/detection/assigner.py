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
        cls_target = torch.full((len(boxes_list), grid_h, grid_w), -1, dtype=torch.long, device=device)

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


__all__ = ["GridAssigner"]