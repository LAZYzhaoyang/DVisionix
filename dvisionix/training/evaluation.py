# -*- coding: utf-8 -*-
"""
检测评估工具

将 GridDetectionModel 的原始输出解码为框，再用 DetectionMetrics 计算
COCO-style mAP / mAP_50 / mAP_75。
"""

from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..metrics import DetectionMetrics


@torch.no_grad()
def evaluate_detection(
    model,
    data_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> Dict[str, float]:
    """
    在数据集上评估检测模型的 mAP。

    要求 model 实现 decode(preds, image_hw, ...) 方法（如 GridDetectionModel），
    data_loader 使用 detection_collate（batch 含 image / boxes / labels 列表）。

    Returns:
        {"mAP": ..., "mAP_50": ..., "mAP_75": ...}
    """
    model.eval()
    metric = DetectionMetrics(num_classes=num_classes)

    for batch in data_loader:
        images = batch["image"].to(device)
        preds = model(images)
        image_hw = (images.shape[2], images.shape[3])
        boxes_list, scores_list, labels_list = model.decode(
            preds, image_hw,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        target_boxes = [b.to(device) for b in batch["boxes"]]
        target_labels = [l.to(device) for l in batch["labels"]]
        metric.update(boxes_list, scores_list, labels_list, target_boxes, target_labels)

    return metric.compute()
