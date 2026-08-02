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
            preds,
            image_hw,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        target_boxes = [b.to(device) for b in batch["boxes"]]
        target_labels = [lb.to(device) for lb in batch["labels"]]
        metric.update(boxes_list, scores_list, labels_list, target_boxes, target_labels)

    return metric.compute()


@torch.no_grad()
def evaluate_mask_ap(
    model,
    data_loader: "DataLoader",
    num_classes: int,
    device: "torch.device",
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
) -> dict:
    """MaskFormer 风格模型的 mask mAP 评估（模型输出需为 full 模式 dict）。

    使用 dvisionix.metrics.MaskAveragePrecision。
    """
    from ..metrics import MaskAveragePrecision
    from ..models.heads.segmentation.maskformer import maskformer_decode

    model.eval()
    metric = MaskAveragePrecision(num_classes=num_classes)
    for batch in data_loader:
        images = batch["image"].to(device)
        preds = model(images)
        if not isinstance(preds, dict):
            raise ValueError(
                "evaluate_mask_ap 需要模型输出 dict（MaskFormerHead output_mode='full'）"
            )
        masks_list, scores_list, labels_list = maskformer_decode(
            preds,
            (images.shape[2], images.shape[3]),
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )
        # 目标 mask 对齐预测分辨率
        target_size = masks_list[0].shape[-2:] if masks_list and masks_list[0].numel() else (1, 1)
        import torch.nn.functional as _F

        target_masks = [
            _F.interpolate(
                m.to(device).float().unsqueeze(0).unsqueeze(0), size=target_size, mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
            .bool()
            for m in batch["mask"]
        ]
        target_labels = [
            lb.to(device)
            for lb in batch.get("labels", [torch.full_like(m, 1) for m in target_masks])
        ]
        metric.update(masks_list, scores_list, labels_list, target_masks, target_labels)
    return metric.compute()
