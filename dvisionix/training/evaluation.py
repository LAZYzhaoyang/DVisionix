# -*- coding: utf-8 -*-
"""
检测 / 掩码评估工具

- evaluate_detection：检测模型 decode 后用 DetectionMetrics 计算 COCO mAP。
- evaluate_mask_ap：MaskFormerHead full 模式 -> mask mAP。
- panoptic_decode / evaluate_panoptic：full 模式 -> 全景 id 图 -> PanopticQuality（PQ/SQ/RQ）。
"""

from typing import Dict

import torch
import torch.nn.functional as F
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
    from ..models.postprocess import maskformer_decode

    model.eval()
    metric = MaskAveragePrecision(num_classes=num_classes)
    for batch in data_loader:
        images = batch["image"].to(device)
        preds = model(images)
        if not isinstance(preds, dict):
            raise ValueError(
                "evaluate_mask_ap 需要模型输出 dict（MaskFormerHead output_mode='full'）"
            )
        decode_fn = getattr(model, "decode", None) or maskformer_decode
        masks_list, scores_list, labels_list = decode_fn(
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
            ).bool()  # (1, H, W)
            for m in batch["mask"]
        ]
        target_labels = [
            lb.to(device)
            for lb in batch.get("labels", [torch.tensor([1], device=device) for _ in target_masks])
        ]
        metric.update(masks_list, scores_list, labels_list, target_masks, target_labels)
    return metric.compute()


@torch.no_grad()
def panoptic_decode(
    preds,
    image_hw,
    num_classes: int,
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    max_instances: int = 100,
    id_scale: int = 1000,
):
    """MaskFormerHead full 模式 -> 每张图的全景 id 图列表（(H, W) int64）。

    id 编码：category_id * id_scale + instance_id；实例掩码覆盖语义预测（实例优先）。
    """
    logits = preds["pred_logits"]
    masks = preds["pred_masks"].sigmoid()
    semantic = preds.get("semantic_logits")
    probs = torch.softmax(logits, dim=-1)[..., :-1]  # (B, Q, C)
    scores, labels = probs.max(dim=-1)
    pan_list = []
    for b in range(logits.shape[0]):
        if semantic is not None:
            sem = semantic[b].argmax(0)
        else:
            sem = torch.einsum("qc,qhw->chw", probs[b], masks[b]).argmax(0)
        pan = sem * id_scale
        keep = scores[b] >= score_threshold
        idx = scores[b][keep].argsort(descending=True)
        if idx.numel() > max_instances:
            idx = idx[:max_instances]
        inst_id = 1
        for q in idx.tolist():
            m = masks[b, q] > mask_threshold
            pan[m] = int(labels[b, q]) * id_scale + inst_id
            inst_id += 1
        if tuple(pan.shape) != tuple(image_hw):
            pan = (
                F.interpolate(
                    pan.unsqueeze(0).unsqueeze(0).float(),
                    size=tuple(image_hw),
                    mode="nearest",
                )
                .long()
                .squeeze(0)
                .squeeze(0)
            )
        pan_list.append(pan)
    return pan_list


@torch.no_grad()
def evaluate_panoptic(
    model,
    data_loader: "DataLoader",
    num_classes: int,
    device: "torch.device",
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    gt_key: str = "panoptic",
    id_scale: int = 1000,
) -> dict:
    """MaskFormer 风格模型的全景分割评估（PQ / SQ / RQ）。

    模型输出需为 full 模式 dict；GT 取 batch["panoptic"]（全景 id 图），
    若缺失则退化为 batch["mask"] * id_scale（纯语义）。
    """
    from ..metrics import PanopticQuality

    model.eval()
    metric = PanopticQuality(num_categories=num_classes, id_scale=id_scale)
    for batch in data_loader:
        images = batch["image"].to(device)
        preds = model(images)
        pan_preds = panoptic_decode(
            preds,
            (images.shape[2], images.shape[3]),
            num_classes,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            id_scale=id_scale,
        )
        for i, pp in enumerate(pan_preds):
            if gt_key in batch:
                gt = batch[gt_key][i].to(device)
            else:
                gt = batch["mask"][i].to(device) * id_scale
            metric.update(pp, gt)
    return metric.compute()
