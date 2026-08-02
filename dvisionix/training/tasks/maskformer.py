# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: MaskFormer 实例分割任务组件（MaskFormerHead full 模式 + MaskFormerLoss ...
"""MaskFormer 实例分割任务组件（MaskFormerHead full 模式 + MaskFormerLoss + mask mAP）。"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...metrics import MaskAveragePrecision, PanopticQuality
from ...models.losses import MaskFormerLoss, compute_loss
from ..evaluation import panoptic_decode
from .base import BaseTask, _merge_legacy_hyperparams


class MaskFormerTask(BaseTask):
    """实例分割任务（Mask2Former 风格）。

    要求模型输出 full 模式 dict（MaskFormerHead output_mode="full"，或 SegmentationModel 透传）：
    {"pred_logits": (B, Q, C+1), "pred_masks": (B, Q, H, W), "semantic_logits": (B, C, H, W)}。

    默认损失：MaskFormerLoss（匈牙利匹配 + CE + mask BCE + Dice）；默认指标：MaskAveragePrecision（mask mAP）。
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        loss: Any = None,
        metrics: Any = None,
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        max_detections: int = 100,
        panoptic: bool = False,
        id_scale: int = 1000,
    ):
        super().__init__(optimizer_cfg, scheduler_cfg, loss=loss, metrics=metrics)
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.max_detections = max_detections
        self.panoptic = bool(panoptic)
        self.id_scale = int(id_scale)
        self.optimizer_cfg = _merge_legacy_hyperparams(
            self.optimizer_cfg, learning_rate, weight_decay
        )
        if self.loss is None:
            self.loss = MaskFormerLoss(num_classes=num_classes)
        if self.metrics is None:
            self.metrics = MaskAveragePrecision(num_classes=num_classes)
        self._panoptic_metric = (
            PanopticQuality(num_categories=num_classes, id_scale=self.id_scale)
            if self.panoptic
            else None
        )

    def training_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """MaskFormer 训练步：mask 监督损失。"""
        images = batch["image"].to(device)
        preds = model(images)
        if not isinstance(preds, dict):
            raise ValueError(
                "MaskFormerTask 需要模型输出 dict（MaskFormerHead output_mode='full'）"
            )
        loss, extras = compute_loss(self.loss, preds, batch, device=device)
        return {"loss": loss, **extras}

    def validation_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """MaskFormer 验证步：mask/全景指标更新。"""
        images = batch["image"].to(device)
        with torch.no_grad():
            preds = model(images)
            image_hw = (images.shape[2], images.shape[3])
            loss, extras = compute_loss(self.loss, preds, batch, device=device)
            masks_list, scores_list, labels_list = model.decode(
                preds,
                image_hw,
                score_threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                max_detections=self.max_detections,
            )
            pred_hw = (
                tuple(masks_list[0].shape[-2:]) if masks_list and masks_list[0].numel() else None
            )
            if batch.get("instance_masks") is not None:
                target_masks = [im.to(device) for im in batch["instance_masks"]]
                target_labels = [lb.to(device) for lb in batch["instance_labels"]]
            else:
                # 语义掩码退化：每图一个"实例"（整张掩码 + 默认类别 1）
                target_masks = [m.to(device).unsqueeze(0) for m in batch["mask"]]
                target_labels = [torch.tensor([1], device=device) for _ in target_masks]
            if pred_hw is not None:
                target_masks = [
                    (
                        F.interpolate(m.float().unsqueeze(0), size=pred_hw, mode="nearest")
                        .bool()
                        .squeeze(0)
                    )
                    for m in target_masks
                ]
        out = {
            "loss": loss,
            **extras,
            "preds": (masks_list, scores_list, labels_list),
            "targets": (target_masks, target_labels),
        }
        if self.panoptic:
            pans = panoptic_decode(
                preds,
                image_hw,
                self.num_classes,
                score_threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                id_scale=self.id_scale,
            )
            pan_targets = [
                (
                    batch["panoptic"][i].to(device)
                    if "panoptic" in batch
                    else batch["mask"][i].to(device) * self.id_scale
                )
                for i in range(len(pans))
            ]
            out["preds"] = out["preds"] + (pans,)
            out["targets"] = out["targets"] + (pan_targets,)
        return out

    def update_metrics(self, preds: Any, targets: Any) -> None:
        """用 (preds, targets) 更新 mask/实例指标。"""
        if self.metrics is None:
            return
        if self.panoptic:
            masks_list, scores_list, labels_list, pans = preds
            target_masks, target_labels, pan_targets = targets
            self.metrics.update(masks_list, scores_list, labels_list, target_masks, target_labels)
            for pp, gt in zip(pans, pan_targets):
                self._panoptic_metric.update(pp, gt)
        else:
            masks_list, scores_list, labels_list = preds
            target_masks, target_labels = targets
            self.metrics.update(masks_list, scores_list, labels_list, target_masks, target_labels)

    def on_validation_epoch_end(self) -> Dict[str, float]:
        """验证结束：计算全景（PQ/SQ/RQ）或 mask 指标。"""
        result = self.metrics.compute() if self.metrics is not None else {}
        if self.metrics is not None:
            self.metrics.reset()
        if self._panoptic_metric is not None:
            result.update(self._panoptic_metric.compute())
            self._panoptic_metric.reset()
        return result

    def reset_metrics(self) -> None:
        """重置 mask/全景指标。"""
        super().reset_metrics()
        panoptic_metric = getattr(self, "_panoptic_metric", None)
        if panoptic_metric is not None:
            panoptic_metric.reset()


__all__ = ["MaskFormerTask"]
