# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\metrics\\segmentation.py

"""
分割任务指标

计算 mIoU (mean Intersection over Union)、像素准确率等分割指标。
"""

import torch
import numpy as np
from typing import Dict, Optional


class SegmentationMetrics:
    """分割指标计算器"""
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        compute_per_class: bool = False,
    ):
        """
        Args:
            num_classes: 类别数量
            ignore_index: 忽略的类别标签
            compute_per_class: 是否计算每类指标
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.compute_per_class = compute_per_class
        
        self.reset()
    
    def reset(self) -> None:
        """重置统计"""
        # 混淆矩阵: true_classes x pred_classes
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新统计
        
        Args:
            logits: 模型输出 (B, C, H, W)
            targets: 目标标签 (B, H, W)
        """
        preds = logits.argmax(dim=1)
        
        # 转换为 numpy
        preds_np = preds.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        
        # 忽略指定类别
        if self.ignore_index is not None:
            mask = (targets_np != self.ignore_index)
            preds_np = preds_np[mask]
            targets_np = targets_np[mask]
        
        # 过滤无效类别
        mask = (targets_np >= 0) & (targets_np < self.num_classes)
        preds_np = preds_np[mask]
        targets_np = targets_np[mask]
        
        # 更新混淆矩阵
        hist = np.bincount(
            self.num_classes * targets_np + preds_np,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            包含各指标的字典
        """
        # 计算 IoU
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0) -
            intersection
        )
        
        # 避免除以零
        valid = union > 0
        if not np.any(valid):
            result = {
                "mIoU": 0.0,
                "pixel_accuracy": 0.0,
            }
            if self.compute_per_class:
                result["iou_per_class"] = [0.0] * self.num_classes
            return result
        
        iou_per_class = np.zeros(self.num_classes, dtype=np.float64)
        iou_per_class[valid] = intersection[valid] / union[valid]
        
        # 平均 IoU
        mean_iou = np.mean(iou_per_class[valid])
        
        # 像素准确率
        total_pixels = self.confusion_matrix.sum()
        correct_pixels = intersection.sum()
        pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        result = {
            "mIoU": float(mean_iou),
            "pixel_accuracy": float(pixel_accuracy),
        }
        
        if self.compute_per_class:
            result["iou_per_class"] = iou_per_class.tolist()
        
        return result
