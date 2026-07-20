# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\metrics\\classification.py

"""
分类任务指标

计算准确率、精确率、召回率、F1 分数等分类指标。
"""

import torch
import numpy as np
from typing import Dict, Optional


class ClassificationMetrics:
    """分类指标计算器"""
    
    def __init__(self, num_classes: Optional[int] = None, average: str = "macro"):
        """
        Args:
            num_classes: 类别数量（None 表示自动推断）
            average: 平均方式: 'micro', 'macro', 'weighted', 'none'
        """
        self.num_classes = num_classes
        self.average = average
        
        # 累积统计
        self.reset()
    
    def reset(self) -> None:
        """重置统计"""
        self.total = 0
        self.correct = 0
        self.confusion_matrix = None
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新统计
        
        Args:
            logits: 模型输出 (B, C) 或 (B,)
            targets: 目标标签 (B,)
        """
        if logits.dim() > 1:
            preds = logits.argmax(dim=1)
        else:
            preds = (logits > 0.5).long()
        
        self.total += targets.size(0)
        self.correct += (preds == targets).sum().item()
        
        # 更新混淆矩阵
        if self.num_classes is None:
            self.num_classes = max(preds.max().item(), targets.max().item()) + 1
        
        if self.confusion_matrix is None:
            self.confusion_matrix = np.zeros(
                (self.num_classes, self.num_classes), dtype=np.int64
            )
        
        for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
            self.confusion_matrix[t, p] += 1
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            包含各指标的字典
        """
        if self.total == 0:
            return {"accuracy": 0.0}
        
        accuracy = self.correct / self.total
        result = {"accuracy": accuracy}
        
        if self.confusion_matrix is not None:
            # 计算每类的 precision, recall, f1
            tp = np.diag(self.confusion_matrix)
            fp = self.confusion_matrix.sum(axis=0) - tp
            fn = self.confusion_matrix.sum(axis=1) - tp
            
            # 避免除以零
            precision = np.divide(
                tp, tp + fp,
                out=np.zeros_like(tp, dtype=np.float64),
                where=(tp + fp) != 0
            )
            recall = np.divide(
                tp, tp + fn,
                out=np.zeros_like(tp, dtype=np.float64),
                where=(tp + fn) != 0
            )
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision, dtype=np.float64),
                where=(precision + recall) != 0
            )
            
            if self.average == "micro":
                total_tp = tp.sum()
                total_fp = fp.sum()
                total_fn = fn.sum()
                result["precision"] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                result["recall"] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                result["f1"] = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"]) if (result["precision"] + result["recall"]) > 0 else 0
            elif self.average == "macro":
                result["precision"] = precision.mean()
                result["recall"] = recall.mean()
                result["f1"] = f1.mean()
            elif self.average == "weighted":
                weights = self.confusion_matrix.sum(axis=1)
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                result["precision"] = (precision * weights).sum()
                result["recall"] = (recall * weights).sum()
                result["f1"] = (f1 * weights).sum()
            elif self.average == "none":
                result["precision_per_class"] = precision.tolist()
                result["recall_per_class"] = recall.tolist()
                result["f1_per_class"] = f1.tolist()
        
        return result
