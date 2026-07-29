# -*- coding: utf-8 -*-
"""分类任务的原子指标。

提供可自由组合的分类指标：Accuracy / TopKAccuracy / Precision / Recall / F1Score。
每个指标独立累积状态，遵循 BaseMetric 的 update/compute/reset 接口。
"""

from typing import Optional

import numpy as np
import torch

from .base import BaseMetric
from ..registry import METRICS


def _to_preds(logits: torch.Tensor) -> torch.Tensor:
    """将 logits/概率 (B, C) 或 (B,) 转为预测类别索引 (B,)。"""
    if logits.dim() > 1:
        return logits.argmax(dim=1)
    return (logits > 0.5).long()


class _ConfusionMatrixMetric(BaseMetric):
    """基于混淆矩阵的指标基类（供 Precision/Recall/F1 复用）。

    Args:
        num_classes: 类别数（None 时按见到的最大标签自动推断）。
        average: 平均方式 'macro' / 'micro' / 'weighted' / 'none'。
        name: 指标名称。
    """

    def __init__(self, num_classes: Optional[int] = None, average: str = "macro", name: str = "metric"):
        self.num_classes = num_classes
        self.average = average
        super().__init__(name)

    def reset(self) -> None:
        self.confusion_matrix = (
            np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            if self.num_classes is not None
            else None
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = _to_preds(logits)
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        if self.num_classes is None:
            self.num_classes = int(max(preds_np.max(initial=0), targets_np.max(initial=0))) + 1
        if self.confusion_matrix is None:
            self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        valid = (targets_np >= 0) & (targets_np < self.num_classes)
        hist = np.bincount(
            self.num_classes * targets_np[valid].astype(np.int64) + preds_np[valid].astype(np.int64),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist

    def _prf_per_class(self):
        cm = self.confusion_matrix
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
        f1 = np.divide(
            2 * precision * recall, precision + recall,
            out=np.zeros_like(tp), where=(precision + recall) != 0,
        )
        return precision, recall, f1, tp, fp, fn

    def _reduce(self, per_class: np.ndarray, kind: str):
        if self.confusion_matrix is None:
            return 0.0
        if self.average == "none":
            return per_class.tolist()
        if self.average == "macro":
            return float(per_class.mean()) if per_class.size else 0.0
        if self.average == "weighted":
            weights = self.confusion_matrix.sum(axis=1).astype(np.float64)
            total = weights.sum()
            if total == 0:
                return 0.0
            return float((per_class * (weights / total)).sum())
        if self.average == "micro":
            precision, recall, f1, tp, fp, fn = self._prf_per_class()
            total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
            if kind == "precision":
                return float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
            if kind == "recall":
                return float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        raise ValueError(f"Unknown average: {self.average!r}")


@METRICS.register()
@METRICS.register(name="accuracy")
class Accuracy(BaseMetric):
    """Top-1 准确率。"""

    def __init__(self, name: str = "accuracy"):
        super().__init__(name)

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = _to_preds(logits)
        self.correct += int((preds == targets).sum().item())
        self.total += int(targets.numel())

    def compute(self) -> float:
        return self.correct / self.total if self.total else 0.0


@METRICS.register()
@METRICS.register(name="top_k_accuracy")
class TopKAccuracy(BaseMetric):
    """Top-K 准确率（要求输入为 (B, C) logits）。"""

    def __init__(self, k: int = 5, name: Optional[str] = None):
        self.k = k
        super().__init__(name or f"top{k}_acc")

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.dim() < 2:
            raise ValueError("TopKAccuracy 需要 (B, C) 形状的 logits。")
        k = min(self.k, logits.size(1))
        topk = logits.topk(k, dim=1).indices
        match = (topk == targets.unsqueeze(1)).any(dim=1)
        self.correct += int(match.sum().item())
        self.total += int(targets.numel())

    def compute(self) -> float:
        return self.correct / self.total if self.total else 0.0


@METRICS.register()
@METRICS.register(name="precision")
class Precision(_ConfusionMatrixMetric):
    """精确率（Precision）。"""

    def __init__(self, num_classes: Optional[int] = None, average: str = "macro", name: str = "precision"):
        super().__init__(num_classes=num_classes, average=average, name=name)

    def compute(self):
        if self.confusion_matrix is None:
            return 0.0
        precision, _, _, _, _, _ = self._prf_per_class()
        return self._reduce(precision, "precision")


@METRICS.register()
@METRICS.register(name="recall")
class Recall(_ConfusionMatrixMetric):
    """召回率（Recall）。"""

    def __init__(self, num_classes: Optional[int] = None, average: str = "macro", name: str = "recall"):
        super().__init__(num_classes=num_classes, average=average, name=name)

    def compute(self):
        if self.confusion_matrix is None:
            return 0.0
        _, recall, _, _, _, _ = self._prf_per_class()
        return self._reduce(recall, "recall")


@METRICS.register()
@METRICS.register(name="f1_score")
class F1Score(_ConfusionMatrixMetric):
    """F1 分数。"""

    def __init__(self, num_classes: Optional[int] = None, average: str = "macro", name: str = "f1"):
        super().__init__(num_classes=num_classes, average=average, name=name)

    def compute(self):
        if self.confusion_matrix is None:
            return 0.0
        _, _, f1, _, _, _ = self._prf_per_class()
        return self._reduce(f1, "f1")


__all__ = ["Accuracy", "TopKAccuracy", "Precision", "Recall", "F1Score"]