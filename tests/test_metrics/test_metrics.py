# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: Metrics 测试：原子指标正确性 + 组合容器 + 预设 + 配置构建 + 自定义范例。
"""Metrics 测试：原子指标正确性 + 组合容器 + 预设 + 配置构建 + 自定义范例。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch

from dvisionix.metrics import (
    Accuracy,
    BaseMetric,
    ClassificationMetrics,
    DetectionMetrics,
    DiceScore,
    F1Score,
    MeanAveragePrecision,
    MeanIoU,
    MetricCollection,
    PixelAccuracy,
    Precision,
    Recall,
    TopKAccuracy,
    build_metric,
    get_preset_metrics,
)
from dvisionix.registry import METRICS


def _cls_logits():
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 0, 0])
    idx = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 1, 1])  # 8 对 2 错
    logits = torch.nn.functional.one_hot(idx, 5).float() * 10
    return logits, targets


def _seg_logits():
    target = torch.tensor(
        [
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 0, 0],
                [2, 2, 0, 0],
            ]
        ]
    )
    logits = torch.zeros(1, 3, 4, 4)
    for c in range(3):
        logits[0, c] = (target[0] == c).float() * 10
    return logits, target


class TestAtomicClassification:
    def test_accuracy(self):
        logits, targets = _cls_logits()
        m = Accuracy()
        m.update(logits, targets)
        assert m.compute() == pytest.approx(0.8, abs=0.01)

    def test_topk_accuracy(self):
        logits, targets = _cls_logits()
        m = TopKAccuracy(k=2)
        m.update(logits, targets)
        assert m.compute() == pytest.approx(1.0, abs=0.01)

    def test_precision_recall_f1_macro(self):
        logits, targets = _cls_logits()
        for cls in (Precision, Recall, F1Score):
            m = cls(num_classes=5, average="macro")
            m.update(logits, targets)
            v = m.compute()
            assert 0.0 <= v <= 1.0

    def test_f1_per_class(self):
        logits, targets = _cls_logits()
        m = F1Score(num_classes=5, average="none")
        m.update(logits, targets)
        v = m.compute()
        assert isinstance(v, list) and len(v) == 5

    def test_accumulation_across_batches(self):
        logits, targets = _cls_logits()
        m = Accuracy()
        m.update(logits[:5], targets[:5])
        m.update(logits[5:], targets[5:])
        assert m.compute() == pytest.approx(0.8, abs=0.01)


class TestAtomicSegmentation:
    def test_miou_perfect(self):
        logits, target = _seg_logits()
        m = MeanIoU(num_classes=3)
        m.update(logits, target)
        assert m.compute() == pytest.approx(1.0, abs=0.01)

    def test_pixel_accuracy_perfect(self):
        logits, target = _seg_logits()
        m = PixelAccuracy(num_classes=3)
        m.update(logits, target)
        assert m.compute() == pytest.approx(1.0, abs=0.01)

    def test_dice_perfect(self):
        logits, target = _seg_logits()
        m = DiceScore(num_classes=3)
        m.update(logits, target)
        assert m.compute() == pytest.approx(1.0, abs=0.01)


class TestAtomicDetection:
    def test_perfect(self):
        box = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        m = MeanAveragePrecision(num_classes=1, iou_thresholds=[0.5])
        m.update([box], [torch.tensor([0.99])], [torch.tensor([0])], [box], [torch.tensor([0])])
        assert m.compute()["mAP_50"] == pytest.approx(1.0, abs=0.01)

    def test_empty(self):
        m = MeanAveragePrecision(num_classes=2, iou_thresholds=[0.5])
        m.update(
            [torch.zeros((0, 4))],
            [torch.zeros(0)],
            [torch.zeros(0, dtype=torch.long)],
            [torch.zeros((0, 4))],
            [torch.zeros(0, dtype=torch.long)],
        )
        assert float(m.compute()["mAP"]) == 0.0


class TestMetricCollection:
    def test_mixed_specs(self):
        logits, targets = _cls_logits()
        mc = MetricCollection(
            [
                "accuracy",
                {"type": "f1_score", "average": "macro", "num_classes": 5},
                TopKAccuracy(k=2),
            ]
        )
        mc.update(logits, targets)
        res = mc.compute()
        assert set(["accuracy", "f1", "top2_acc"]).issubset(res.keys())
        assert res["accuracy"] == pytest.approx(0.8, abs=0.01)

    def test_reset_broadcast(self):
        logits, targets = _cls_logits()
        mc = MetricCollection([Accuracy()])
        mc.update(logits, targets)
        mc.reset()
        mc.update(logits[:5], targets[:5])
        assert mc.compute()["accuracy"] == pytest.approx(1.0, abs=0.01)

    def test_add_chaining(self):
        mc = MetricCollection(["accuracy"]).add(TopKAccuracy(k=2))
        assert len(mc) == 2


class TestPresets:
    def test_preset_function(self):
        logits, targets = _cls_logits()
        mc = get_preset_metrics("classification", num_classes=5)
        mc.update(logits, targets)
        res = mc.compute()
        assert {"accuracy", "precision", "recall", "f1"}.issubset(res.keys())

    def test_backward_compat_task_type(self):
        logits, targets = _cls_logits()
        mc = MetricCollection(task_type="classification", num_classes=5)
        mc.update(logits, targets)
        assert "accuracy" in mc.compute()

    def test_preset_class(self):
        logits, targets = _cls_logits()
        m = ClassificationMetrics(num_classes=5)
        m.update(logits, targets)
        assert m.compute()["accuracy"] == pytest.approx(0.8, abs=0.01)

    def test_detection_preset_class(self):
        box = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        m = DetectionMetrics(num_classes=1)
        m.update([box], [torch.tensor([0.99])], [torch.tensor([0])], [box], [torch.tensor([0])])
        assert "mAP" in m.compute()


class TestRegistryBuild:
    def test_build_atomic(self):
        m = build_metric({"type": "accuracy"})
        assert isinstance(m, Accuracy)

    def test_registered_names(self):
        for name in ["accuracy", "top_k_accuracy", "f1_score", "mean_iou", "map"]:
            assert name in METRICS


class TestCustomMetricExample:
    """自定义指标范例：继承 BaseMetric，实现 update/compute/reset。"""

    def test_custom_metric(self):
        class ErrorRate(BaseMetric):
            def __init__(self, name="error_rate"):
                super().__init__(name)

            def reset(self):
                self.correct = 0
                self.total = 0

            def update(self, logits, targets):
                preds = logits.argmax(dim=1)
                self.correct += int((preds == targets).sum())
                self.total += int(targets.numel())

            def compute(self):
                return 1.0 - (self.correct / self.total if self.total else 0.0)

        logits, targets = _cls_logits()
        mc = MetricCollection([Accuracy(), ErrorRate()])
        mc.update(logits, targets)
        res = mc.compute()
        assert res["accuracy"] + res["error_rate"] == pytest.approx(1.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
