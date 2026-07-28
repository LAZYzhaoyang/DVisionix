# -*- coding: utf-8 -*-
"""阶段 5 测试：Metrics 正确性 + MetricCollection 检测分支。"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import pytest

from dvisionix.metrics import ClassificationMetrics, DetectionMetrics, SegmentationMetrics
from dvisionix.metrics.collection import MetricCollection


class TestClassificationMetrics:
    def test_accuracy_80_percent(self):
        m = ClassificationMetrics(num_classes=5)
        # 前 8 个样本正确，后 2 个故意错
        targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 0, 0])
        pred_idx = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 1, 1])  # 8 对 2 错
        preds = torch.nn.functional.one_hot(pred_idx, 5).float() * 10
        m.update(preds, targets)
        res = m.compute()
        assert res["accuracy"] == pytest.approx(0.8, abs=0.01)


class TestSegmentationMetrics:
    def test_miou_perfect(self):
        m = SegmentationMetrics(num_classes=3)
        # 期望的类别标签
        target = torch.tensor([[
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 0, 0],
            [2, 2, 0, 0],
        ]])  # (B=1, H=4, W=4)
        # 构造完美 logits：(B, C, H, W)
        logits = torch.zeros(1, 3, 4, 4)
        for c in range(3):
            logits[0, c] = (target[0] == c).float() * 10
        m.update(logits, target)
        res = m.compute()
        assert res["mIoU"] == pytest.approx(1.0, abs=0.01)


class TestDetectionMetrics:
    def test_perfect_detection(self):
        m = DetectionMetrics(num_classes=1, iou_thresholds=[0.5])
        box = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        score = torch.tensor([0.99])
        label = torch.tensor([0])
        m.update(
            pred_boxes=[box], pred_scores=[score], pred_labels=[label],
            target_boxes=[box], target_labels=[label],
        )
        res = m.compute()
        assert res["mAP_50"] == pytest.approx(1.0, abs=0.01)

    def test_empty_detection(self):
        m = DetectionMetrics(num_classes=2, iou_thresholds=[0.5])
        m.update(
            pred_boxes=[torch.zeros((0, 4))], pred_scores=[torch.zeros(0)],
            pred_labels=[torch.zeros(0, dtype=torch.long)],
            target_boxes=[torch.zeros((0, 4))], target_labels=[torch.zeros(0, dtype=torch.long)],
        )
        res = m.compute()
        assert float(res["mAP"]) == 0.0


class TestMetricCollectionDetection:
    def test_detection_branch_dispatches(self):
        mc = MetricCollection(task_type="detection", num_classes=2)
        outputs = {
            "pred_boxes": [torch.randn(2, 4)],
            "pred_scores": [torch.rand(2)],
            "pred_labels": [torch.zeros(2, dtype=torch.long)],
        }
        batch = {
            "boxes": [torch.randn(1, 4)],
            "labels": [torch.zeros(1, dtype=torch.long)],
        }
        mc.update(outputs, batch)
        res = mc.compute()
        assert "mAP" in res