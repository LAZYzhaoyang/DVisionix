# -*- coding: utf-8 -*-
"""检测推理解码、NMS 与 mAP 评估的单元测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader, Dataset

from dvisionix.data import detection_collate
from dvisionix.models import GridDetectionModel, batched_nms, box_iou, nms
from dvisionix.training import evaluate_detection


def test_box_iou():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]])
    iou = box_iou(a, b)
    assert abs(iou[0, 0].item() - 1.0) < 1e-5
    assert iou[0, 1].item() < 1e-5


def test_nms_removes_overlaps():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.5, 0.5, 10.5, 10.5],  # 与第一个高度重叠
            [50.0, 50.0, 60.0, 60.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert 0 in keep.tolist()
    assert 2 in keep.tolist()
    assert 1 not in keep.tolist()


def test_batched_nms_keeps_diff_classes():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.5, 0.5, 10.5, 10.5],
        ]
    )
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 1])  # 不同类别，不应互相抑制
    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
    assert len(keep) == 2


def test_decode_shapes():
    model = GridDetectionModel(num_classes=3)
    out = model(torch.randn(2, 3, 64, 64))
    boxes, scores, labels = model.decode(out, (64, 64), score_threshold=0.0)
    assert len(boxes) == len(scores) == len(labels) == 2
    assert boxes[0].shape[1] == 4


class _DetDS(Dataset):
    def __init__(self, n=8, num_classes=3, size=64):
        self.n, self.num_classes, self.size = n, num_classes, size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "image": torch.randn(3, self.size, self.size),
            "boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]]),
            "labels": torch.tensor([i % self.num_classes]),
        }


def test_evaluate_detection_runs():
    model = GridDetectionModel(num_classes=3)
    ds = _DetDS()
    loader = DataLoader(ds, batch_size=4, collate_fn=detection_collate)
    result = evaluate_detection(
        model, loader, num_classes=3, device=torch.device("cpu"), score_threshold=0.0
    )
    assert "mAP" in result and "mAP_50" in result and "mAP_75" in result
    for v in result.values():
        assert 0.0 <= v <= 1.0


def test_map_perfect_and_wrong():
    from dvisionix.metrics import DetectionMetrics

    tb = [torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]])]
    tl = [torch.tensor([0, 1])]

    perfect = DetectionMetrics(num_classes=2)
    perfect.update(
        [torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]])],
        [torch.tensor([0.95, 0.9])],
        [torch.tensor([0, 1])],
        tb,
        tl,
    )
    r = perfect.compute()
    assert r["mAP_50"] > 0.99

    wrong = DetectionMetrics(num_classes=2)
    wrong.update(
        [torch.tensor([[100.0, 100.0, 110.0, 110.0]])],
        [torch.tensor([0.9])],
        [torch.tensor([0])],
        tb,
        tl,
    )
    assert wrong.compute()["mAP_50"] == 0.0


if __name__ == "__main__":
    print("Running detection eval tests...")
    test_box_iou()
    print("ok test_box_iou")
    test_nms_removes_overlaps()
    print("ok test_nms_removes_overlaps")
    test_batched_nms_keeps_diff_classes()
    print("ok test_batched_nms_keeps_diff_classes")
    test_decode_shapes()
    print("ok test_decode_shapes")
    test_evaluate_detection_runs()
    print("ok test_evaluate_detection_runs")
    test_map_perfect_and_wrong()
    print("ok test_map_perfect_and_wrong")
    print("All detection eval tests passed!")
