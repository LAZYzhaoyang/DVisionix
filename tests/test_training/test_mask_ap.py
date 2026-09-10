# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: mask AP 目标尺寸契约测试（空预测时不得把 GT 缩成 1x1）。
"""mask AP 目标尺寸契约测试（CodePlan 7.4 步骤 2-4 / 7.1.2 D11）。

``evaluate_mask_ap`` 需要把 GT mask 插值到模型输出分辨率后再喂给指标。
v1.0.0 的写法是：

    target_size = masks_list[0].shape[-2:] if masks_list and masks_list[0].numel() else (1, 1)

当**第一张图没有预测 mask** 时（很常见：置信度阈值过滤后为空），目标尺寸会退化成
``(1, 1)`` —— GT 被缩成单像素，mask AP 随之完全失真，而且不会有任何报错。

这里用一个"首图空预测"的假模型把这个退化路径固定下来。
"""

import pytest
import torch
from torch.utils.data import DataLoader

from dvisionix.training import evaluation as ev

IMAGE_HW = (16, 16)
NUM_CLASSES = 3


class _FakeMaskModel(torch.nn.Module):
    """首图无预测、其余图各返回一个掩码的假模型。"""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        b, _, h, w = x.shape
        return {"dummy": torch.zeros(b, 1, h, w)}

    def decode(self, preds, image_hw, score_threshold=0.3, mask_threshold=0.5):
        batch = preds["dummy"].shape[0]
        # 关键：首图给出 **空且形状为 (0,1,1)** 的预测，用于复现 v1.0.0 的退化分支
        masks = [torch.zeros(0, 1, 1)]
        scores = [torch.zeros(0)]
        labels = [torch.zeros(0, dtype=torch.long)]
        for _ in range(batch - 1):
            masks.append(torch.zeros(1, *image_hw))
            scores.append(torch.tensor([0.9]))
            labels.append(torch.tensor([0]))
        return masks, scores, labels


def _loader(n: int = 3):
    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return {
                "image": torch.zeros(3, *IMAGE_HW),
                "mask": torch.zeros(*IMAGE_HW, dtype=torch.long),
                "labels": torch.tensor([0]),
            }

    return DataLoader(_DS(), batch_size=n)


@pytest.mark.unit
def test_target_size_is_image_sized_when_first_image_has_no_predictions(monkeypatch):
    captured = {}

    class _SpyMetric:
        def __init__(self, num_classes=0, **kwargs):
            self.num_classes = num_classes

        def update(self, masks, scores, labels, targets, target_labels):
            captured["target_shapes"] = [tuple(t.shape) for t in targets]

        def compute(self):
            return {}

    monkeypatch.setattr("dvisionix.metrics.MaskAveragePrecision", _SpyMetric)

    ev.evaluate_mask_ap(
        _FakeMaskModel(),
        _loader(3),
        num_classes=NUM_CLASSES,
        device=torch.device("cpu"),
    )

    shapes = captured.get("target_shapes")
    assert shapes, "指标未被调用"
    assert all(
        shape[-2:] == IMAGE_HW for shape in shapes
    ), f"GT mask 未对齐到输入图像尺寸：{shapes}（v1.0.0 会退化成 (1, 1)）"
    assert (1, 1) not in [shape[-2:] for shape in shapes], shapes
