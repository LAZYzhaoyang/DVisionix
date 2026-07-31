# -*- coding: utf-8 -*-
"""models.losses 模块测试：基类 / 组合 / 构建 / 数值正确性。"""

import pytest
import torch

from dvisionix.models.losses import (
    BaseLoss,
    LossComposer,
    build_loss,
    build_losses,
    compute_loss,
    CrossEntropy,
    FocalLoss,
    DiceLoss,
    CombinedSegmentationLoss,
    GIoULoss,
    CIoULoss,
    L1BoxLoss,
    GridAssigner,
    GridDetectionLoss,
)
from dvisionix.registry import LOSSES


def test_base_loss_requires_forward():
    class NoForward(BaseLoss):
        pass

    with pytest.raises(NotImplementedError):
        NoForward()(torch.randn(2, 3), torch.randint(0, 3, (2,)))


def test_cross_entropy_build_and_forward():
    loss = build_loss({"type": "cross_entropy", "weight": 0.5, "label_smoothing": 0.1})
    assert isinstance(loss, CrossEntropy)
    assert loss.weight == 0.5
    out = loss(torch.randn(4, 3), torch.randint(0, 3, (4,)))
    assert out.shape == () and torch.isfinite(out)


def test_focal_loss_2d_and_seg():
    logits = torch.randn(4, 3)
    targets = torch.randint(0, 3, (4,))
    assert torch.isfinite(FocalLoss()(logits, targets))
    seg_logits = torch.randn(2, 3, 8, 8)
    seg_targets = torch.randint(0, 3, (2, 8, 8))
    assert torch.isfinite(FocalLoss(ignore_index=255)(seg_logits, seg_targets))


def test_dice_loss_and_combined():
    seg_logits = torch.randn(2, 3, 8, 8)
    seg_targets = torch.randint(0, 3, (2, 8, 8))
    assert torch.isfinite(DiceLoss()(seg_logits, seg_targets))
    assert torch.isfinite(CombinedSegmentationLoss()(seg_logits, seg_targets))


def test_box_losses_matched_pairs():
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    target = torch.tensor([[1.0, 1.0, 11.0, 11.0], [5.0, 5.0, 15.0, 15.0]])
    assert GIoULoss()(pred, target) > 0
    assert CIoULoss()(pred, target) > 0
    assert L1BoxLoss()(pred, target) > 0
    # 完全重合 -> IoU 类损失趋近 0
    same = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert GIoULoss()(same, same.clone()) < 1e-3


def test_build_losses_composition_and_weights():
    composer = build_losses([
        {"type": "cross_entropy", "weight": 2.0},
        {"type": "focal", "weight": 0.5},
    ])
    assert isinstance(composer, LossComposer)
    logits = torch.randn(4, 3)
    targets = torch.randint(0, 3, (4,))
    out = composer(logits, targets)
    assert "loss" in out and "cross_entropy_loss" in out and "focal_loss" in out
    # 单元素列表退化为单个损失
    single = build_losses([{"type": "dice"}])
    assert not isinstance(single, LossComposer)
    # None -> None
    assert build_losses(None) is None


def test_compute_loss_parses_dict_and_scalar():
    logits = torch.randn(4, 3)
    targets = torch.randint(0, 3, (4,))
    total, extras = compute_loss(CrossEntropy(), logits, targets)
    assert isinstance(total, torch.Tensor) and extras == {}

    class DictLoss(BaseLoss):
        def forward(self, preds, targets, **kw):
            return {"loss": preds.sum(), "extra": preds.mean()}

    total, extras = compute_loss(DictLoss(), logits, targets)
    assert "extra" in extras


def test_grid_assigner_shapes():
    assigner = GridAssigner(num_classes=3)
    boxes_list = [torch.tensor([[5.0, 5.0, 25.0, 25.0]])]
    labels_list = [torch.tensor([0])]
    obj, box, cls, num_pos = assigner((1, 8, 8, 8), boxes_list, labels_list, (64, 64), torch.device("cpu"))
    assert obj.shape == (1, 8, 8) and box.shape == (1, 4, 8, 8) and cls.shape == (1, 8, 8)
    assert num_pos == 1 and obj.sum().item() == 1


def test_grid_detection_loss():
    loss = GridDetectionLoss(num_classes=3)
    preds = torch.randn(2, 8, 8, 8, requires_grad=True)
    batch = {
        "boxes": [torch.tensor([[5.0, 5.0, 25.0, 25.0]]) for _ in range(2)],
        "labels": [torch.tensor([0]) for _ in range(2)],
    }
    out = loss(preds, batch, image_hw=(64, 64))
    assert "loss" in out and "obj_loss" in out and "box_loss" in out and "cls_loss" in out
    out["loss"].backward()
    assert preds.grad is not None


def test_losses_registered():
    for name in ("CrossEntropy", "FocalLoss", "DiceLoss", "GIoULoss", "GridDetectionLoss"):
        assert name in LOSSES