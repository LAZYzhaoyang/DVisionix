# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 检测回归损失语义测试：GIoU / L1 显式权重、归一化与边界行为。
"""检测回归损失语义测试（CodePlan 7.4 步骤 2-2 / 7.1.2 D6 + D7）。

背景：v1.0.0 的 FCOS / YOLO 回归损失写作

    total_reg += (giou_loss if use_giou else l1_loss) + l1_loss

- ``use_giou=True``（默认）实际是 **GIoU + L1**，``False`` 时是 **2×L1**；
  参数名与实际训练信号不符，且两种模式下都无法单独关闭 L1。
- 各分量按 (图, 层) 求均值后直接累加，不做正样本归一化 →
  loss 量级随 batch size 与特征层数漂移，不同 batch size 的 ``val_loss`` 不可比。

本文件锁定修复后的语义：显式 ``giou_weight`` / ``l1_weight``、
每个分量**只算一次**、回归损失是**跨正样本的真实均值**、空目标不产生 NaN。
"""

import pytest
import torch

from dvisionix.models.losses.detection.losses import FCOSDetectionLoss, YOLOLoss

# 128x128 输入下的 5 级特征图（与默认 strides 对应）
STRIDES = (8, 16, 32, 64, 128)
FEATURE_SHAPES = ((16, 16), (8, 8), (4, 4), (2, 2), (1, 1))
IMAGE_HW = (128, 128)
NUM_CLASSES = 3

# 32x32 的框，落在 stride=8 那一级的尺度区间内
BOX = [[48.0, 48.0, 80.0, 80.0]]


def _fcos_preds(batch: int):
    """构造 FCOS 预测：同一份随机张量复制 batch 份，保证各样本完全一致。"""
    torch.manual_seed(0)

    def _stack(channels):
        one = [torch.randn(1, channels, h, w) for (h, w) in FEATURE_SHAPES]
        return [t.repeat(batch, 1, 1, 1) for t in one]

    return {"cls": _stack(NUM_CLASSES), "reg": _stack(4), "center": _stack(1)}


def _yolo_preds(batch: int):
    """构造 YOLO 预测：同上，各样本完全一致。"""
    torch.manual_seed(0)

    def _stack(channels):
        one = [torch.randn(1, channels, h, w) for (h, w) in FEATURE_SHAPES]
        return [t.repeat(batch, 1, 1, 1) for t in one]

    return {"cls": _stack(NUM_CLASSES), "reg": _stack(4)}


def _batch(batch: int, with_boxes: bool = True):
    if not with_boxes:
        return {
            "boxes": [torch.zeros(0, 4) for _ in range(batch)],
            "labels": [torch.zeros(0, dtype=torch.long) for _ in range(batch)],
        }
    boxes = torch.tensor(BOX)
    labels = torch.tensor([1])
    return {
        "boxes": [boxes.clone() for _ in range(batch)],
        "labels": [labels.clone() for _ in range(batch)],
    }


#: (名称, 损失类, 预测构造函数, 构造损失时的额外参数)
CASES = [
    pytest.param(
        FCOSDetectionLoss, _fcos_preds, {"center_weight": 0.0}, id="fcos", marks=pytest.mark.unit
    ),
    pytest.param(YOLOLoss, _yolo_preds, {}, id="yolo", marks=pytest.mark.unit),
]


def _make(loss_cls, extra, **kwargs):
    params = dict(num_classes=NUM_CLASSES, strides=STRIDES)
    params.update(extra)
    params.update(kwargs)
    return loss_cls(**params)


@pytest.mark.unit
@pytest.mark.parametrize("loss_cls,preds_fn,extra", CASES)
def test_default_is_giou_only(loss_cls, preds_fn, extra):
    """默认只算 GIoU（与 FCOS 论文/torchvision 参考实现一致）。"""
    fn = _make(loss_cls, extra)
    assert (fn.giou_weight, fn.l1_weight) == (1.0, 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("loss_cls,preds_fn,extra", CASES)
def test_three_modes_are_distinguishable_and_counted_once(loss_cls, preds_fn, extra):
    """GIoU / L1 / 组合三种模式可区分，且组合 == 两者之和（证明各只算一次）。"""
    preds = preds_fn(1)
    batch = _batch(1)

    def reg_loss(giou_w, l1_w):
        fn = _make(
            loss_cls, extra, cls_weight=0.0, reg_weight=1.0, giou_weight=giou_w, l1_weight=l1_w
        )
        return float(fn(preds, batch, image_hw=IMAGE_HW)["reg_loss"])

    giou_only = reg_loss(1.0, 0.0)
    l1_only = reg_loss(0.0, 1.0)
    combined = reg_loss(1.0, 1.0)

    assert giou_only > 0.0 and l1_only > 0.0
    # 两种模式确实是不同的信号
    assert giou_only != pytest.approx(l1_only, rel=1e-3)
    # 关键回归：组合模式必须恰好等于两者相加 —— 旧实现会多算一份 L1
    assert combined == pytest.approx(giou_only + l1_only, rel=1e-5)
    assert giou_only < combined


@pytest.mark.unit
@pytest.mark.parametrize("loss_cls,preds_fn,extra", CASES)
def test_weights_scale_linearly(loss_cls, preds_fn, extra):
    """权重应是纯粹的比例因子（旧实现下 L1 权重无法生效）。"""
    preds = preds_fn(1)
    batch = _batch(1)

    def reg_loss(**kwargs):
        fn = _make(loss_cls, extra, cls_weight=0.0, reg_weight=1.0, **kwargs)
        return float(fn(preds, batch, image_hw=IMAGE_HW)["reg_loss"])

    base = reg_loss(giou_weight=1.0, l1_weight=0.0)
    doubled = reg_loss(giou_weight=2.0, l1_weight=0.0)
    assert doubled == pytest.approx(2.0 * base, rel=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("loss_cls,preds_fn,extra", CASES)
def test_use_giou_legacy_switch_migrates_by_intent(loss_cls, preds_fn, extra):
    """旧布尔开关按其字面意图迁移，并发出 DeprecationWarning。"""
    with pytest.warns(DeprecationWarning, match="use_giou"):
        fn = _make(loss_cls, extra, use_giou=True)
    assert (fn.giou_weight, fn.l1_weight) == (1.0, 0.0)

    with pytest.warns(DeprecationWarning, match="use_giou"):
        fn = _make(loss_cls, extra, use_giou=False)
    assert (fn.giou_weight, fn.l1_weight) == (0.0, 1.0)


@pytest.mark.unit
@pytest.mark.parametrize("loss_cls,preds_fn,extra", CASES)
def test_use_giou_conflicts_with_explicit_weights(loss_cls, preds_fn, extra):
    """同时给出旧开关与显式权重必须报错，而不是静默忽略其中一个。"""
    with pytest.raises(ValueError, match="use_giou"):
        _make(loss_cls, extra, use_giou=True, giou_weight=2.0)
    with pytest.raises(ValueError, match="use_giou"):
        _make(loss_cls, extra, use_giou=False, l1_weight=3.0)


@pytest.mark.unit
@pytest.mark.parametrize("loss_cls,preds_fn,extra", CASES)
def test_loss_is_invariant_to_batch_size(loss_cls, preds_fn, extra):
    """D7 的验收：同样内容复制成更大 batch，loss（含 cls/reg 全部项）不应变化。

    修复前：各分量按 (图, 层) 求均值后累加、不做正样本归一化，
    所以 batch 翻倍 loss 也近似翻倍 —— 不同 batch size 的 val_loss 完全不可比。
    """
    fn = _make(loss_cls, extra)
    small = float(fn(preds_fn(1), _batch(1), image_hw=IMAGE_HW)["loss"])
    large = float(fn(preds_fn(2), _batch(2), image_hw=IMAGE_HW)["loss"])
    assert large == pytest.approx(small, rel=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize("loss_cls,preds_fn,extra", CASES)
def test_empty_targets_produce_finite_loss(loss_cls, preds_fn, extra):
    """无正样本的 batch 必须给出有限 loss，且 num_pos 为 0（不得出现 NaN）。"""
    fn = _make(loss_cls, extra)
    out = fn(preds_fn(1), _batch(1, with_boxes=False), image_hw=IMAGE_HW)
    assert torch.isfinite(out["loss"]).all(), out
    assert float(out["num_pos"]) == 0.0
    assert torch.isfinite(out["reg_loss"]).all(), out
