# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 数值/精度回归：AMP(fp16) 下的失稳与若干静默错算。
"""数值与精度回归测试。

背景（为什么需要这个文件）：

CI 装的是 **CPU-only torch**（``.github/workflows/ci.yml`` 的
``--index-url .../whl/cpu``），而 ``Trainer`` 只在 ``device.type == "cuda"`` 时创建
``GradScaler``，``torch.autocast`` 因此恒为关闭。于是所有"只在 fp16 / autocast 下
才出问题"的路径在 CI 里从未被执行，实测确认的问题包括：

- GIoU：fp16 面积和溢出 65504 → ``union=inf`` → **loss = -inf**（边长 ≈181px 即触发）；
- CenterNet 热图 focal：fp16 下 ``1-1e-4`` 舍入成 1.0，clamp 上界失效 → ``log(0)`` → **nan**；
- MaskFormer/Mask2Former：``F.binary_cross_entropy`` 对 autocast 不安全 → **直接抛错**；
- 度量学习头：fp16 下 ``1.0-1e-7`` 舍入成 1.0 → ``acos`` 反传 → **梯度全 NaN**。

另有若干与精度无关、但在默认配置或默认数据分布下才暴露的静默错算：
``DiceLoss`` 遇 ``ignore_index=255``、``LinearClassifier`` 丢弃 kwargs 使 margin 失效、
``RetinaNetLoss`` 未按 batch 归一化、``SegFormerV3`` 的通道注意力被平方、
``ConvNormAct(norm="ln")`` 归一化错误的轴、度量头用 legacy ``torch.FloatTensor``
忽略全局默认 dtype、``UPerNetHead.fpn_convs`` 是死参数。

只在真实 CUDA 上才有意义的用例用 ``pytest.mark.cuda`` 标记；其余（精度无关的
静默错算）在无 GPU 的 CI 上同样会运行。
"""

import pytest
import torch
import torch.nn as nn

from dvisionix.models.heads.classification.arcface import ArcFaceHead
from dvisionix.models.heads.segmentation.segformer_v3 import SegFormerV3Head
from dvisionix.models.heads.segmentation.upernet import UPerNetHead
from dvisionix.models.layers.basic import ConvNormAct
from dvisionix.models.layers.norm import LayerNorm2d
from dvisionix.models.losses.detection.box_loss import GIoULoss
from dvisionix.models.losses.detection.losses import (
    CenterNetLoss,
    FCOSDetectionLoss,
    RetinaNetLoss,
)
from dvisionix.models.losses.segmentation import (
    CombinedSegmentationLoss,
    DiceLoss,
    MaskFormerLoss,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 设备")


# ---------------------------------------------------------------------------
# GIoU：fp16 面积溢出
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_giou_loss_finite_for_large_boxes_in_fp16():
    """fp16 下大框的 GIoU 损失必须是有限值（旧实现给出 -inf）。"""
    loss = GIoULoss()
    side = 200.0  # 面积 4e4，两框相加 8e4 > fp16 上限 65504
    pred = torch.tensor([[0.0, 0.0, side, side]], dtype=torch.float16)
    target = torch.tensor([[0.0, 0.0, side, side]], dtype=torch.float16)
    value = loss(pred, target)
    assert torch.isfinite(value), f"fp16 下 GIoU 出现非有限值：{value}"
    assert float(value) == pytest.approx(0.0, abs=1e-3), "完全重合的两框 GIoU 损失应为 0"


@pytest.mark.unit
def test_giou_loss_fp16_matches_fp32_reference():
    """在同样的（已舍入的）输入上，fp16 内部计算不应比 fp32 漂移。"""
    loss = GIoULoss()
    boxes_p = torch.tensor([[0.0, 0.0, 300.0, 300.0], [10.0, 10.0, 120.0, 120.0]])
    boxes_t = torch.tensor([[5.0, 5.0, 280.0, 280.0], [12.0, 9.0, 118.0, 130.0]])
    # 用同一组 fp16 舍入后的数值做 fp32 参考，隔离"内部精度"这一唯一变量
    p16, t16 = boxes_p.half(), boxes_t.half()
    reference = float(loss(p16.float(), t16.float()))
    value = float(loss(p16, t16))
    assert abs(reference - value) < 1e-3, f"fp16 内部计算漂移过大：{value} vs 参考 {reference}"


# ---------------------------------------------------------------------------
# CenterNet：热图 focal 在 fp16 下 log(0)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_centernet_heatmap_loss_finite_in_fp16():
    """fp16 热图 logit 一旦饱和，focal 损失仍必须是有限值。"""
    loss = CenterNetLoss(num_classes=3)
    batch = {
        "boxes": [torch.tensor([[2.0, 2.0, 10.0, 10.0]])],
        "labels": [torch.tensor([0])],
    }
    for dtype in (torch.float32, torch.float16):
        for logit in (0.0, 8.5, 20.0):
            hm = torch.full((1, 3, 16, 16), logit, dtype=dtype)
            wh = torch.zeros((1, 2, 16, 16), dtype=dtype)
            off = torch.zeros((1, 2, 16, 16), dtype=dtype)
            out = loss({"heatmap": hm, "wh": wh, "offset": off}, batch)
            value = out["hm_loss"]
            assert torch.isfinite(value), f"{dtype} logit={logit} 的热图损失非有限：{value}"


# ---------------------------------------------------------------------------
# MaskFormer：autocast 不安全算子
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.cuda
@pytest.mark.unit
def test_maskformer_loss_runs_under_autocast():
    """MaskFormer/Mask2Former 损失必须能在 autocast(fp16) 下计算。"""
    loss = MaskFormerLoss(num_classes=3).cuda()
    preds = {
        "pred_logits": torch.randn(1, 4, 4, device="cuda"),
        "pred_masks": torch.randn(1, 4, 16, 16, device="cuda"),
    }
    batch = {"mask": [torch.randint(0, 3, (16, 16), device="cuda")]}
    with torch.autocast("cuda", dtype=torch.float16):
        out = loss(preds, batch)
    assert torch.isfinite(out["loss"]), f"autocast 下 loss 非有限：{out['loss']}"


# ---------------------------------------------------------------------------
# 分割损失：ignore_index=255
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_dice_loss_accepts_ignore_index_255():
    """掩膜含 void(255) 时 Dice 不应崩溃，且忽略区域不参与计算。"""
    loss = DiceLoss()
    logits = torch.randn(2, 3, 8, 8)
    targets = torch.randint(0, 3, (2, 8, 8))
    clean = targets.clone()
    targets[0, 0, 0] = 255

    value = loss(logits, targets)
    assert torch.isfinite(value)
    # 只有一个像素被忽略，改动应极小；不崩溃即是主要断言
    assert float(value) == pytest.approx(float(loss(logits, clean)), abs=0.05)


@pytest.mark.unit
def test_combined_segmentation_loss_accepts_ignore_index_255():
    """组合损失（CE + Dice）同样必须容忍 void 标签。"""
    loss = CombinedSegmentationLoss()
    logits = torch.randn(2, 3, 8, 8)
    targets = torch.randint(0, 3, (2, 8, 8))
    targets[1, 3, 3] = 255
    assert torch.isfinite(loss(logits, targets))


# ---------------------------------------------------------------------------
# 度量学习头：fp16 下 acos 反传
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.cuda
@pytest.mark.unit
def test_arcface_fp16_gradient_is_finite_when_cosine_saturates():
    """目标类余弦饱和到 1.0 时，fp16 反传不应产生 NaN 梯度。"""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16):
        head = ArcFaceHead(in_channels=32, num_classes=8, s=30.0, m=0.5).cuda().to(dtype)
        head.train()
        with torch.no_grad():
            weight = torch.nn.functional.normalize(head.weight.detach().float(), dim=1)
            labels = torch.arange(8, device="cuda")
            # 特征与目标类权重完全同向 → cos_theta == 1，正是 margin 收敛后的必然状态
            x = weight[labels].to(dtype).clone().requires_grad_(True)
        head(x, labels).sum().backward()
        assert not torch.isnan(x.grad).any(), f"{dtype} 下出现 NaN 梯度"
        assert torch.isfinite(x.grad).all(), f"{dtype} 下梯度非有限"


# ---------------------------------------------------------------------------
# margin 头：kwargs 必须透传
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_linear_classifier_forwards_labels_to_head():
    """LinearClassifier 必须把 kwargs（尤其是 labels）透传给 head，否则 margin 永不生效。"""
    from dvisionix.models import build_model

    model = build_model(
        {
            "type": "linear_classifier",
            "num_classes": 4,
            "backbone": {
                "type": "sequential_backbone",
                "features_only": False,
                "stages": [
                    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 8, "stride": 2}
                ],
            },
            "head": {"type": "arcface_head", "num_classes": 4, "s": 30.0, "m": 0.5},
        }
    )
    model.train()
    x = torch.randn(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 2, 3])
    with_labels = model(x, labels=labels)
    without = model(x)
    assert not torch.allclose(
        with_labels, without
    ), "model(x, labels=...) 与 model(x) 完全相同 → kwargs 被丢弃，margin 没有生效"


@pytest.mark.unit
def test_classification_task_passes_labels_to_model():
    """ClassificationTask.training_step 必须把标签传给模型（margin 头依赖它）。"""
    from dvisionix.training import ClassificationTask

    class _Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.zeros(()))
            self.seen_labels = None

        def forward(self, x, **kwargs):
            self.seen_labels = kwargs.get("labels")
            return x.flatten(1)[:, :3] + self.scale

    model = _Spy()
    task = ClassificationTask(num_classes=3)
    batch = {"image": torch.randn(4, 3, 8, 8), "label": torch.tensor([0, 1, 2, 0])}
    task.training_step(model, batch, torch.device("cpu"))
    assert model.seen_labels is not None, "training_step 没有把 labels 传给模型"
    assert torch.equal(model.seen_labels, batch["label"])


# ---------------------------------------------------------------------------
# 检测损失：batch 归一化
# ---------------------------------------------------------------------------
def _detector_preds(
    batch_size: int,
    cls_channels: int,
    reg_channels: int,
    with_center: bool = False,
    seed: int = 0,
):
    gen = torch.Generator().manual_seed(seed)
    one_cls = [
        torch.randn(1, cls_channels, 8, 8, generator=gen),
        torch.randn(1, cls_channels, 4, 4, generator=gen),
    ]
    one_reg = [
        torch.randn(1, reg_channels, 8, 8, generator=gen),
        torch.randn(1, reg_channels, 4, 4, generator=gen),
    ]
    preds = {
        "cls": [t.repeat(batch_size, 1, 1, 1) for t in one_cls],
        "reg": [t.repeat(batch_size, 1, 1, 1) for t in one_reg],
    }
    if with_center:
        preds["center"] = [
            torch.randn(1, 1, 8, 8, generator=gen).repeat(batch_size, 1, 1, 1),
            torch.randn(1, 1, 4, 4, generator=gen).repeat(batch_size, 1, 1, 1),
        ]
    batch = {
        "boxes": [torch.tensor([[2.0, 2.0, 20.0, 20.0]])] * batch_size,
        "labels": [torch.tensor([0])] * batch_size,
    }
    return preds, batch


@pytest.mark.unit
def test_retinanet_loss_is_invariant_to_batch_size():
    """同一份数据重复成 bs=1/2/4 时 loss 必须基本不变（旧实现严格成正比）。"""
    values = []
    for bs in (1, 2, 4):
        loss = RetinaNetLoss(num_classes=3, strides=(8, 16), base_sizes=(16, 32))
        preds, batch = _detector_preds(bs, cls_channels=27, reg_channels=36)  # A = 3*3 = 9
        values.append(float(loss(preds, batch, image_hw=(64, 64))["loss"]))
    assert values[1] == pytest.approx(values[0], rel=1e-4), f"bs 翻倍后 loss 漂移：{values}"
    assert values[2] == pytest.approx(values[0], rel=1e-4), f"bs 再翻倍后 loss 漂移：{values}"


@pytest.mark.unit
def test_fcos_loss_is_invariant_to_batch_size():
    """对照组：FCOS 已在 D7 修正过归一化，这里作为口径基准。"""
    values = []
    for bs in (1, 2, 4):
        loss = FCOSDetectionLoss(num_classes=3, strides=(8, 16))
        preds, batch = _detector_preds(bs, cls_channels=3, reg_channels=4, with_center=True)
        values.append(float(loss(preds, batch, image_hw=(64, 64))["loss"]))
    assert values[1] == pytest.approx(values[0], rel=1e-4)


# ---------------------------------------------------------------------------
# SegFormerV3：通道注意力只作用一次
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_segformer_v3_applies_se_exactly_once():
    """decode 收到的必须是 se 的输出本身，而不是 se 的输出再乘一遍输入（fused²）。"""
    head = SegFormerV3Head(in_channels_list=[8, 16], num_classes=4, d_model=16, num_blocks=1)
    captured = {}

    original_se = head.se

    class _Spy(nn.Module):
        def forward(self, x):
            out = original_se(x)
            captured["se_in"] = x.detach().clone()
            captured["se_out"] = out.detach().clone()
            return out

    head.se = _Spy()
    head.eval()  # decode 内含 dropout，eval 才能复算比对
    original_decode = head.decode
    feats = [torch.randn(1, 8, 8, 8), torch.randn(1, 16, 4, 4)]
    with torch.no_grad():
        out = head(feats)
        expected = original_decode(captured["se_out"])

    assert torch.allclose(out, expected), (
        "head 输出不等于 decode(se(fused)) → 通道注意力被重复施加"
        "（旧写法 fused * se(fused) = fused²·scale）"
    )


# ---------------------------------------------------------------------------
# LayerNorm 轴 / 默认 dtype / 死参数
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_conv_norm_act_layernorm_is_channels_first():
    """norm="ln" 在 (B, C, H, W) 上必须按通道归一化，而不是按最后一维 W。"""
    for size in (8, 16):
        layer = ConvNormAct(8, 8, kernel_size=3, padding=1, norm="ln")
        out = layer(torch.randn(1, 8, size, size))
        assert out.shape == (1, 8, size, size)
        inner = layer.norm
        assert isinstance(
            inner, LayerNorm2d
        ), f"norm='ln' 构建出的是 {type(inner).__name__}，在 channels-first 上会归一化错误的轴"


@pytest.mark.unit
def test_metric_learning_head_respects_default_dtype():
    """度量头的权重必须遵循全局默认 dtype（legacy torch.FloatTensor 会无视它）。"""
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        head = ArcFaceHead(in_channels=4, num_classes=3)
        assert (
            head.weight.dtype == torch.float64
        ), f"默认 dtype 为 float64 时权重却是 {head.weight.dtype}"
        # 不抛 dtype 冲突才算真正遵循
        logits = head(torch.randn(2, 4).double(), torch.tensor([0, 1]))
        assert logits.dtype == torch.float64
    finally:
        torch.set_default_dtype(original)


@pytest.mark.unit
def test_upernet_fpn_convs_receive_gradients():
    """fpn_convs 不能是死参数：任何一层拿不到梯度，DDP 下第二轮就会中断训练。"""
    head = UPerNetHead(in_channels_list=[8, 16, 32], num_classes=4, channels=16)
    head.train()
    # batch=2：PPM 的 AdaptiveAvgPool2d(1) 会得到 1x1，train 模式下 BN 需要 >1 个值/通道
    feats = [torch.randn(2, 8, 16, 16), torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)]
    head(feats).sum().backward()
    for idx, conv in enumerate(head.fpn_convs):
        assert conv.weight.grad is not None, f"fpn_convs[{idx}] 没有梯度（未被使用）"
        assert conv.weight.grad.abs().sum() > 0
