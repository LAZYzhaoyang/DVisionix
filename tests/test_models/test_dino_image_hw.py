# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DINO 去噪目标的图像尺寸来源测试（禁止硬编码 stride=4 推断）。
"""DINO 去噪目标的图像尺寸来源测试（CodePlan 7.5 步骤 3-1 / 7.1.2 D4）。

v1.0.0 的 ``DINODetrHead`` 用 ``feats[0].shape[2] * 4`` 当图像尺寸去归一化 GT 框。
只有当骨干第一级特征图恰好是 stride 4 时才成立；官方
``configs/detection/dino_synthetic.yaml`` 用的是 3 个 stride=2 的 stage，
``feats[0]`` 实为 stride 2，代理值成为真值的 **2 倍** ——
而 ``DINOLoss`` 用的是真值，于是同一个 ``bbox_embed`` 在去噪分支与主分支
被训练在两个不同的坐标系里。

本文件把「去噪目标必须按真实图像尺寸归一化」固定为可执行断言。
"""

import pytest
import torch

from dvisionix.data.collate import detection_collate
from dvisionix.models import build_model

# 关掉去噪噪声，让 dn_box_target 恰好等于归一化 GT，便于精确断言
NO_NOISE = 0.0


def _backbone(stride: int):
    return {
        "type": "sequential_backbone",
        "features_only": True,
        "stages": [
            {"type": "conv_norm_act", "in_channels": 3, "out_channels": 8, "stride": stride}
        ],
    }


def _head(**extra):
    head = {
        "type": "dino_head",
        "num_classes": 3,
        "d_model": 32,
        "num_queries": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": 2,
        "num_points": 1,
        "topk": 4,
        "dn_noise_scale_box": NO_NOISE,
    }
    head.update(extra)
    return head


def _model(stride: int, **head_extra):
    return build_model(
        {
            "type": "dinodetr",
            "num_classes": 3,
            "backbone": _backbone(stride),
            "head": _head(**head_extra),
        }
    )


def _batch(h, w, box):
    return detection_collate(
        [
            {
                "image": torch.randn(3, h, w),
                "boxes": torch.tensor([box], dtype=torch.float32),
                "labels": torch.tensor([1]),
            }
        ]
    )


def _expected_norm(box, h, w):
    """xyxy 像素框 -> 归一化 cxcywh。"""
    x1, y1, x2, y2 = box
    return torch.tensor([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])


#: (骨干首级步长, 图像尺寸, 像素框)
#: 第 1 例复现官方配置（stride=2，旧代码差 2 倍）；第 2 例是旧代码恰好正确的对照；
#: 第 3 例是非方形输入。
CASES = [
    pytest.param(
        2, (128, 128), [32.0, 32.0, 96.0, 96.0], id="stride2-square", marks=pytest.mark.unit
    ),
    pytest.param(
        4, (128, 128), [32.0, 32.0, 96.0, 96.0], id="stride4-square", marks=pytest.mark.unit
    ),
    pytest.param(
        2, (128, 256), [64.0, 32.0, 192.0, 96.0], id="stride2-nonsquare", marks=pytest.mark.unit
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("stride,image_hw,box", CASES)
def test_denoising_targets_use_true_image_size(stride, image_hw, box):
    h, w = image_hw
    model = _model(stride)
    model.train()

    batch = _batch(h, w, box)
    preds = model(batch["image"], batch=batch)

    target = preds["dn_box_target"]
    assert bool(preds["dn_positive_mask"][0, 0]), "第 0 个去噪 query 应为正样本"
    assert torch.allclose(target[0, 0], _expected_norm(box, h, w), atol=1e-5), (
        f"去噪目标未按真实图像尺寸 {image_hw} 归一化：" f"{target[0, 0].tolist()}"
    )


@pytest.mark.unit
def test_wrong_proxy_would_differ_for_stride2():
    """确认该断言确实能区分对错：按 4 倍特征图推断会得到不同的归一化结果。"""
    image_hw, box = (128, 128), [32.0, 32.0, 96.0, 96.0]
    feat = 128 // 2  # stride=2 -> feats[0] 空间尺寸
    proxy_hw = (feat * 4, feat * 4)  # v1.0.0 的写法：256x256，实际是 128x128
    assert not torch.allclose(
        _expected_norm(box, *proxy_hw), _expected_norm(box, *image_hw), atol=1e-5
    )


@pytest.mark.unit
def test_missing_image_and_missing_stride_raises():
    """batch 无 image 且未配置 out_stride 时必须报错，而不是静默猜一个尺寸。"""
    model = _model(2)
    model.train()
    images = torch.randn(1, 3, 128, 128)
    batch = {"boxes": [torch.tensor([[1.0, 1.0, 2.0, 2.0]])], "labels": [torch.tensor([1])]}
    with pytest.raises(ValueError, match="真实图像尺寸"):
        model(images, batch=batch)


@pytest.mark.unit
def test_out_stride_fallback_when_batch_has_no_image():
    """显式给出 out_stride 时，允许在 batch 无 image 的情况下按步长推断。"""
    image_hw, box = (128, 128), [32.0, 32.0, 96.0, 96.0]
    model = _model(2, out_stride=2)
    model.train()
    images = torch.randn(1, 3, *image_hw)
    batch = {
        "boxes": [torch.tensor([box], dtype=torch.float32)],
        "labels": [torch.tensor([1])],
    }
    preds = model(images, batch=batch)
    assert torch.allclose(preds["dn_box_target"][0, 0], _expected_norm(box, *image_hw), atol=1e-5)
