# -*- coding: utf-8 -*-
"""v0.7.1 阶段 A 测试：decode 统一契约 / input_style 自声明 / RT-DETR neck / 注册名别名。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.models import build_model
from dvisionix.registry import HEADS

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]
BACKBONE = {"type": "sequential_backbone", "stages": STAGES, "features_only": True}
NECK = {"type": "fpn", "out_channels": 64}


def test_input_style_attributes():
    from dvisionix.models.heads.classification.linear import ClsHead
    from dvisionix.models.heads.detection.fcos import FCOSHead
    from dvisionix.models.heads.detection.rtdetr import RTDETRHead
    from dvisionix.models.heads.segmentation.maskformer import MaskFormerHead
    from dvisionix.models.heads.segmentation.segformer import SegFormerHead
    from dvisionix.models.heads.segmentation.unet import UNetDecoder

    for cls in (UNetDecoder, SegFormerHead, MaskFormerHead, RTDETRHead):
        assert cls.input_style == "multi_scale", cls.__name__
    for cls in (FCOSHead, ClsHead):
        assert getattr(cls, "input_style", "single_scale") == "single_scale", cls.__name__


def test_multiscale_injection_with_neck():
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 4,
            "backbone": BACKBONE,
            "neck": NECK,
            "head": {"type": "unet_decoder", "num_classes": 4},
        }
    )
    assert model.head.in_channels_list == [64, 64, 64]
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 4, 64, 64)


def test_maskformer_decode_via_model():
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 3,
            "backbone": BACKBONE,
            "head": {
                "type": "maskformer_head",
                "num_classes": 3,
                "d_model": 32,
                "num_queries": 8,
                "output_mode": "full",
            },
        }
    )
    preds = model(torch.randn(1, 3, 64, 64))
    masks, scores, labels = model.decode(preds, (64, 64), score_threshold=0.0)
    assert masks[0].ndim == 3 and masks[0].dtype == torch.bool
    assert masks[0].shape[0] == scores[0].shape[0] == labels[0].shape[0]


def test_decode_not_available_on_single_scale_head():
    model = build_model(
        {
            "type": "segmentation_model",
            "num_classes": 4,
            "backbone": BACKBONE,
            "head": {"type": "deeplabv3_head", "num_classes": 4},
        }
    )
    with pytest.raises(NotImplementedError):
        model.decode({}, (64, 64))


def test_rtdetr_with_neck():
    model = build_model(
        {
            "type": "rtdetr",
            "num_classes": 3,
            "backbone": BACKBONE,
            "neck": NECK,
            "head": {
                "type": "rtdetr_head",
                "num_classes": 3,
                "d_model": 32,
                "topk": 8,
                "num_decoder_layers": 2,
                "num_heads": 4,
            },
        }
    )
    assert model.head.in_channels_list == [64, 64, 64]
    preds = model(torch.randn(1, 3, 128, 128))
    assert set(preds.keys()) == {"logits", "boxes"}
    boxes, scores, labels = model.decode(preds, (128, 128), score_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 4


def test_classification_head_aliases():
    for name in (
        "arcface_head",
        "cosface_head",
        "sphereface_head",
        "adaface_head",
        "multi_label_head",
    ):
        assert name in HEADS, name
    model = build_model(
        {
            "type": "linear_classifier",
            "backbone": {"type": "sequential_backbone", "stages": STAGES, "features_only": False},
            "head": {"type": "arcface_head", "num_classes": 5},
            "num_classes": 5,
        }
    )
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape[-1] == 5
