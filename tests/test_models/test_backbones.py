# -*- coding: utf-8 -*-
"""timm 骨干网络单元测试（使用随机初始化，避免下载权重）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from dvisionix.models import TimmBackbone, TimmClassifier


def test_resnet_backbone():
    backbone = TimmBackbone("resnet18", pretrained=False)
    feats = backbone(torch.randn(2, 3, 224, 224))
    assert feats.shape == (2, backbone.num_features)


def test_vit_classifier():
    model = TimmClassifier("vit_tiny_patch16_224", num_classes=10, pretrained=False)
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 10)


def test_swin_classifier():
    model = TimmClassifier("swin_tiny_patch4_window7_224", num_classes=5, pretrained=False)
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 5)


def test_freeze_backbone():
    model = TimmClassifier("resnet18", num_classes=10, pretrained=False)
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())


if __name__ == "__main__":
    print("Running backbone tests...")
    test_resnet_backbone(); print("ok test_resnet_backbone")
    test_vit_classifier(); print("ok test_vit_classifier")
    test_swin_classifier(); print("ok test_swin_classifier")
    test_freeze_backbone(); print("ok test_freeze_backbone")
    print("All backbone tests passed!")
