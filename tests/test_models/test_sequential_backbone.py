# -*- coding: utf-8 -*-
"""SequentialBackbone 测试：多尺度/分类输出、通道推导、配置构建、组合模型。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch

from dvisionix.models import SequentialBackbone
from dvisionix.registry import BACKBONES

STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 32, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
]


class TestSequentialBackbone:
    def test_features_only_output(self):
        backbone = SequentialBackbone(STAGES, features_only=True)
        feats = backbone(torch.randn(2, 3, 64, 64))
        assert isinstance(feats, list) and len(feats) == 2
        assert backbone.out_channels == [32, 64]
        assert backbone.num_features == 64
        assert feats[0].shape[1] == 32 and feats[1].shape[1] == 64

    def test_classification_vector_output(self):
        backbone = SequentialBackbone(STAGES, features_only=False)
        vec = backbone(torch.randn(2, 3, 64, 64))
        assert vec.shape == (2, 64)

    def test_nested_stage_list(self):
        stages = [
            {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
            [
                {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
                {"type": "se", "channels": 32},
            ],
        ]
        backbone = SequentialBackbone(stages, features_only=True)
        assert backbone.out_channels == [16, 32]

    def test_out_indices(self):
        stages = STAGES + [
            {"type": "conv_norm_act", "in_channels": 64, "out_channels": 128, "stride": 2},
        ]
        backbone = SequentialBackbone(stages, features_only=True, out_indices=[1, 2])
        feats = backbone(torch.randn(1, 3, 64, 64))
        assert backbone.out_channels == [64, 128]
        assert len(feats) == 2

    def test_registry_build(self):
        backbone = BACKBONES.build(
            {"type": "sequential_backbone", "stages": STAGES, "features_only": True}
        )
        assert backbone.out_channels == [32, 64]

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError):
            SequentialBackbone([])

    def test_classification_composition(self):
        """SequentialBackbone + ClsHead 手工组合（组件即插即用）。"""
        from dvisionix.models.heads import ClsHead

        backbone = SequentialBackbone(STAGES, features_only=False)
        head = ClsHead(in_channels=64, num_classes=10)
        logits = head(backbone(torch.randn(2, 3, 64, 64)))
        assert logits.shape == (2, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
