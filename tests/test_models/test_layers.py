# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: layers 模块测试：内置层 / builder 工具 / 配置驱动 / timm 封装 / 自定义注册。
"""layers 模块测试：内置层 / builder 工具 / 配置驱动 / timm 封装 / 自定义注册。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch
import torch.nn as nn

from dvisionix.models import (
    MLP,
    ConvNormAct,
    DropPath,
    SEBlock,
    build_activation_layer,
    build_layer,
    build_norm_layer,
)
from dvisionix.registry import LAYERS


class TestBuiltinLayers:
    def test_conv_norm_act_shape(self):
        layer = ConvNormAct(3, 16, kernel_size=3, stride=2)
        out = layer(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 16, 16, 16)

    def test_conv_norm_act_no_norm_uses_bias(self):
        layer = ConvNormAct(3, 8, norm=None, act=None)
        assert layer.conv.bias is not None
        assert isinstance(layer.norm, nn.Identity)
        assert isinstance(layer.act, nn.Identity)

    def test_conv_norm_act_with_norm_disables_bias(self):
        layer = ConvNormAct(3, 8, norm="bn")
        assert layer.conv.bias is None

    def test_mlp_shape(self):
        layer = MLP(32, hidden_features=64, out_features=10)
        out = layer(torch.randn(4, 32))
        assert out.shape == (4, 10)

    def test_se_preserves_shape(self):
        layer = SEBlock(16, reduction=4)
        x = torch.randn(2, 16, 8, 8)
        assert layer(x).shape == x.shape

    def test_drop_path_eval_is_identity(self):
        layer = DropPath(0.5).eval()
        x = torch.randn(4, 8)
        assert torch.allclose(layer(x), x)

    def test_drop_path_zero_prob_is_identity(self):
        layer = DropPath(0.0).train()
        x = torch.randn(4, 8)
        assert torch.allclose(layer(x), x)


class TestBuilders:
    def test_build_norm_bn(self):
        assert isinstance(build_norm_layer("bn", 16), nn.BatchNorm2d)

    def test_build_norm_gn_adjusts_groups(self):
        # 24 通道，请求 32 组 -> 应自动缩减为可整除的组数
        layer = build_norm_layer({"type": "gn", "num_groups": 32}, 24)
        assert isinstance(layer, nn.GroupNorm)
        assert 24 % layer.num_groups == 0

    def test_build_norm_none(self):
        assert isinstance(build_norm_layer(None, 16), nn.Identity)

    def test_build_activation_variants(self):
        assert isinstance(build_activation_layer("relu"), nn.ReLU)
        assert isinstance(build_activation_layer("gelu"), nn.GELU)
        assert isinstance(build_activation_layer(None), nn.Identity)

    def test_build_activation_unknown_raises(self):
        with pytest.raises(KeyError):
            build_activation_layer("not_a_real_act")


class TestConfigDriven:
    def test_build_layer_by_type(self):
        layer = build_layer({"type": "conv_norm_act", "in_channels": 3, "out_channels": 8})
        assert isinstance(layer, ConvNormAct)

    def test_build_layer_alias(self):
        layer = build_layer({"type": "se", "channels": 32})
        assert isinstance(layer, SEBlock)


class TestTimmLayers:
    def test_create_timm_layer(self):
        pytest.importorskip("timm")
        from dvisionix.models import create_timm_layer

        se = create_timm_layer("SqueezeExcite", 16)
        out = se(torch.randn(2, 16, 8, 8))
        assert out.shape == (2, 16, 8, 8)

    def test_build_timm_layer_via_registry(self):
        pytest.importorskip("timm")
        layer = build_layer({"type": "timm_squeeze_excite", "channels": 16})
        out = layer(torch.randn(2, 16, 8, 8))
        assert out.shape == (2, 16, 8, 8)

    def test_list_timm_layers(self):
        pytest.importorskip("timm")
        from dvisionix.models import list_timm_layers

        names = list_timm_layers()
        assert "SqueezeExcite" in names


class TestCustomRegistration:
    def test_register_custom_layer(self):
        @LAYERS.register(name="_test_double_layer")
        class DoubleLayer(nn.Module):
            def forward(self, x):
                return x * 2

        try:
            layer = build_layer({"type": "_test_double_layer"})
            out = layer(torch.ones(3))
            assert torch.allclose(out, torch.full((3,), 2.0))
        finally:
            # 清理注册，避免污染其他测试
            LAYERS._registry.pop("_test_double_layer", None)
