# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: PANet 颈部测试（S5）。
"""PANet 颈部测试（S5）。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.models import build_neck


def test_panet_build_and_forward():
    panet = build_neck({"type": "panet", "in_channels": [16, 32, 64], "out_channels": 32})
    feats = [torch.randn(1, c, 64 // (2**i), 64 // (2**i)) for i, c in enumerate([16, 32, 64])]
    out = panet(feats)
    assert len(out) == 3
    assert all(o.shape[1] == 32 for o in out)
    # 自底向上路径输出尺寸与 FPN 一致
    assert [o.shape[-2:] for o in out] == [(64, 64), (32, 32), (16, 16)]


def test_panet_extra_outs():
    panet = build_neck(
        {"type": "panet", "in_channels": [16, 32, 64], "out_channels": 16, "num_outs": 4}
    )
    feats = [torch.randn(1, c, 64 // (2**i), 64 // (2**i)) for i, c in enumerate([16, 32, 64])]
    out = panet(feats)
    assert len(out) == 4
