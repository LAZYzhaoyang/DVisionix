# -*- coding: utf-8 -*-
"""v0.17-P2 测试：超参搜索工具 + 特征蒸馏。"""

import pytest

torch = pytest.importorskip("torch")

from dvisionix.models.losses import FeatureDistillLoss
from dvisionix.registry import LOSSES


def test_feature_distill_loss():
    assert "feature_distill" in LOSSES
    loss_fn = FeatureDistillLoss()
    s = torch.randn(4, 8)
    t = torch.randn(4, 8)
    out = loss_fn(s, t)
    assert out.dim() == 0
    # 相同特征损失为 0
    assert loss_fn(s, s.clone()) < 1e-6
    # 多层特征列表
    out_list = loss_fn(
        [torch.randn(4, 8), torch.randn(4, 16)], [torch.randn(4, 8), torch.randn(4, 16)]
    )
    assert out_list.dim() == 0


def test_distill_callback_feature_extractor():
    from dvisionix.training import DistillCallback

    teacher = torch.nn.Linear(4, 4)

    def fe(m, x):
        return [m(x)]

    cb = DistillCallback(teacher=teacher, feature_extractor=fe)
    assert cb.feature_extractor is fe


def test_hparam_search_helpers():
    from tools import hparam_search

    spec = {"training.optimizer.lr": [1e-4, 1e-3], "model.backbone.depths": [[1, 1], [2, 2]]}
    keys, combos = hparam_search.combinations(spec)
    assert len(combos) == 4
    assert hparam_search.format_value([1, 2]) == "[1,2]"
    assert hparam_search.format_value(0.001) == "0.001"
    # --cfg-options 需要每个 k=v 独立成参（nargs="*"），列表形式可与 train.py 直接对接
    opts = hparam_search.to_cfg_options(keys, combos[0])
    assert isinstance(opts, list) and opts == [
        "training.optimizer.lr=0.0001",
        "model.backbone.depths=[1,1]",
    ]
    # 随机采样
    import numpy as np

    keys2, combos2 = hparam_search.random_trials(spec, 3, np.random.default_rng(0))
    assert len(combos2) == 3 and len(keys2) == 2
