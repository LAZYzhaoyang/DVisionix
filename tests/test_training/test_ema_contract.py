# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: EMA 回调契约测试：原地更新、只处理浮点张量、resume 保留影子。
"""EMA 回调契约测试（CodePlan 7.7 步骤 5-1）。

v1.0.0 的实现有三个问题：

1. ``shadow[k] = decay * shadow[k] + (1 - decay) * v.float()`` 每步为每个参数
   新建张量（25M 参数模型每步多分配 ~100MB），实测开销占训练总耗时 17%；
2. 遍历 ``state_dict()`` 的**全部**条目，包括整型 buffer
   （``num_batches_tracked`` 这类计数器），对它们做滑动平均没有意义；
3. ``on_train_begin`` 无条件用当前模型权重重建影子 —— 续训时会把
   ``load_state_dict`` 刚恢复的 EMA 状态清零。
"""

import pytest
import torch
import torch.nn as nn

from dvisionix.training.callbacks import EMA


class _ModelWithIntBuffer(nn.Module):
    """同时含浮点参数、浮点 buffer 与整型 buffer 的模型。"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.register_buffer("running_mean", torch.zeros(4))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        return self.fc(x)


class _FakeTrainer:
    def __init__(self, model):
        self.model = model
        self.current_epoch = 0
        self.work_dir = None


@pytest.mark.unit
def test_shadow_covers_only_floating_tensors():
    model = _ModelWithIntBuffer()
    ema = EMA(decay=0.9)
    ema.on_train_begin(_FakeTrainer(model))

    assert "fc.weight" in ema.shadow
    assert "running_mean" in ema.shadow
    assert "num_batches_tracked" not in ema.shadow, "整型计数器不应参与滑动平均"
    assert ema.shadow["fc.weight"].dtype == torch.float32


@pytest.mark.unit
def test_inplace_update_matches_reference_formula():
    """原地更新必须与参考公式逐元素一致。"""
    model = _ModelWithIntBuffer()
    ema = EMA(decay=0.9)
    trainer = _FakeTrainer(model)
    ema.on_train_begin(trainer)

    reference = {k: v.clone() for k, v in ema.shadow.items()}
    for step in range(5):
        with torch.no_grad():
            model.fc.weight.add_(0.1 * (step + 1))
            model.running_mean.add_(0.2)
        ema.on_batch_end(trainer, step, {}, "train")
        for key in reference:
            source = trainer.model.state_dict()[key].float()
            reference[key] = 0.9 * reference[key] + 0.1 * source

    for key, expected in reference.items():
        assert torch.allclose(ema.shadow[key], expected, atol=1e-6), key


@pytest.mark.unit
def test_shadow_tensors_are_updated_in_place():
    """影子张量对象在更新前后必须是同一个（否则说明仍在每步分配新张量）。"""
    model = _ModelWithIntBuffer()
    ema = EMA(decay=0.9)
    trainer = _FakeTrainer(model)
    ema.on_train_begin(trainer)

    before = {k: id(v) for k, v in ema.shadow.items()}
    with torch.no_grad():
        model.fc.weight.add_(1.0)
    ema.on_batch_end(trainer, 0, {}, "train")
    after = {k: id(v) for k, v in ema.shadow.items()}
    assert before == after, "影子张量被替换了，说明不是原地更新"


@pytest.mark.unit
def test_validation_mode_does_not_update_shadow():
    model = _ModelWithIntBuffer()
    ema = EMA(decay=0.9)
    trainer = _FakeTrainer(model)
    ema.on_train_begin(trainer)
    snapshot = {k: v.clone() for k, v in ema.shadow.items()}

    with torch.no_grad():
        model.fc.weight.add_(5.0)
    ema.on_batch_end(trainer, 0, {}, "val")
    for key, value in snapshot.items():
        assert torch.equal(ema.shadow[key], value), key


@pytest.mark.unit
def test_resume_preserves_shadow_instead_of_resetting():
    """续训时 on_train_begin 不得用当前权重覆盖已恢复的影子。"""
    model = _ModelWithIntBuffer()
    ema = EMA(decay=0.999)
    trainer = _FakeTrainer(model)
    ema.on_train_begin(trainer)

    # 模拟训练一段时间后保存
    with torch.no_grad():
        model.fc.weight.add_(3.0)
    for step in range(3):
        ema.on_batch_end(trainer, step, {}, "train")
    saved = {k: v.clone() for k, v in ema.shadow.items()}
    assert not torch.allclose(saved["fc.weight"], model.fc.weight.float())

    # 新回调从 checkpoint 恢复影子，然后开始新一轮训练
    resumed = EMA(decay=0.999)
    resumed.load_state_dict({**ema.state_dict()})
    resumed.on_train_begin(trainer)

    for key, value in saved.items():
        assert torch.equal(
            resumed.shadow[key], value
        ), f"{key} 的影子在 on_train_begin 后被重置，续训会丢失 EMA 状态"
    # 恢复后引用对必须补齐，更新才生效
    assert len(resumed._pairs) == len(saved)
    resumed.on_batch_end(trainer, 0, {}, "train")
    assert not torch.equal(resumed.shadow["fc.weight"], saved["fc.weight"])


@pytest.mark.unit
def test_state_dict_roundtrip_includes_keys():
    model = _ModelWithIntBuffer()
    ema = EMA(decay=0.9)
    ema.on_train_begin(_FakeTrainer(model))
    state = ema.state_dict()
    assert state["keys"] == ema._keys

    restored = EMA()
    restored.load_state_dict(state)
    assert restored._keys == ema._keys
    assert set(restored.shadow) == set(ema.shadow)


@pytest.mark.unit
def test_legacy_state_dict_without_keys_is_supported():
    """v1.0.0 保存的 EMA 状态没有 keys 字段，必须仍能恢复。"""
    model = _ModelWithIntBuffer()
    ema = EMA(decay=0.9)
    ema.on_train_begin(_FakeTrainer(model))
    legacy = {
        "decay": ema.decay,
        "decay_warmup_epochs": ema.decay_warmup_epochs,
        "shadow": ema.shadow,
    }

    restored = EMA()
    restored.load_state_dict(legacy)
    assert set(restored._keys) == set(ema.shadow)
