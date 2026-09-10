# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: checkpoint 契约测试：元信息、resume 一致性校验与旧格式兼容。
"""checkpoint 契约测试（CodePlan 7.5 步骤 3-3）。

v1.0.0 的 checkpoint 只存 ``epoch / global_step / *_state_dict / rng_state``，
resume 时**不做任何校验**：拿另一个任务或另一份配置的 checkpoint 续训当前 Trainer，
只要张量形状兼容就会静默加载，得到一个看不出问题的错误训练状态。
"""

import os

import pytest
import torch
from torch.utils.data import DataLoader

from dvisionix.models import SimpleCNN
from dvisionix.training import ClassificationTask, Trainer
from dvisionix.training.trainer import CHECKPOINT_SCHEMA_VERSION, _unwrap_module

NUM_CLASSES = 3


class _DS(torch.utils.data.Dataset):
    def __init__(self, n: int = 4):
        gen = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, 32, 32, generator=gen)
        self.labels = torch.arange(n) % NUM_CLASSES

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"image": self.images[i], "label": self.labels[i]}


def _trainer(config_hash=None):
    loader = DataLoader(_DS(4), batch_size=2)
    return Trainer(
        ClassificationTask(num_classes=NUM_CLASSES),
        loader,
        loader,
        max_epochs=1,
        seed=0,
        log_interval=999,
        config_hash=config_hash,
    )


def _save(tmp_path, config_hash=None):
    trainer = _trainer(config_hash=config_hash)
    trainer.model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3)
    path = os.path.join(str(tmp_path), "ckpt.pt")
    trainer.save_checkpoint(path)
    return path


def _load_raw(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _mutate(path, **changes):
    ckpt = _load_raw(path)
    ckpt.update(changes)
    torch.save(ckpt, path)


@pytest.mark.unit
def test_checkpoint_contains_contract_metadata(tmp_path):
    path = _save(tmp_path, config_hash="abcd1234")
    ckpt = _load_raw(path)
    assert ckpt["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert ckpt["task_type"] == "ClassificationTask"
    assert ckpt["model_type"] == "SimpleCNN"
    assert ckpt["config_hash"] == "abcd1234"


@pytest.mark.unit
def test_resume_rejects_config_hash_mismatch_unless_overridden(tmp_path):
    path = _save(tmp_path, config_hash="aaaa1111")
    model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3)

    with pytest.raises(ValueError, match="配置哈希"):
        _trainer(config_hash="bbbb2222").load_checkpoint(path, model)

    # 显式 override 才放行
    _trainer(config_hash="bbbb2222").load_checkpoint(path, model, allow_config_mismatch=True)


@pytest.mark.unit
def test_resume_accepts_matching_config_hash(tmp_path):
    path = _save(tmp_path, config_hash="samehash")
    model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3)
    _trainer(config_hash="samehash").load_checkpoint(path, model)  # 不应抛错


@pytest.mark.unit
def test_resume_rejects_task_type_mismatch(tmp_path):
    path = _save(tmp_path)
    _mutate(path, task_type="SomeOtherTask")
    with pytest.raises(ValueError, match="任务类型"):
        _trainer().load_checkpoint(path, SimpleCNN(num_classes=NUM_CLASSES, in_channels=3))


@pytest.mark.unit
def test_resume_rejects_model_type_mismatch(tmp_path):
    path = _save(tmp_path)
    _mutate(path, model_type="SomeOtherModel")
    with pytest.raises(ValueError, match="模型类型"):
        _trainer().load_checkpoint(path, SimpleCNN(num_classes=NUM_CLASSES, in_channels=3))


@pytest.mark.unit
def test_resume_rejects_newer_schema_version(tmp_path):
    path = _save(tmp_path)
    _mutate(path, schema_version=CHECKPOINT_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="schema_version"):
        _trainer().load_checkpoint(path, SimpleCNN(num_classes=NUM_CLASSES, in_channels=3))


@pytest.mark.unit
def test_legacy_checkpoint_without_metadata_warns_but_loads(tmp_path):
    """v1.0.0 之前的 checkpoint 没有元信息：给出告警但仍可加载。"""
    path = _save(tmp_path, config_hash="oldhash")
    ckpt = _load_raw(path)
    for key in ("schema_version", "task_type", "model_type", "config_hash"):
        ckpt.pop(key, None)
    torch.save(ckpt, path)

    model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3)
    with pytest.warns(DeprecationWarning, match="schema_version"):
        _trainer(config_hash="newhash").load_checkpoint(path, model)


@pytest.mark.unit
def test_plain_state_dict_is_rejected(tmp_path):
    """纯 state_dict 不是完整 checkpoint，必须明确拒绝而不是解出空状态。"""
    path = os.path.join(str(tmp_path), "plain.pt")
    torch.save(SimpleCNN(num_classes=NUM_CLASSES, in_channels=3).state_dict(), path)
    with pytest.raises(ValueError, match="不是完整 checkpoint"):
        _trainer().load_checkpoint(path, SimpleCNN(num_classes=NUM_CLASSES, in_channels=3))


@pytest.mark.unit
def test_unwrap_module_peels_ddp_and_compile_wrappers():
    inner = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3)

    class _DDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    class _Compiled(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

    assert _unwrap_module(_DDP(inner)) is inner
    assert _unwrap_module(_Compiled(inner)) is inner
    assert _unwrap_module(_Compiled(_DDP(inner))) is inner
    assert _unwrap_module(inner) is inner
    assert _unwrap_module(None) is None


class _WithBackbone(torch.nn.Module):
    """带 ``.backbone`` 子模块的最小模型，用于 load_backbone 契约测试。"""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)


@pytest.mark.unit
class TestLoadTrustPolicy:
    """反序列化 pickle 等于执行任意代码，因此第三方权重默认按不可信处理。"""

    def test_load_backbone_accepts_pure_state_dict_in_safe_mode(self, tmp_path):
        from dvisionix.training import load_backbone

        path = os.path.join(str(tmp_path), "backbone.pt")
        torch.save(_WithBackbone().state_dict(), path)

        model = _WithBackbone()
        out = load_backbone(model, path)
        assert out["missing"] == [] and out["unexpected"] == []

    def test_load_backbone_rejects_untrusted_full_checkpoint_by_default(self, tmp_path):
        """完整 Trainer checkpoint 携带非张量状态，安全模式下必须报错并给出指引。"""
        from dvisionix.training import load_backbone

        # 由本项目的 save_checkpoint 产出：含 rng_state（numpy RNG 状态等）
        path = _save(tmp_path, config_hash="hash")
        with pytest.raises(ValueError, match="trusted=True"):
            load_backbone(_WithBackbone(), path)

        # 显式声明可信后放行（本测试里文件确实由本项目产出）
        out = load_backbone(_WithBackbone(), path, trusted=True)
        assert isinstance(out["missing"], list)

    def test_trainer_safe_mode_rejects_full_state(self, tmp_path):
        path = _save(tmp_path)
        model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3)
        with pytest.raises(ValueError, match="trusted=True"):
            _trainer().load_checkpoint(path, model, trusted=False)

    def test_trainer_default_mode_can_resume_full_state(self, tmp_path):
        path = _save(tmp_path)
        model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3)
        _trainer().load_checkpoint(path, model)  # 默认 trusted=True，不应抛错


@pytest.mark.unit
def test_load_checkpoint_rejects_non_tensor_model_state(tmp_path):
    """结构校验：model_state_dict 必须只含张量。"""
    path = _save(tmp_path)
    ckpt = _load_raw(path)
    ckpt["model_state_dict"]["evil"] = "not-a-tensor"
    torch.save(ckpt, path)

    with pytest.raises(ValueError, match="非张量"):
        _trainer().load_checkpoint(path, SimpleCNN(num_classes=NUM_CLASSES, in_channels=3))
