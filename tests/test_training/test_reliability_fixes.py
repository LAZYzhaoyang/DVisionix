# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 训练可靠性与工具链回归：非标量指标、resume 语义、早停恢复、
#       checkpoint 包装前缀、超参搜索的指标读取。
"""训练可靠性与工具链回归测试。

覆盖的问题（均已在 v1.1.0 实测复现）：

- **非标量指标直接崩训练**：``MeanIoU/DiceScore(per_class=True)`` 返回 ``list``，
  进入 epoch 汇总后 ``float(list)`` 抛 ``TypeError``，训练在第一个 epoch 末尾崩溃。
- **``resume: true`` 被当成路径**：``find_checkpoint`` 原样返回布尔值，
  最终 ``torch.load(True)`` 抛 ``AttributeError: 'bool' object has no attribute
  'seek'``；同时 ``resolve_work_dir`` 还会另起一个带新时间戳的目录。
- **续训后 ``restore_best_weights`` 静默失效**：``EarlyStopping.best_weights`` 是
  内存副本、不进 checkpoint，续训后恒为 ``None``，触发早停时既不恢复也不提示。
- **checkpoint 存的是包装后的键**：DDP 的 ``module.`` / ``torch.compile`` 的
  ``_orig_mod.`` 前缀让跨模式续训、``load_backbone`` 前缀匹配与 EMA 的 key 过滤
  全部失配（``load_backbone`` 失配时静默一个权重都不加载）。
- **超参搜索的监控指标列恒为空**：``--work-dir`` 是工作根，``resolve_work_dir``
  会在其下再拼 ``<experiment>/<时间戳>-<哈希>``，而汇总时读的是 ``work/history.csv``。
"""

import os

import pytest
import torch
import torch.nn as nn

from dvisionix.training import ClassificationTask, Trainer
from dvisionix.training.workdir import find_checkpoint, resolve_work_dir

# ---------------------------------------------------------------------------
# 非标量指标
# ---------------------------------------------------------------------------


class _SegDS(torch.utils.data.Dataset):
    def __init__(self, n: int = 8, num_classes: int = 4):
        gen = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, 32, 32, generator=gen)
        self.masks = torch.randint(0, num_classes, (n, 32, 32), generator=gen)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return {"image": self.images[i], "mask": self.masks[i]}


@pytest.mark.unit
def test_per_class_metric_does_not_crash_training(tmp_path):
    """per_class=True 的指标必须能跑完训练（旧实现直接 TypeError）。"""
    from torch.utils.data import DataLoader

    from dvisionix.models import SimpleSegmentationModel
    from dvisionix.training import SegmentationTask

    loader = DataLoader(_SegDS(), batch_size=4)
    task = SegmentationTask(
        num_classes=4,
        metrics=[{"type": "dice_score", "num_classes": 4, "per_class": True}],
    )
    trainer = Trainer(
        task,
        loader,
        loader,
        max_epochs=1,
        device="cpu",
        log_interval=999,
        work_dir=str(tmp_path),
    )
    result = trainer.fit(SimpleSegmentationModel(num_classes=4, in_channels=3))

    last = result["history"][-1]
    # 逐类信息应被展开成标量子键，而不是被丢弃
    assert any(k.startswith("val_dice_") for k in last), f"逐类指标未被展开：{sorted(last)}"
    assert all(not isinstance(v, (list, tuple)) for v in last.values())
    # history.csv 也要能写出（含逗号的 list 会破坏 CSV 结构）
    assert os.path.exists(os.path.join(str(tmp_path), "history.csv"))


@pytest.mark.unit
def test_flatten_metric_values_expands_lists_only():
    """展开只作用于 list/tuple，其余值原样透传。"""
    from dvisionix.training.trainer import _flatten_metric_values

    tensor = torch.tensor(1.5)
    flat = _flatten_metric_values({"a": [1.0, 2.0], "b": tensor, "c": 3})
    assert flat["a_0"] == pytest.approx(1.0) and flat["a_1"] == pytest.approx(2.0)
    assert flat["b"] is tensor
    assert flat["c"] == 3


# ---------------------------------------------------------------------------
# resume 语义
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_resume_true_is_treated_as_auto(tmp_path):
    """`resume: true` 必须等价于 'auto'，而不是被当成 checkpoint 路径。"""
    ckpt_dir = os.path.join(str(tmp_path), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    last = os.path.join(ckpt_dir, "last.pt")
    torch.save({"epoch": 0, "model_state_dict": {}}, last)

    assert find_checkpoint(str(tmp_path), True) == last
    assert find_checkpoint(str(tmp_path), "auto") == last
    assert find_checkpoint(str(tmp_path), None) is None


@pytest.mark.unit
def test_find_checkpoint_rejects_non_string_types(tmp_path):
    """非字符串/布尔的 resume 值必须给出清晰错误，而不是等到 torch.load 崩。"""
    for bad in (123, 1.5, ["auto"]):
        with pytest.raises(ValueError, match="resume"):
            find_checkpoint(str(tmp_path), bad)


class _Cfg(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


@pytest.mark.unit
def test_resolve_work_dir_handles_resume_true(tmp_path):
    """resume=True 时应复用最近一次 run 目录，而不是另起新目录。"""
    experiment = "exp"
    run_dir = os.path.join(str(tmp_path), experiment, "20260101-000000-abcd1234")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    torch.save({"epoch": 0}, os.path.join(run_dir, "checkpoints", "last.pt"))

    cfg = _Cfg({"experiment_name": experiment})
    resolved = resolve_work_dir(cfg, cli_work_dir=str(tmp_path), resume=True)
    assert resolved == run_dir


# ---------------------------------------------------------------------------
# 早停恢复
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_early_stopping_restores_from_disk_when_no_snapshot(tmp_path):
    """本进程没有 best_weights 快照时，应改从 checkpoints/best.pt 恢复。"""
    from torch.utils.data import DataLoader

    from dvisionix.models import SimpleCNN
    from dvisionix.training.callbacks import EarlyStopping

    loader = DataLoader(
        torch.utils.data.TensorDataset(torch.randn(6, 3, 16, 16), torch.randint(0, 3, (6,))),
        batch_size=2,
    )

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 6

        def __getitem__(self, i):
            return {"image": torch.randn(3, 16, 16), "label": torch.tensor(i % 3)}

    loader = DataLoader(_DS(), batch_size=2)
    model = SimpleCNN(num_classes=3, in_channels=3)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 人为把模型参数改掉，模拟"续训后已在别处训练过"
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    ckpt_dir = os.path.join(str(tmp_path), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"model_state_dict": best_state}, os.path.join(ckpt_dir, "best.pt"))

    cb = EarlyStopping(monitor="val_loss", patience=1)
    assert cb.best_weights is None

    trainer = Trainer(
        ClassificationTask(num_classes=3),
        loader,
        loader,
        max_epochs=1,
        device="cpu",
        log_interval=999,
        work_dir=str(tmp_path),
    )
    trainer.model = model
    assert cb._restore_from_disk(trainer) is True

    for key, value in best_state.items():
        assert torch.equal(model.state_dict()[key], value), f"{key} 未被恢复"


# ---------------------------------------------------------------------------
# checkpoint 包装前缀
# ---------------------------------------------------------------------------
class _Wrapper(nn.Module):
    """模拟 DDP 的 ``module`` 包装。"""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.module = inner

    def forward(self, x):
        return self.module(x)


def _trainer(tmp_path):
    from torch.utils.data import DataLoader

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {"image": torch.randn(3, 16, 16), "label": torch.tensor(i % 3)}

    loader = DataLoader(_DS(), batch_size=2)
    return Trainer(
        ClassificationTask(num_classes=3),
        loader,
        loader,
        max_epochs=1,
        device="cpu",
        log_interval=999,
        work_dir=str(tmp_path),
    )


@pytest.mark.unit
def test_save_checkpoint_stores_unwrapped_keys(tmp_path):
    """被包装的模型保存时必须剥掉 module./_orig_mod. 前缀。"""
    from dvisionix.models import SimpleCNN

    trainer = _trainer(tmp_path)
    trainer.model = _Wrapper(SimpleCNN(num_classes=3, in_channels=3))
    path = os.path.join(str(tmp_path), "wrapped.pt")
    trainer.save_checkpoint(path)

    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    assert state, "model_state_dict 为空"
    assert not any(k.startswith("module.") for k in state), "仍带 module. 前缀"


@pytest.mark.unit
def test_load_checkpoint_accepts_prefixed_state_dict(tmp_path):
    """历史 checkpoint 里的包装前缀必须被自动剥离后加载。"""
    from dvisionix.models import SimpleCNN

    trainer = _trainer(tmp_path)
    plain = SimpleCNN(num_classes=3, in_channels=3)
    prefixed = {f"module.{k}": v for k, v in plain.state_dict().items()}

    path = os.path.join(str(tmp_path), "prefixed.pt")
    os.makedirs(str(tmp_path), exist_ok=True)
    trainer.model = plain
    torch.save(
        {
            "epoch": 0,
            "global_step": 0,
            "model_state_dict": prefixed,
            "task_type": "ClassificationTask",
            "model_type": "SimpleCNN",
        },
        path,
    )
    trainer.load_checkpoint(path, plain, allow_config_mismatch=True)
    assert torch.equal(
        plain.state_dict()["features.0.weight"], prefixed["module.features.0.weight"]
    )


@pytest.mark.unit
def test_load_backbone_strips_wrapper_prefixes():
    """load_backbone 必须能处理带包装前缀的 state_dict（否则静默不加载）。"""
    from dvisionix.models import build_model
    from dvisionix.training import load_backbone

    def make():
        return build_model(
            {
                "type": "linear_classifier",
                "num_classes": 3,
                "backbone": {
                    "type": "sequential_backbone",
                    "features_only": False,
                    "stages": [{"type": "conv_norm_act", "in_channels": 3, "out_channels": 8}],
                },
                "head": {"type": "cls_head", "num_classes": 3},
            }
        )

    source = make()
    prefixed = {f"module.{k}": v for k, v in source.state_dict().items()}

    target = make()
    with torch.no_grad():
        for p in target.backbone.parameters():
            p.zero_()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bb.pt")
        torch.save({"model_state_dict": prefixed}, path)
        load_backbone(target, path, prefix="backbone.")

    # 至少要有权重被真正写入（旧实现会因前缀不匹配而一个都没加载）
    assert any(p.abs().sum() > 0 for p in target.backbone.parameters())


# ---------------------------------------------------------------------------
# 超参搜索读取指标
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_hparam_search_reads_metrics_from_nested_run_dir(tmp_path):
    """``--work-dir`` 是工作根：必须能向下找到真正的 history.csv。"""
    from tools.hparam_search import best_from_history

    run_dir = os.path.join(str(tmp_path), "demo_exp", "20260101-000000-abcd1234")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "history.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("epoch,val_accuracy\n0,0.25\n1,0.75\n")

    best = best_from_history(str(tmp_path), "val_accuracy", "max")
    assert best is not None, "未能从嵌套 run 目录读到 history.csv"
    assert best[1] == pytest.approx(0.75)


@pytest.mark.unit
def test_hparam_search_also_accepts_direct_run_dir(tmp_path):
    """直接指向 run 目录（history.csv 就在本层）时同样要能读到。"""
    from tools.hparam_search import best_from_history

    with open(os.path.join(str(tmp_path), "history.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("epoch,val_loss\n0,1.5\n1,0.5\n")
    best = best_from_history(str(tmp_path), "val_loss", "min")
    assert best is not None and best[1] == pytest.approx(0.5)
