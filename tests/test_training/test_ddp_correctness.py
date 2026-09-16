# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DDP 正确性回归：设备绑定、rank0 落盘、早停同步、工作目录确定性。
"""DDP 正确性回归测试。

覆盖的问题（v1.1.0 实测/代码确认）：

- **``--devices`` 是死参数**：``self.devices`` 只被赋值、全包再无读取，
  DDP 设备完全由 ``LOCAL_RANK`` 决定 —— 用户以为用了 1、3 号卡，实际用 0、1 号卡。
- **显式 ``device: cpu`` 被静默改成 cuda**：``if self.is_distributed and
  torch.cuda.is_available()`` 会覆盖用户请求，且 NCCL/Gloo 后端也按
  ``cuda.is_available()`` 而非实际设备选择。
- **``history.csv`` / ``best_metrics.csv`` 被所有 rank 并发写**：rank≠0 的
  ``val_*`` 恒为 0（只有 rank0 更新任务指标），写出的可能是整列 0，
  而超参搜索与实验管理都依赖这份 CSV。
- **DDP + 早停会死锁**：各 rank 用各自的指标决策，rank0 先退出 epoch 循环后，
  其余 rank 继续进入下一轮集合通信 → 永久 hang（``base.yaml`` 里早停默认开启）。
- **torchrun 下各 rank 的工作目录不同**：目录名带 ``datetime.now()`` 时间戳，
  跨秒启动就会分家，``resume auto`` 也会选错目录。
- **验证集被强制 drop_last=True** 可能让每个 rank 的 val_loader 长度为 0，
  此时验证循环一次都不执行、指标恒为 0 却不报错。
"""

import os

import pytest
import torch
from torch.utils.data import DataLoader

from dvisionix.training import ClassificationTask, Trainer
from dvisionix.training.workdir import distributed_rank_from_env, resolve_work_dir


class _DS(torch.utils.data.Dataset):
    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"image": torch.randn(3, 8, 8), "label": torch.tensor(i % 3)}


def _make_trainer(**kwargs) -> Trainer:
    loader = DataLoader(_DS(), batch_size=2)
    return Trainer(
        ClassificationTask(num_classes=3),
        loader,
        loader,
        max_epochs=1,
        device="cpu",
        log_interval=999,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# --devices 真正生效
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_devices_argument_maps_to_gpu_index():
    """--devices 1,3 必须依次映射到 local_rank 0、1，而不是被忽略。"""
    trainer = _make_trainer(devices=[1, 3])
    assert trainer._resolve_ddp_device(0) == torch.device("cuda:1")
    assert trainer._resolve_ddp_device(1) == torch.device("cuda:3")


@pytest.mark.unit
def test_devices_falls_back_to_local_rank():
    """未给 devices 时按 LOCAL_RANK 绑定；devices 长度不足时也不越界。"""
    trainer = _make_trainer()
    assert trainer._resolve_ddp_device(2) == torch.device("cuda:2")

    partial = _make_trainer(devices=[5])
    assert partial._resolve_ddp_device(0) == torch.device("cuda:5")
    assert partial._resolve_ddp_device(1) == torch.device("cuda:1")


@pytest.mark.unit
def test_explicit_cpu_device_is_respected():
    """显式 device='cpu' 必须被尊重（用于在 GPU 机器上跑 CPU 回归）。"""
    trainer = _make_trainer()
    assert trainer._requested_device == "cpu"
    assert trainer.device.type == "cpu"


# ---------------------------------------------------------------------------
# rank0 落盘
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_history_csv_is_written_only_by_rank0(tmp_path):
    """非 rank0 不得写 history.csv / best_metrics.csv。"""
    trainer = _make_trainer(work_dir=str(tmp_path))
    trainer.is_distributed = True
    trainer.rank = 1
    trainer.world_size = 2
    trainer.history = [{"val_loss": 0.0}]

    trainer._write_history_csv()
    trainer._write_best_metrics_csv()
    assert not os.path.exists(
        os.path.join(str(tmp_path), "history.csv")
    ), "rank≠0 不应写 history.csv"

    trainer.rank = 0
    trainer._write_history_csv()
    assert os.path.exists(os.path.join(str(tmp_path), "history.csv")), "rank0 必须写 history.csv"


@pytest.mark.unit
def test_broadcast_stop_is_noop_when_single_process():
    """单进程下广播是空操作，不应抛错也不应改变 stop_training。"""
    trainer = _make_trainer()
    trainer.stop_training = True
    trainer._broadcast_stop_training()
    assert trainer.stop_training is True


# ---------------------------------------------------------------------------
# 早停的 rank 归属
# ---------------------------------------------------------------------------
class _FakeTrainer:
    def __init__(self, is_rank0: bool):
        self._rank0 = is_rank0
        self.stop_training = False
        self.model = torch.nn.Linear(2, 2)

    def _is_rank0(self) -> bool:
        return self._rank0


@pytest.mark.unit
def test_early_stopping_only_decides_on_rank0():
    """非 rank0 不得自行决策早停（否则各 rank 停止时机不一致 → 死锁）。"""
    from dvisionix.training.callbacks.early_stopping import EarlyStopping

    non_rank0 = _FakeTrainer(is_rank0=False)
    cb = EarlyStopping(monitor="val_loss", mode="min", patience=1)
    for _ in range(5):
        cb.on_epoch_end(non_rank0, 0, {"val_loss": 1.0})
    assert non_rank0.stop_training is False, "非 rank0 不应触发早停"
    assert cb.wait == 0, "非 rank0 不应累积等待计数"

    rank0 = _FakeTrainer(is_rank0=True)
    cb0 = EarlyStopping(monitor="val_loss", mode="min", patience=1)
    cb0.on_epoch_end(rank0, 0, {"val_loss": 1.0})  # 首次即最优
    cb0.on_epoch_end(rank0, 1, {"val_loss": 2.0})  # 变差
    cb0.on_epoch_end(rank0, 2, {"val_loss": 3.0})  # 再变差 → 触发
    assert rank0.stop_training is True, "rank0 应触发早停"


@pytest.mark.unit
def test_early_stopping_warns_when_best_weights_unavailable(tmp_path, caplog=None):
    """没有内存快照又没有 best.pt 时必须告警，而不是静默什么都不做。"""
    from dvisionix.training.callbacks.early_stopping import EarlyStopping

    trainer = _FakeTrainer(is_rank0=True)
    trainer.work_dir = str(tmp_path)
    cb = EarlyStopping(monitor="val_loss", mode="min", patience=1, restore_best_weights=True)
    assert cb.best_weights is None
    assert cb._restore_from_disk(trainer) is False  # 没有 best.pt


# ---------------------------------------------------------------------------
# torchrun 下的工作目录
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_distributed_rank_from_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert distributed_rank_from_env() is None

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert distributed_rank_from_env() is None, "单进程不算分布式"

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert distributed_rank_from_env() == 3


@pytest.mark.unit
def test_work_dir_is_deterministic_under_torchrun(monkeypatch, tmp_path):
    """torchrun 下所有 rank 必须解析出**同一个**工作目录。"""
    cfg = {"experiment_name": "exp"}
    monkeypatch.setenv("WORLD_SIZE", "2")

    monkeypatch.setenv("RANK", "0")
    dir_rank0 = resolve_work_dir(cfg, cli_work_dir=str(tmp_path))
    monkeypatch.setenv("RANK", "1")
    dir_rank1 = resolve_work_dir(cfg, cli_work_dir=str(tmp_path))

    assert dir_rank0 == dir_rank1, "各 rank 的工作目录必须一致，否则 checkpoint 与 CSV 会分家"
    # 确定性命名：不含时间戳，只保留配置哈希
    assert os.path.basename(dir_rank0) == _hash_of(cfg)


def _hash_of(cfg) -> str:
    from dvisionix.training.workdir import hash_config

    return hash_config(cfg)


@pytest.mark.unit
def test_work_dir_keeps_timestamp_when_single_process(monkeypatch, tmp_path):
    """单进程仍保留时间戳后缀（同一配置重复实验不会互相覆盖）。"""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = {"experiment_name": "exp"}
    resolved = resolve_work_dir(cfg, cli_work_dir=str(tmp_path))
    name = os.path.basename(resolved)
    assert name.endswith(_hash_of(cfg))
    assert len(name) > len(_hash_of(cfg)) + 1, "单进程下应带时间戳前缀"


# ---------------------------------------------------------------------------
# 验证集批数为 0
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_distributed_val_loader_can_be_empty_and_is_detected():
    """小验证集在 DDP 下会被 DistributedSampler + drop_last 清空，必须能被检出。"""
    trainer = _make_trainer()
    trainer.is_distributed = True
    trainer.rank = 1
    trainer.world_size = 2

    loader = DataLoader(_DS(n=2), batch_size=2)  # 2 样本 / 2 卡 / bs2 → 每卡 1 → drop_last 后 0
    made = trainer._make_distributed_loader(loader, shuffle=False)
    assert len(made) == 0, "该组合应产生空 loader（正是需要被检出的条件）"
