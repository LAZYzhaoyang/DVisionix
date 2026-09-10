# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: CPU gloo 双进程 DDP 一致性测试（无需 GPU，可在 CI 长期运行）。
"""CPU gloo 双进程 DDP 一致性测试。

CodePlan 7.4 步骤 2-1 的验收要求：**单进程与双进程的全局指标必须在 1e-6 内一致**。

用 CPU + gloo 后端起两个进程，因此任何机器（含无 GPU 的 CI）都能长期运行；
需要真实 2+ GPU 的 NCCL 冒烟见 `tests/test_training/test_ddp_smoke.py`，
需要 2+ GPU 的机器（本机无 GPU 时自动跳过）。

本文件覆盖 CodePlan 7.1.2 的 D5（DDP 聚合拍平嵌套结构）与 D8（`validate()` 缺
DDP 分支，只统计 rank0 分片）：两者都会让多卡下的评估结果静默错误。
"""

import os
import socket
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dvisionix.training import ClassificationTask, Trainer

pytestmark = pytest.mark.ddp

WORLD_SIZE = 2
NUM_SAMPLES = 8
BATCH_SIZE = 4
NUM_CLASSES = 3


def _distributed_environment_usable() -> tuple:
    """预检：新的 Python 进程能否加载 torch 及其分布式后端。

    Windows 上若未激活 conda 环境，``Library\\bin`` 不在 PATH，子进程 import torch
    会以 ``DLL load failed while importing _C`` 失败。这不是本项目的缺陷，但会让
    mp.spawn 冒出难以理解的错误 —— 因此这里先探测，探测失败就 skip 并给出明确指引。
    """
    probe = "import torch, torch.distributed as d; assert d.is_gloo_available()"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:  # pragma: no cover
        return False, f"探测进程启动失败: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {proc.returncode}"
        return False, f"子进程无法加载 torch 分布式后端: {tail}"
    return True, ""


class _FixedDataset(torch.utils.data.Dataset):
    """确定性数据集：标签为 ``i % NUM_CLASSES``，使指标完全可预测。"""

    def __init__(self, n: int = NUM_SAMPLES):
        self.images = torch.zeros(n, 3, 8, 8)
        self.labels = torch.arange(n) % NUM_CLASSES

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index):
        return {"image": self.images[index], "label": self.labels[index]}


class _ConstantModel(torch.nn.Module):
    """恒定 logits 模型：argmax 恒为类别 0，使指标与随机初始化、样本划分都无关。

    这样「单进程 vs 双进程」的差异只可能来自聚合逻辑本身，而不是训练带来的权重漂移。
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        logits = torch.zeros(x.shape[0], self.num_classes, device=x.device)
        logits[:, 0] = 1.0 + self.dummy
        return logits


def _expected_accuracy() -> float:
    """常量模型下的期望准确率 = 标签为 0 的样本占比。"""
    labels = torch.arange(NUM_SAMPLES) % NUM_CLASSES
    return float((labels == 0).float().mean())


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _validate_single_process() -> dict:
    """单进程基线：不初始化进程组，直接对全部 8 个样本做一次 validate。"""
    dataset = _FixedDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    task = ClassificationTask(num_classes=NUM_CLASSES)
    trainer = Trainer(task, loader, loader, max_epochs=1, log_interval=999)
    with torch.no_grad():
        return trainer.validate(_ConstantModel())


def _worker(rank: int, world_size: int, port: int, queue) -> None:
    """glue 子进程：每个 rank 只持有数据分片，由 all_gather 汇总成全局指标。"""
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        dataset = _FixedDataset()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

        task = ClassificationTask(num_classes=NUM_CLASSES)
        trainer = Trainer(task, loader, loader, max_epochs=1, strategy="ddp", log_interval=999)
        assert trainer.is_distributed, "Trainer 未进入分布式模式"
        assert trainer.world_size == world_size

        with torch.no_grad():
            metrics = trainer.validate(_ConstantModel())

        if rank == 0:
            queue.put({"ok": True, "metrics": dict(metrics), "world_size": trainer.world_size})

        dist.barrier()
    except Exception as exc:  # pragma: no cover - 仅在子进程失败时触发
        import traceback

        queue.put(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()}
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(scope="module")
def gloo_ready():
    """确保本环境能真正跑起 CPU gloo 子进程，否则给出明确原因后跳过。"""
    ok, reason = _distributed_environment_usable()
    if not ok:
        pytest.skip(reason)


def _run_two_processes() -> dict:
    """启动两个 gloo 进程并取回 rank0 的结果。"""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    try:
        mp.spawn(_worker, args=(WORLD_SIZE, port, queue), nprocs=WORLD_SIZE, join=True)
    except Exception as exc:
        if queue.empty():
            raise AssertionError(
                f"双进程启动失败且未回传任何结果：{type(exc).__name__}: {exc}"
            ) from exc
        raise
    assert not queue.empty(), "子进程未回传结果（疑似死锁）"
    result = queue.get()
    assert result["ok"], f"子进程失败：{result.get('error')}\n{result.get('tb', '')}"
    return result


def test_distributed_validate_matches_single_process(gloo_ready):
    """双进程 gloo 的全局指标必须与单进程在 1e-6 内一致。"""
    if not dist.is_available():
        pytest.skip("torch.distributed 不可用")

    single = _validate_single_process()
    assert "accuracy" in single, f"单进程 validate 未返回 accuracy: {sorted(single)}"
    expected = _expected_accuracy()
    assert single["accuracy"] == pytest.approx(
        expected, abs=1e-6
    ), f"单进程基线异常：{single['accuracy']} != {expected}"

    result = _run_two_processes()
    metrics = result["metrics"]
    assert result["world_size"] == WORLD_SIZE
    assert metrics["accuracy"] == pytest.approx(
        single["accuracy"], abs=1e-6
    ), f"双进程全局指标 {metrics['accuracy']} 与单进程 {single['accuracy']} 不一致"
    assert metrics["accuracy"] == pytest.approx(expected, abs=1e-6)


def test_distributed_metric_uses_all_ranks_not_rank0_only(gloo_ready):
    """回归 D8：全局指标必须覆盖所有 rank，而不是只统计 rank0 的分片。

    数据集标签为 [0,1,2,0,1,2,0,1]，类别 0 占 3/8。
    rank0 的分片（前 4 个）里类别 0 只有 1 个（1/4 = 0.25）。
    若只算 rank0，准确率会退化成 0.25；正确的全局值是 0.375。
    """
    if not dist.is_available():
        pytest.skip("torch.distributed 不可用")

    result = _run_two_processes()
    accuracy = result["metrics"]["accuracy"]

    # rank0 分片（前 BATCH_SIZE 个样本）上类别 0 的占比 —— 若只统计 rank0 就会得到它
    rank0_labels = torch.arange(NUM_SAMPLES)[:BATCH_SIZE] % NUM_CLASSES
    rank0_only = float((rank0_labels == 0).float().mean())
    global_expected = _expected_accuracy()

    assert rank0_only != pytest.approx(global_expected, abs=1e-6), "测试数据未构成区分度"
    assert accuracy != pytest.approx(
        rank0_only, abs=1e-6
    ), f"全局指标 {accuracy} 与 rank0 分片指标 {rank0_only} 相同，说明 all_gather 未生效"
    assert accuracy == pytest.approx(global_expected, abs=1e-6)
