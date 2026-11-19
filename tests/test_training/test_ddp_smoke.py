# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DDP 多卡冒烟测试。
"""DDP 多卡冒烟测试。

本测试**必须由 torchrun 启动**（否则单进程跑 DDP 断言没有意义、且在缺进程组时
直接报错）；本机无多卡时同样跳过。在有 2+ 张 CUDA 卡的机器上运行：

    torchrun --nproc_per_node=2 -m pytest tests/test_training/test_ddp_smoke.py -q
"""

import os

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="需要 2+ 张 CUDA 卡（本机无 GPU 时跳过）",
    ),
    pytest.mark.skipif(
        "RANK" not in os.environ or "WORLD_SIZE" not in os.environ,
        reason="必须由 torchrun 启动（未检测到 RANK/WORLD_SIZE）；"
        "请用 torchrun --nproc_per_node=2 -m pytest tests/test_training/test_ddp_smoke.py -q",
    ),
]


def test_ddp_smoke_train_and_metrics():
    from torch.utils.data import DataLoader

    from dvisionix.models import SimpleCNN
    from dvisionix.training import ClassificationTask, Trainer

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world >= 2

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, i):
            return {"image": torch.randn(3, 32, 32), "label": torch.tensor(i % 3)}

    loader = DataLoader(DS(), batch_size=4)

    task = ClassificationTask(num_classes=3)
    trainer = Trainer(task, loader, loader, max_epochs=1, strategy="ddp", seed=42, log_interval=999)
    model = SimpleCNN(num_classes=3, in_channels=3)
    result = trainer.fit(model)
    if rank == 0:
        last = result["history"][-1]
        # v1.1 起 epoch 级指标统一带 val_ 前缀（`accuracy` → `val_accuracy`），
        # 见 docs/history/v1.1_changes.md 1.2；此处断言必须同步，否则该 GPU-only 用例
        # 一旦真的跑起来就会失败（它在 CI 中因无 2 张卡而永远跳过）。
        assert "train_loss" in last and "val_accuracy" in last
    dist.barrier()
