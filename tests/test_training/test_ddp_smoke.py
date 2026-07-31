# -*- coding: utf-8 -*-
"""DDP 多卡冒烟测试。

本机无多卡时自动跳过；在有 2+ 张 CUDA 卡的机器上通过以下命令运行：

    torchrun --nproc_per_node=2 -m pytest tests/test_training/test_ddp_smoke.py -q
"""

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="需要 2+ 张 CUDA 卡（本机无 GPU 时跳过）",
)


def test_ddp_smoke_train_and_metrics():
    from torch.utils.data import DataLoader
    from dvisionix.training import Trainer, ClassificationTask
    from dvisionix.models import SimpleCNN

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
        assert "train_loss" in last and "accuracy" in last
    dist.barrier()