# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: GPU 可用性回归：CUDA 上的断点续训与线性评估。
"""GPU 可用性回归测试。

背景（为什么需要这个文件）：

CI 装的是 **CPU-only torch**，因此两条"只在 CUDA 上才走得到"的路径长期未被覆盖：

1. ``Trainer.load_checkpoint`` 用 ``map_location=self.device`` 加载。device 为 cuda 时
   整份 checkpoint（含 ``rng_state["torch"]`` 这个 **CPU** ByteTensor）会被搬到 GPU，
   随后 ``torch.set_rng_state()`` 抛 ``TypeError: RNG state must be a torch.ByteTensor``
   —— **GPU 上 `--resume` 必然崩溃**，而 CPU 下 ``map_location="cpu"`` 是空操作，
   CI 永远看不到。

2. ``LinearEvalTask`` 在 ``configure_optimizers`` 里用默认设备建线性头（CPU），
   而特征来自已搬到 CUDA 的 backbone → ``RuntimeError: Expected all tensors to be on
   the same device, but found cpu and cuda:0``。官方配置 ``configs/classification/
   linear_eval.yaml`` 的 ``device: auto`` 在有 GPU 的机器上就是 cuda，因此该配置
   在 GPU 上**从来跑不起来**；它又被 ``test_e2e_configs.py`` 作为"模板配置"排除在
   端到端门禁之外，于是连"配置能不能跑"这道网也漏掉了它。

另有一条与设备无关、但同样只在续训时才暴露的问题：线性头是 Task 持有的参数，
不在 ``model.state_dict()`` 内，若不实现 ``state_dict()/load_state_dict()``，
续训会重新随机初始化线性头却仍加载旧头的优化器动量。
"""

import os

import pytest
import torch
from torch.utils.data import DataLoader

from dvisionix.models import SimpleCNN
from dvisionix.training import ClassificationTask, LinearEvalTask, Trainer

NUM_CLASSES = 3
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 设备")


class _DS(torch.utils.data.Dataset):
    def __init__(self, n: int = 8):
        gen = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, 32, 32, generator=gen)
        self.labels = torch.arange(n) % NUM_CLASSES

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"image": self.images[i], "label": self.labels[i]}


def _linear_backbone_model(num_classes: int = NUM_CLASSES):
    from dvisionix.models import build_model

    return build_model(
        {
            "type": "linear_classifier",
            "num_classes": num_classes,
            "backbone": {
                "type": "sequential_backbone",
                "features_only": False,
                "stages": [
                    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 16, "stride": 2},
                    {"type": "conv_norm_act", "in_channels": 16, "out_channels": 32, "stride": 2},
                ],
            },
            "head": {"type": "cls_head", "num_classes": num_classes},
        }
    )


# ---------------------------------------------------------------------------
# 1. CUDA 上的断点续训
# ---------------------------------------------------------------------------
@needs_cuda
@pytest.mark.cuda
@pytest.mark.unit
def test_resume_on_cuda_restores_rng_state(tmp_path):
    """device=cuda 时必须能完整恢复 checkpoint（含 RNG 状态）。"""
    loader = DataLoader(_DS(), batch_size=4)

    def make(device: str) -> Trainer:
        return Trainer(
            ClassificationTask(num_classes=NUM_CLASSES),
            loader,
            loader,
            max_epochs=1,
            seed=0,
            log_interval=999,
            device=device,
        )

    path = os.path.join(str(tmp_path), "cuda_ckpt.pt")
    saver = make("cuda")
    assert saver.device.type == "cuda"
    saver.model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3).to("cuda")
    saver.save_checkpoint(path)

    # 修复前：map_location="cuda" 会把 CPU 的 RNG 状态张量搬到 GPU，
    # 这里抛 "RNG state must be a torch.ByteTensor"
    resumed = make("cuda")
    resumed.load_checkpoint(path, SimpleCNN(num_classes=NUM_CLASSES, in_channels=3))
    assert resumed.device.type == "cuda"


@needs_cuda
@pytest.mark.cuda
@pytest.mark.unit
def test_checkpoint_rng_tensor_stays_on_cpu(tmp_path):
    """RNG 状态张量必须留在 CPU（torch.set_rng_state 的硬性要求）。"""
    loader = DataLoader(_DS(), batch_size=4)
    trainer = Trainer(
        ClassificationTask(num_classes=NUM_CLASSES),
        loader,
        loader,
        max_epochs=1,
        seed=0,
        log_interval=999,
        device="cuda",
    )
    trainer.model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3).to("cuda")
    path = os.path.join(str(tmp_path), "rng.pt")
    trainer.save_checkpoint(path)

    state = torch.load(path, map_location="cpu", weights_only=False)["rng_state"]
    assert state["torch"].device.type == "cpu"
    # 直接喂回 set_rng_state 不应抛错
    torch.set_rng_state(state["torch"])


# ---------------------------------------------------------------------------
# 2. LinearEvalTask 的设备一致性
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_linear_eval_head_follows_model_device():
    """线性头必须建在 backbone 所在设备上（CPU 与 CUDA 都要成立）。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(_DS(), batch_size=4)
    task = LinearEvalTask(num_classes=NUM_CLASSES)
    trainer = Trainer(
        task,
        loader,
        loader,
        max_epochs=1,
        device=str(device),
        log_interval=999,
    )
    trainer.fit(_linear_backbone_model())
    assert task.linear is not None
    assert (
        task.linear.weight.device.type == device.type
    ), f"线性头在 {task.linear.weight.device}，模型在 {device}"


@needs_cuda
@pytest.mark.cuda
@pytest.mark.integration
def test_linear_eval_trains_on_cuda():
    """官方 linear_eval 配置的核心路径必须能在 CUDA 上跑完一个 epoch。"""
    loader = DataLoader(_DS(), batch_size=4)
    task = LinearEvalTask(num_classes=NUM_CLASSES)
    trainer = Trainer(
        task,
        loader,
        loader,
        max_epochs=1,
        device="cuda",
        log_interval=999,
    )
    trainer.fit(_linear_backbone_model())
    assert task.linear.weight.device.type == "cuda"


@needs_cuda
@pytest.mark.cuda
@pytest.mark.integration
def test_official_linear_eval_config_runs_on_cuda():
    """用仓库自带配置构建，确认 device=auto 在 GPU 机器上不会崩。"""
    from dvisionix.config import Config
    from dvisionix.models import build_model
    from dvisionix.training import build_task

    cfg = Config.from_yaml("configs/classification/linear_eval.yaml")
    model_cfg = cfg.model.to_dict()
    model_cfg.pop("pretrained_backbone", None)  # 配置里是占位路径，E2E 场景跳过
    num_classes = int(model_cfg["num_classes"])
    model = build_model(model_cfg)
    task = build_task({"type": "LinearEvalTask", "num_classes": num_classes})
    loader = DataLoader(_DS(8), batch_size=8)
    trainer = Trainer(task, loader, loader, max_epochs=1, device="auto", log_interval=999)
    assert trainer.device.type == "cuda"
    trainer.fit(model)


# ---------------------------------------------------------------------------
# 3. 线性头的 checkpoint 往返
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_linear_eval_head_state_dict_roundtrip():
    """state_dict()/load_state_dict() 必须能精确还原线性头权重。"""
    loader = DataLoader(_DS(), batch_size=4)
    task = LinearEvalTask(num_classes=NUM_CLASSES)
    trainer = Trainer(task, loader, loader, max_epochs=1, device="cpu", log_interval=999)
    model = _linear_backbone_model()
    trainer.fit(model)

    saved = task.state_dict()
    assert "linear" in saved and saved["linear"], "state_dict 必须包含线性头权重"

    # 头尚未构建时 load_state_dict 必须先缓存，等 configure_optimizers 建头后套用
    restored = LinearEvalTask(num_classes=NUM_CLASSES)
    restored.load_state_dict(saved)
    assert restored.linear is None, "此时不应已经建头"
    restored.configure_optimizers(_linear_backbone_model())

    assert restored.linear is not None
    for (k1, v1), (k2, v2) in zip(saved["linear"].items(), restored.linear.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2), f"线性头权重 {k1} 未被正确恢复"


@pytest.mark.unit
def test_linear_eval_resume_keeps_head_weights(tmp_path):
    """整条 resume 链路：头权重必须进 checkpoint 并在续训时被恢复。"""
    loader = DataLoader(_DS(), batch_size=4)
    path = os.path.join(str(tmp_path), "linear.pt")

    task = LinearEvalTask(num_classes=NUM_CLASSES)
    trainer = Trainer(task, loader, loader, max_epochs=1, device="cpu", log_interval=999)
    trainer.fit(_linear_backbone_model())
    trainer.save_checkpoint(path)
    expected = {k: v.clone() for k, v in task.linear.state_dict().items()}

    # 直接走 load_checkpoint，避免"续训又会继续训练"污染比对
    resumed_task = LinearEvalTask(num_classes=NUM_CLASSES)
    resumed = Trainer(resumed_task, loader, loader, max_epochs=1, device="cpu", log_interval=999)
    resumed.model = _linear_backbone_model()
    resumed.task.configure_optimizers(resumed.model)  # 建头（等价于 fit 的第一步）
    resumed.load_checkpoint(path, resumed.model)

    assert resumed_task.linear is not None
    for key, value in expected.items():
        assert torch.equal(
            resumed_task.linear.state_dict()[key], value
        ), f"续训后线性头权重 {key} 与 checkpoint 不一致"
