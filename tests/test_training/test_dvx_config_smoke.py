# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 训练入口装配冒烟测试（data.dataset=dvx + 任务预设变换）。
"""DVX 训练入口装配冒烟测试（快速，不训练）。

覆盖契约 P2-5 的装配层：

- ``data.dataset: dvx`` + ``data.spec`` 时 ``tools.train.build_data`` 返回的
  ``DataLoader`` 能取出一个 batch（mini 分类数据集，tmp_path 现场生成）；
- 变换来源是 Task（``task.build_transforms(train, image_size)``），入口不再硬编码
  任务分派；
- 缺少 ``data.spec`` / 任务未提供预设变换时抛 ``ConfigError``。

本文件只测装配层，不构建 Trainer、不执行训练，因此可在 CPU 上快速运行。
"""

import cv2
import numpy as np
import pytest
import torch

from dvisionix.config import Config
from dvisionix.data import DVXDataset
from dvisionix.data.transforms import LabelToTensor, ToTensor, TransformPipeline
from dvisionix.errors import ConfigError
from dvisionix.training import ClassificationTask, build_task

pytestmark = pytest.mark.unit

IMAGE_SIZE = 8
CLASSES = ["cat", "dog"]
SAMPLES_PER_CLASS = 4


def _write_image(path):
    """写一张真实可解码的 8x8 小图（imencode 兼容非 ASCII 路径）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(path.suffix.lower(), image)
    assert ok, f"cv2.imencode 失败: {path}"
    path.write_bytes(buffer.tobytes())


@pytest.fixture()
def dvx_cls_spec(tmp_path):
    """迷你 DVX 分类数据集：2 类 x 4 张 8x8 图片 + dvx_dataset.yaml。"""
    root = tmp_path / "data"
    for class_name in CLASSES:
        for index in range(SAMPLES_PER_CLASS):
            _write_image(root / class_name / f"{class_name}_{index}.jpg")

    spec_dir = tmp_path / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / "dvx_dataset.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "version: 1",
                "data_type: classification",
                "layout: folder_per_class",
                "root: ../data",
                "classes:",
                *[f"  - {name}" for name in CLASSES],
                "splits:",
                "  mode: ratio",
                "  ratio: {train: 0.8, val: 0.2}",
                "  seed: 42",
                "  stratify: class",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return spec_path


def _make_config(spec_path, **data_overrides):
    """构造装配层所需的最小 Config（不校验 schema，不建 Trainer）。"""
    data = {"dataset": "dvx", "spec": str(spec_path), "image_size": IMAGE_SIZE}
    data.update(data_overrides)
    return Config(
        {
            "task_type": "classification",
            "model": {"num_classes": len(CLASSES)},
            "data": data,
            "training": {"num_epochs": 1, "batch_size": 2, "num_workers": 0},
        }
    )


def _make_task():
    """构建分类任务实例（预设变换由任务提供）。"""
    return build_task({"type": "ClassificationTask", "num_classes": len(CLASSES)})


def test_dvx_classification_loader_yields_batch(dvx_cls_spec):
    """data.dataset=dvx 时 DataLoader 能取到一个形状/dtype 正确的 batch。"""
    from tools import train as train_tool

    cfg = _make_config(dvx_cls_spec)
    task = _make_task()
    train_loader, val_loader = train_tool.build_data(cfg, task)

    assert isinstance(train_loader.dataset, DVXDataset)
    assert train_loader.dataset.data_type == "classification"
    assert train_loader.dataset.num_classes == len(CLASSES)
    assert len(train_loader.dataset) > 0 and len(val_loader.dataset) > 0
    # 分类数据集没有自定义 collate，DataLoader 使用 PyTorch 默认 collate
    assert train_loader.dataset.collate_fn is None

    batch = next(iter(train_loader))
    assert "image" in batch and "label" in batch
    assert batch["image"].shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["image"].dtype == torch.float32
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == torch.long

    val_batch = next(iter(val_loader))
    assert val_batch["image"].shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_build_data_uses_task_build_transforms(monkeypatch, dvx_cls_spec):
    """变换必须来自 Task：build_data 按 train/val 调用 task.build_transforms。"""
    from tools import train as train_tool

    calls = []
    pipelines = {}

    def _fake_build_transforms(cls, train, image_size):
        calls.append((train, image_size))
        pipeline = TransformPipeline([ToTensor(keys=("image",)), LabelToTensor()])
        pipelines[train] = pipeline
        return pipeline

    # raising=False：Wave 1 完成前也先验证入口接线逻辑，不修改 dvisionix 源码。
    monkeypatch.setattr(
        ClassificationTask,
        "build_transforms",
        classmethod(_fake_build_transforms),
        raising=False,
    )

    cfg = _make_config(dvx_cls_spec)
    task = _make_task()
    train_loader, val_loader = train_tool.build_data(cfg, task)

    assert calls == [(True, IMAGE_SIZE), (False, IMAGE_SIZE)]
    assert train_loader.dataset.transforms is pipelines[True]
    assert val_loader.dataset.transforms is pipelines[False]

    batch = next(iter(train_loader))
    assert batch["image"].shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_task_without_preset_transforms_raises_config_error(monkeypatch, dvx_cls_spec):
    """任务未提供预设变换（返回 None）时，入口必须抛可操作的 ConfigError。"""
    from tools import train as train_tool

    monkeypatch.setattr(
        ClassificationTask,
        "build_transforms",
        classmethod(lambda cls, train, image_size: None),
        raising=False,
    )

    cfg = _make_config(dvx_cls_spec)
    with pytest.raises(ConfigError, match="未提供预设变换"):
        train_tool.build_data(cfg, _make_task())


def test_dvx_without_spec_raises_config_error():
    """data.dataset=dvx 但缺少 data.spec 时，必须在装配前给出明确错误。"""
    from tools import train as train_tool

    cfg = _make_config("./placeholder/dvx_dataset.yaml", spec=None)
    with pytest.raises(ConfigError, match="data.spec"):
        train_tool.build_data(cfg, _make_task())


# ----------------------------------------------------------------------
# P3：路由式多任务分类装配
# ----------------------------------------------------------------------

MULTI_TASK_NAMES = ("licence", "logo")
MULTI_TASK_CLASSES = ["a", "b"]
MULTI_TASK_SAMPLES_PER_CLASS = 2


@pytest.fixture()
def dvx_multi_task_spec(tmp_path):
    """迷你 DVX 多任务数据集：2 任务 x 2 类 x 2 张 8x8 图片 + dvx_dataset.yaml。

    按契约 §5 构造（``<任务>/<类别>/*.jpg``）。ratio 取 0.5/0.5：在按
    ``(task, label)`` 分层的 2 样本小组里各分 1 张，train/val 都非空。
    """
    root = tmp_path / "data"
    for task_name in MULTI_TASK_NAMES:
        for class_name in MULTI_TASK_CLASSES:
            for index in range(MULTI_TASK_SAMPLES_PER_CLASS):
                _write_image(root / task_name / class_name / f"{class_name}_{index}.jpg")

    spec_dir = tmp_path / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / "dvx_dataset.yaml"
    lines = [
        "version: 1",
        "data_type: multi_task_classification",
        "layout: folder_per_class",
        "root: ../data",
        "tasks:",
    ]
    for task_name in MULTI_TASK_NAMES:
        lines += [
            f"  - name: {task_name}",
            f"    path: {task_name}",
            "    classes:",
            *[f"      - {class_name}" for class_name in MULTI_TASK_CLASSES],
        ]
    lines += [
        "splits:",
        "  mode: ratio",
        "  ratio: {train: 0.5, val: 0.5}",
        "  seed: 42",
        "  stratify: class",
        "",
    ]
    spec_path.write_text("\n".join(lines), encoding="utf-8")
    return spec_path


def _make_multi_task_config(spec_path):
    """构造多任务装配所需的最小 Config（无全局 model.num_classes）。"""
    return Config(
        {
            "task_type": "multi_task_classification",
            "model": {
                "type": "multi_task_classifier",
                "backbone": {"type": "mobilenetv3_backbone", "features_only": False},
                "heads": {
                    task_name: {
                        "type": "cls_head",
                        "num_classes": len(MULTI_TASK_CLASSES),
                    }
                    for task_name in MULTI_TASK_NAMES
                },
            },
            "data": {
                "dataset": "dvx",
                "spec": str(spec_path),
                "image_size": IMAGE_SIZE,
            },
            "training": {"num_epochs": 1, "batch_size": 2, "num_workers": 0},
        }
    )


def test_multi_task_dvx_assembly_routes_by_task(dvx_multi_task_spec):
    """多任务 DVX 装配：不注入 num_classes，``batch["task"]`` 为 list[str]，模型可按 task 路由。"""
    from dvisionix.models import build_model
    from tools import train as train_tool

    cfg = _make_multi_task_config(dvx_multi_task_spec)

    # 1) 任务装配：多任务不注入全局 num_classes（类别数在 model.heads 中）
    task_cfg = train_tool.build_task_cfg(cfg)
    assert task_cfg["type"] == "MultiTaskClassificationTask"
    assert "num_classes" not in task_cfg

    task = build_task(task_cfg)

    # 2) 数据装配：DataLoader 能出 batch，且 task 字段保持 list[str]
    train_loader, val_loader = train_tool.build_data(cfg, task)
    assert train_loader.dataset.data_type == "multi_task_classification"
    assert train_loader.dataset.task_names == list(MULTI_TASK_NAMES)
    assert len(train_loader.dataset) > 0 and len(val_loader.dataset) > 0

    batch = next(iter(train_loader))
    assert isinstance(batch["task"], list)
    assert all(isinstance(name, str) for name in batch["task"])
    assert len(batch["task"]) == batch["label"].shape[0] == batch["image"].shape[0]

    # 3) 模型装配：多任务模型按样本 task 路由到对应头
    model = build_model(cfg.model.to_dict())
    assert model.task_names == MULTI_TASK_NAMES
    assert model.num_classes == {
        task_name: len(MULTI_TASK_CLASSES) for task_name in MULTI_TASK_NAMES
    }

    model.eval()
    with torch.no_grad():
        outputs = model(batch["image"], task=batch["task"])

    expected_counts = {name: batch["task"].count(name) for name in set(batch["task"])}
    assert set(outputs) == set(expected_counts)
    for name, count in expected_counts.items():
        assert outputs[name].shape == (count, len(MULTI_TASK_CLASSES))
