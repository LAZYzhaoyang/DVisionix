# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: P2 任务 / 模型 / 数据契约校验测试。
"""P2 任务 / 模型 / 数据契约校验测试（契约 P2-1 ~ P2-3）。

覆盖：

- ``model_task_types``：``supported_tasks`` 优先、``task_type`` 回退、均无返回 ``()``；
- 7 个内置任务的契约类属性与 ``build_transforms`` 映射；
- 「分类模型 + 分割任务」「检测模型 + 分类任务」必须报 ``TaskModelMismatchError``；
- DVX 数据集 ``data_type`` 不匹配、``num_classes`` 不一致、缺少 batch 字段
  必须报 ``TaskDataMismatchError``；
- 匹配时放行、非 DVX 数据集（无 ``data_type``）跳过数据类型检查；
- Trainer 接线：校验发生在 ``_wrap_model`` / 第一次前向之前。
"""

import types

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dvisionix.data import CustomDataset, DVXDataset, DVXDatasetSpec
from dvisionix.data.spec import ParseConfig, SplitConfig
from dvisionix.errors import TaskDataMismatchError, TaskModelMismatchError
from dvisionix.models import GridDetectionModel, SimpleCNN, SimpleSegmentationModel
from dvisionix.models.base import BaseModel, model_task_types
from dvisionix.training import (
    BaseTask,
    ClassificationTask,
    DetectionTask,
    LinearEvalTask,
    MaskFormerTask,
    MultiLabelTask,
    SegmentationTask,
    SimCLRTask,
    Trainer,
)
from dvisionix.training.contract import check_batch_fields, check_task_data, check_task_model

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 夹具辅助
# ---------------------------------------------------------------------------


def _write_image(path, size=8):
    """写一张真实可解码的小图（避免依赖外部数据目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok, f"cv2.imencode 失败: {path}"
    path.write_bytes(buffer.tobytes())


def _make_dvx_classification_dataset(tmp_path, classes=("cat", "dog")):
    """构造一个真实的二分 DVX 分类数据集（每类 2 张，ratio 0.5/0.5）。"""
    root = tmp_path / "dvx"
    for index in range(4):
        cls = classes[index % len(classes)]
        _write_image(root / cls / f"{cls}_{index}.png")
    spec = DVXDatasetSpec(
        version=1,
        data_type="classification",
        layout="folder_per_class",
        root=str(root),
        classes=list(classes),
        tasks=None,
        splits=SplitConfig(
            mode="ratio",
            ratio={"train": 0.5, "val": 0.5},
            seed=42,
            stratify="class",
        ),
        parse=ParseConfig(),
        spec_path=str(tmp_path / "dvx_dataset.yaml"),
    )
    return DVXDataset(spec, split="train")


def _tiny_loader(n=4, num_classes=3):
    """分类 batch 的极简 loader（图像 + label）。"""

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, index):
            return {
                "image": torch.zeros(3, 32, 32),
                "label": torch.tensor(index % num_classes),
            }

    return DataLoader(_DS(), batch_size=2)


# ---------------------------------------------------------------------------
# P2-1 模型能力声明
# ---------------------------------------------------------------------------


class TestModelTaskTypes:
    """``model_task_types`` 的取值优先级。"""

    def test_supported_tasks_takes_priority(self):
        class MultiTaskModel(BaseModel):
            supported_tasks = ("classification", "detection")
            task_type = "segmentation"

        assert model_task_types(MultiTaskModel()) == ("classification", "detection")

    def test_falls_back_to_task_type(self):
        class TaskTypeOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.task_type = "segmentation"

        assert model_task_types(TaskTypeOnly()) == ("segmentation",)

    def test_returns_empty_when_undeclared(self):
        class Plain(torch.nn.Module):
            def forward(self, x):
                return x

        assert model_task_types(Plain()) == ()

    def test_existing_models_keep_declaration(self):
        assert model_task_types(SimpleCNN(num_classes=3)) == ("classification",)
        assert model_task_types(SimpleSegmentationModel(num_classes=3)) == ("segmentation",)
        assert model_task_types(GridDetectionModel(num_classes=3)) == ("detection",)


# ---------------------------------------------------------------------------
# P2-2 Task 契约
# ---------------------------------------------------------------------------


TASK_CONTRACT_TABLE = [
    (
        ClassificationTask,
        "classification",
        ("classification",),
        ("classification",),
        ("image", "label"),
    ),
    (
        MultiLabelTask,
        "multi_label",
        ("classification",),
        ("multi_label",),
        ("image", "label"),
    ),
    (
        DetectionTask,
        "detection",
        ("detection",),
        ("detection",),
        ("image", "boxes", "labels"),
    ),
    (
        SegmentationTask,
        "segmentation",
        ("segmentation",),
        ("segmentation",),
        ("image", "mask"),
    ),
    (
        MaskFormerTask,
        "segmentation",
        ("segmentation",),
        ("segmentation", "instance_segmentation", "panoptic_segmentation"),
        ("image", "mask"),
    ),
    (SimCLRTask, "simclr", ("classification",), ("simclr",), ("image1", "image2")),
    (
        LinearEvalTask,
        "linear_eval",
        ("classification",),
        ("classification",),
        ("image", "label"),
    ),
]


@pytest.mark.parametrize(
    "task_cls,task_type,accepted_model_tasks,accepted_data_types,required_fields",
    TASK_CONTRACT_TABLE,
    ids=[row[0].__name__ for row in TASK_CONTRACT_TABLE],
)
def test_task_contract_attributes(
    task_cls, task_type, accepted_model_tasks, accepted_data_types, required_fields
):
    """内置任务的契约类属性必须与 P2-2 声明表完全一致。"""
    assert task_cls.task_type == task_type
    assert task_cls.accepted_model_tasks == accepted_model_tasks
    assert task_cls.accepted_data_types == accepted_data_types
    assert task_cls.required_batch_fields == required_fields


@pytest.mark.parametrize(
    "task_cls,transform_name",
    [
        (ClassificationTask, "ClassificationTransforms"),
        (MultiLabelTask, "ClassificationTransforms"),
        (LinearEvalTask, "ClassificationTransforms"),
        (DetectionTask, "DetectionTransforms"),
        (SegmentationTask, "SegmentationTransforms"),
        (MaskFormerTask, "SegmentationTransforms"),
        (SimCLRTask, "SimCLRTransforms"),
    ],
)
def test_task_build_transforms(task_cls, transform_name):
    """``build_transforms`` 必须返回对应任务的预设变换。"""
    train_tf = task_cls.build_transforms(train=True, image_size=32)
    val_tf = task_cls.build_transforms(train=False, image_size=32)
    assert type(train_tf).__name__ == transform_name
    assert type(val_tf).__name__ == transform_name
    assert train_tf.train is True
    assert val_tf.train is False


def test_base_task_build_transforms_defaults_to_none():
    assert BaseTask.build_transforms(train=True, image_size=32) is None


# ---------------------------------------------------------------------------
# P2-3 契约校验
# ---------------------------------------------------------------------------


class TestTaskModelContract:
    """``check_task_model``：任务与模型能力 / 类别数校验。"""

    def test_classification_model_with_segmentation_task_raises(self):
        """分类模型 + 分割任务：必须在训练前报 TaskModelMismatchError。"""
        task = SegmentationTask(num_classes=3)
        model = SimpleCNN(num_classes=3, in_channels=3)

        with pytest.raises(TaskModelMismatchError) as excinfo:
            check_task_model(task, model)

        message = str(excinfo.value)
        assert "SegmentationTask" in message
        assert "SimpleCNN" in message
        assert "classification" in message  # 模型声明的能力
        assert "segmentation" in message  # 任务要求的能力
        assert "提示" in message  # 包含修复建议

    def test_detection_model_with_classification_task_raises(self):
        """检测模型 + 分类任务：必须在训练前报 TaskModelMismatchError。"""
        task = ClassificationTask(num_classes=3)
        model = GridDetectionModel(num_classes=3)

        with pytest.raises(TaskModelMismatchError) as excinfo:
            check_task_model(task, model)

        message = str(excinfo.value)
        assert "ClassificationTask" in message
        assert "GridDetectionModel" in message
        assert "detection" in message
        assert "classification" in message
        assert "提示" in message

    def test_num_classes_mismatch_raises(self, tmp_path):
        """DVX 数据集 num_classes 与模型不一致：报 TaskDataMismatchError。"""
        dataset = _make_dvx_classification_dataset(tmp_path)  # 2 类
        task = ClassificationTask(num_classes=3)
        model = SimpleCNN(num_classes=3, in_channels=3)

        with pytest.raises(TaskDataMismatchError) as excinfo:
            check_task_model(task, model, dataset)

        message = str(excinfo.value)
        assert "num_classes" in message
        assert "2" in message and "3" in message
        assert "提示" in message

    def test_matching_passes(self, tmp_path):
        """分类任务 + 分类模型 + 同类 DVX 数据集：放行。"""
        dataset = _make_dvx_classification_dataset(tmp_path)  # 2 类
        check_task_model(ClassificationTask(num_classes=2), SimpleCNN(num_classes=2), dataset)
        check_task_data(ClassificationTask(num_classes=2), dataset)

    def test_undeclared_task_capabilities_skip_model_check(self):
        """自定义任务未声明 accepted_model_tasks 时跳过能力校验（渐进式契约）。"""

        class UndeclaredTask(ClassificationTask):
            accepted_model_tasks = ()

        task = UndeclaredTask(num_classes=3)
        # 检测模型也不报错：任务没有声明任何能力，无从判断
        check_task_model(task, GridDetectionModel(num_classes=3))


class TestTaskDataContract:
    """``check_task_data``：DVX data_type 校验与非 DVX 放行。"""

    def test_dvx_data_type_mismatch_raises(self, tmp_path):
        """DVX 分类数据集 + 检测任务：报 TaskDataMismatchError。"""
        dataset = _make_dvx_classification_dataset(tmp_path)
        task = DetectionTask(num_classes=2)

        with pytest.raises(TaskDataMismatchError) as excinfo:
            check_task_data(task, dataset)

        message = str(excinfo.value)
        assert "DVXDataset" in message
        assert "classification" in message  # dataset 的 data_type
        assert "detection" in message  # 任务要求的集合
        assert "提示" in message

    def test_non_dvx_dataset_skips_data_type_check(self):
        """没有 data_type 属性的数据集（CustomDataset / 公开预设）直接放行。"""
        dataset = CustomDataset(samples=[{"image": "unused.png", "mask": "unused.png"}])
        assert not hasattr(dataset, "data_type")
        # 即便任务与样本结构不匹配，本函数也不应仅凭 data_type 报错
        check_task_data(SegmentationTask(num_classes=3), dataset)

    def test_undeclared_data_types_skip_data_check(self, tmp_path):
        """自定义任务未声明 accepted_data_types 时跳过 data_type 校验。"""

        class UndeclaredTask(ClassificationTask):
            accepted_data_types = ()

        dataset = _make_dvx_classification_dataset(tmp_path)
        check_task_data(UndeclaredTask(num_classes=2), dataset)


class TestBatchFieldContract:
    """``check_batch_fields``：首个训练 batch 的字段校验。"""

    def test_missing_batch_fields_raises(self):
        batch = {"image": torch.zeros(2, 3, 8, 8)}

        with pytest.raises(TaskDataMismatchError) as excinfo:
            check_batch_fields(ClassificationTask(num_classes=3), batch)

        message = str(excinfo.value)
        assert "label" in message  # 缺失字段
        assert "image" in message  # 实际 batch 键
        assert "data_type" in message and "transforms" in message and "collate" in message
        assert "提示" in message

    def test_complete_batch_passes(self):
        batch = {
            "image": torch.zeros(2, 3, 8, 8),
            "label": torch.tensor([0, 1]),
        }
        check_batch_fields(ClassificationTask(num_classes=3), batch)

    def test_non_dict_batch_raises(self):
        with pytest.raises(TaskDataMismatchError, match="字典 batch"):
            check_batch_fields(ClassificationTask(num_classes=3), [torch.zeros(2, 3, 8, 8)])


# ---------------------------------------------------------------------------
# Trainer 接线
# ---------------------------------------------------------------------------


class TestTrainerWiring:
    """Trainer.fit 必须在训练开始前完成校验。"""

    def test_fit_rejects_mismatch_before_model_wrap(self):
        """模型/任务错配：fit 立即报错，此时 model 尚未被包装/搬设备。"""
        trainer = Trainer(
            SegmentationTask(num_classes=3),
            _tiny_loader(),
            max_epochs=1,
            device="cpu",
            log_interval=999,
        )
        with pytest.raises(TaskModelMismatchError):
            trainer.fit(SimpleCNN(num_classes=3, in_channels=3))
        assert trainer.model is None, "校验必须发生在 _wrap_model 之前"
        assert trainer.global_step == 0, "校验失败时不得进入训练循环"

    def test_fit_rejects_wrong_batch_fields_before_first_step(self):
        """batch 字段不匹配：首个 batch 即报错，且没有执行任何优化步。"""

        class _WrongDS(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return {"image": torch.zeros(3, 8, 8), "mask": torch.zeros(8, 8).long()}

        loader = DataLoader(_WrongDS(), batch_size=2)
        trainer = Trainer(
            ClassificationTask(num_classes=3),
            loader,
            max_epochs=1,
            device="cpu",
            log_interval=999,
        )
        with pytest.raises(TaskDataMismatchError, match="label"):
            trainer.fit(SimpleCNN(num_classes=3, in_channels=3))
        assert trainer.global_step == 0, "字段校验必须发生在第一次优化器步进之前"

    def test_fit_accepts_matching_non_dvx_dataset(self):
        """匹配的非 DVX 数据集（无 data_type）应正常训练，不被契约误拦。"""
        trainer = Trainer(
            ClassificationTask(num_classes=3),
            _tiny_loader(n=4),
            _tiny_loader(n=4),
            max_epochs=1,
            device="cpu",
            log_interval=999,
        )
        result = trainer.fit(SimpleCNN(num_classes=3, in_channels=3))
        assert result["global_step"] > 0


# ---------------------------------------------------------------------------
# 契约结构自检
# ---------------------------------------------------------------------------


def test_contract_module_exports_all_checks():
    from dvisionix.training import contract

    for name in ("check_task_model", "check_task_data", "check_batch_fields"):
        assert callable(getattr(contract, name))


def test_dataset_stub_without_data_type_is_skipped():
    """任意没有 data_type 的对象都应被 check_task_data 放行。"""
    check_task_data(SegmentationTask(num_classes=3), types.SimpleNamespace())
