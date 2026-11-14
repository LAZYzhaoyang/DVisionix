# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: MultiTaskClassifier 测试（路由式多任务分类模型，契约 P3-1）。
"""MultiTaskClassifier 测试（契约 P3-1）。

覆盖：

- 注册：``MODELS`` 的类名与 ``multi_task_classifier`` 别名、``build_model`` 可构建；
- 只读属性：``task_names`` 顺序稳定、``num_classes`` 是**每次新算的 dict** 且不可赋值；
- 前向：``task=None`` 时全部头对整批输出；给定 ``task`` 时只对子集调用对应头，
  且**骨干只前向一次**；
- 边界：未知任务名 / 长度不符 / 空 heads / 头配置缺 ``num_classes`` 报错。
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from dvisionix.errors import ConfigError
from dvisionix.models import MultiTaskClassifier, build_model
from dvisionix.models.base import model_task_types
from dvisionix.models.heads import ClsHead
from dvisionix.registry import MODELS

pytestmark = pytest.mark.unit

IMAGE_SIZE = 8
TASK_NAMES = ("licence", "logo")
NUM_CLASSES = {"licence": 3, "logo": 2}
STAGES = [
    {"type": "conv_norm_act", "in_channels": 3, "out_channels": 8, "stride": 2},
    {"type": "conv_norm_act", "in_channels": 8, "out_channels": 16, "stride": 2},
]


def _backbone_cfg(**overrides):
    cfg = {"type": "sequential_backbone", "stages": STAGES, "input_size": IMAGE_SIZE}
    cfg.update(overrides)
    return cfg


def _heads_cfg():
    return {name: {"num_classes": NUM_CLASSES[name]} for name in TASK_NAMES}


def _model(**backbone_overrides):
    return MultiTaskClassifier(
        backbone=_backbone_cfg(**backbone_overrides),
        heads=_heads_cfg(),
    )


class _RecordingHead(nn.Module):
    """记录每次前向的输入形状，用于验证"每个头只看到自己的子集"。"""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_channels, num_classes)
        self.input_shapes = []

    def forward(self, x, labels=None):
        self.input_shapes.append(tuple(x.shape))
        return self.fc(x)


# ---------------------------------------------------------------------------
# 注册与构建
# ---------------------------------------------------------------------------


class TestRegistration:
    """模型注册表与配置驱动构建。"""

    def test_registered_under_class_name_and_alias(self):
        assert MODELS.get("MultiTaskClassifier") is MultiTaskClassifier
        assert MODELS.get("multi_task_classifier") is MultiTaskClassifier

    def test_build_from_registry(self):
        model = build_model(
            {
                "type": "multi_task_classifier",
                "backbone": _backbone_cfg(),
                "heads": _heads_cfg(),
            }
        )
        assert isinstance(model, MultiTaskClassifier)
        out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert out["licence"].shape == (2, NUM_CLASSES["licence"])
        assert out["logo"].shape == (2, NUM_CLASSES["logo"])

    def test_supported_tasks_capability(self):
        model = _model()
        assert MultiTaskClassifier.supported_tasks == ("multi_task_classification",)
        assert model_task_types(model) == ("multi_task_classification",)


# ---------------------------------------------------------------------------
# 只读属性
# ---------------------------------------------------------------------------


class TestReadOnlyProperties:
    """``task_names`` / ``num_classes`` 的契约形态。"""

    def test_task_names_preserves_config_order(self):
        model = _model()
        assert model.task_names == TASK_NAMES
        assert isinstance(model.task_names, tuple)
        assert model.task_names == model.task_names  # 多次访问结果一致

    def test_num_classes_is_dict(self):
        assert _model().num_classes == NUM_CLASSES
        assert isinstance(_model().num_classes, dict)

    def test_num_classes_is_computed_not_stored(self):
        """返回的是新字典：外部改动不会污染模型（也说明它不是实例属性）。"""
        model = _model()
        snapshot = model.num_classes
        snapshot["licence"] = 99
        assert model.num_classes == NUM_CLASSES

    def test_num_classes_cannot_be_assigned(self):
        model = _model()
        with pytest.raises(AttributeError):
            model.num_classes = {}

    def test_heads_are_built_with_in_channels(self):
        model = _model()
        for name in TASK_NAMES:
            head = model.heads[name]
            assert isinstance(head, ClsHead)
            assert head.num_classes == NUM_CLASSES[name]
            assert head.fc[-1].in_features == model.num_features


# ---------------------------------------------------------------------------
# 前向：整批 vs 路由
# ---------------------------------------------------------------------------


class TestForward:
    """``task=None`` 全头输出；给 ``task`` 时按样本路由。"""

    def test_forward_without_task_runs_all_heads_on_full_batch(self):
        model = _model()
        x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x)
        assert set(out) == set(TASK_NAMES)
        assert out["licence"].shape == (4, NUM_CLASSES["licence"])
        assert out["logo"].shape == (4, NUM_CLASSES["logo"])

    def test_routed_forward_only_calls_head_on_subset(self):
        model = _model()
        recording = {
            name: _RecordingHead(model.num_features, NUM_CLASSES[name]) for name in TASK_NAMES
        }
        for name, head in recording.items():
            model.heads[name] = head
        x = torch.randn(5, 3, IMAGE_SIZE, IMAGE_SIZE)
        task = ["licence", "logo", "licence", "licence", "logo"]

        out = model(x, task=task)

        # 每个头只被调用一次，且入参是整批特征的子集（不是整批）
        assert recording["licence"].input_shapes == [(3, model.num_features)]
        assert recording["logo"].input_shapes == [(2, model.num_features)]
        assert out["licence"].shape == (3, NUM_CLASSES["licence"])
        assert out["logo"].shape == (2, NUM_CLASSES["logo"])

    def test_routed_forward_selects_correct_rows(self):
        """路由结果是"整批输出按任务切片"，而不是任意子集。"""
        model = _model()
        model.eval()
        x = torch.randn(5, 3, IMAGE_SIZE, IMAGE_SIZE)
        task = ["licence", "logo", "licence", "licence", "logo"]
        with torch.no_grad():
            full = model(x)
            routed = model(x, task=task)
        licence_rows = [index for index, name in enumerate(task) if name == "licence"]
        logo_rows = [index for index, name in enumerate(task) if name == "logo"]
        assert torch.allclose(routed["licence"], full["licence"][licence_rows], atol=1e-6)
        assert torch.allclose(routed["logo"], full["logo"][logo_rows], atol=1e-6)

    def test_backbone_is_forwarded_exactly_once(self):
        model = _model()
        calls = []
        handle = model.backbone.register_forward_hook(lambda *args: calls.append(1))
        try:
            x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
            model(x, task=["licence", "logo", "licence", "logo"])
        finally:
            handle.remove()
        assert len(calls) == 1, "路由不得为每个任务重复过骨干"

    def test_single_task_string_accepted(self):
        model = _model()
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x, task="licence")
        assert set(out) == {"licence"}
        assert out["licence"].shape == (1, NUM_CLASSES["licence"])

    def test_single_task_batch_uses_full_features(self):
        """整批同任务时不切片也能得到同样结果（内部走 index_select 的快路径）。"""
        model = _model()
        model.eval()
        x = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            full = model(x)
            routed = model(x, task=["logo", "logo", "logo"])
        assert torch.allclose(routed["logo"], full["logo"], atol=1e-6)

    def test_features_only_backbone_is_pooled(self):
        """骨干显式 features_only=True 时取最后一层特征图并池化（不静默错位）。"""
        model = _model(features_only=True)
        assert model.backbone.features_only is True
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x, task=["licence", "logo"])
        assert out["licence"].shape == (1, NUM_CLASSES["licence"])
        assert out["logo"].shape == (1, NUM_CLASSES["logo"])


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------


class TestErrors:
    """非法任务名 / 长度 / 头配置必须给出可操作的错误。"""

    def test_unknown_task_name_raises(self):
        model = _model()
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        with pytest.raises(ValueError, match="unknown_task"):
            model(x, task=["licence", "unknown_task"])

    def test_task_length_mismatch_raises(self):
        model = _model()
        x = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
        with pytest.raises(ValueError, match="批大小"):
            model(x, task=["licence", "logo"])

    def test_empty_heads_raises(self):
        with pytest.raises(ConfigError, match="heads"):
            MultiTaskClassifier(backbone=_backbone_cfg(), heads={})

    def test_head_without_num_classes_raises(self):
        with pytest.raises(ConfigError, match="num_classes"):
            MultiTaskClassifier(
                backbone=_backbone_cfg(),
                heads={"licence": {"type": "cls_head"}},
            )

    def test_task_name_with_dot_raises(self):
        with pytest.raises(ConfigError, match="ModuleDict"):
            MultiTaskClassifier(
                backbone=_backbone_cfg(),
                heads={"licence.plate": {"num_classes": 3}},
            )
