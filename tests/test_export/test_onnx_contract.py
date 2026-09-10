# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: ONNX 导出契约测试：嵌套输出、状态保护、dynamo 参数、ORT 数值验证。
"""ONNX 导出契约测试（CodePlan 7.5 步骤 3-4 / 7.1.2 D12）。

v1.0.0 的导出器有四类问题：

- ``_flatten_outputs`` 只处理一层：检测模型常见的「dict 里含 list」或「嵌套 dict」
  会把 list 本身当作 Tensor 传给 ``.numpy()`` 而崩溃；
- ``__init__`` 里直接 ``model.to(device).eval()``，导出一次就永久改变调用者模型；
- ``backend='dynamo'`` 不传 ``input_names``/``output_names``，且对 ``dynamic_axes``
  静默忽略 —— 「动态 batch」的承诺失效；
- ``verify`` 直接 ``.numpy()``（CUDA 或 requires_grad 的张量会抛错），
  且用 ``zip`` 对比输出，数量不一致时**静默截断**。
"""

import json
import os

import pytest
import torch
import torch.nn as nn

from dvisionix.export import ONNXExporter
from dvisionix.export.onnx_exporter import _flatten_outputs
from dvisionix.models import SimpleCNN

try:
    import onnxruntime  # noqa: F401

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import onnx  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class NestedOutputModel(nn.Module):
    """模拟检测头：输出是「dict 里含 list」再套一层 dict，共 3 个 Tensor。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        feat = self.conv(x)
        return {
            "cls": [feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))],
            "box": {"main": feat.flatten(2).mean(-1)},
        }


class BoomModel(nn.Module):
    def forward(self, x):
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# 递归展平
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlattenOutputs:
    def test_flat_tensor(self):
        tensors, names, paths = _flatten_outputs(torch.zeros(2, 3))
        assert len(tensors) == 1 and names == ["output"] and paths == {"output": "output"}

    def test_dict_of_tensors_keeps_keys(self):
        tensors, names, paths = _flatten_outputs({"logits": torch.zeros(1), "feat": torch.zeros(1)})
        assert names == ["logits", "feat"]
        assert paths == {"logits": "logits", "feat": "feat"}

    def test_list_inside_dict_is_flattened(self):
        """v1.0.0 的崩溃点：dict 的值是 list 时会把 list 当成 Tensor。"""
        tensors, names, paths = _flatten_outputs({"cls": [torch.zeros(1), torch.zeros(1)]})
        assert len(tensors) == 2 and all(isinstance(t, torch.Tensor) for t in tensors)
        assert names == ["cls_0", "cls_1"]
        assert paths == {"cls_0": "cls[0]", "cls_1": "cls[1]"}

    def test_nested_dict(self):
        _, names, paths = _flatten_outputs({"a": {"b": torch.zeros(1)}})
        assert names == ["a_b"]
        assert paths == {"a_b": "a.b"}

    def test_tuple_of_lists(self):
        """检测任务 decode 风格的 (boxes_list, scores_list, labels_list)。"""
        _, names, paths = _flatten_outputs(
            ([torch.zeros(1), torch.zeros(1)], [torch.zeros(1)], [torch.zeros(1)])
        )
        assert names == ["output_0_0", "output_0_1", "output_1_0", "output_2_0"]
        assert paths["output_0_1"] == "[0][1]"

    def test_non_tensor_leaf_raises(self):
        with pytest.raises(TypeError, match="叶子类型"):
            _flatten_outputs({"a": 1})

    def test_no_tensor_raises(self):
        with pytest.raises(ValueError, match="不包含任何 Tensor"):
            _flatten_outputs([])


# ---------------------------------------------------------------------------
# 模型状态保护
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelStatePreservation:
    def test_construction_does_not_mutate_model(self):
        model = SimpleCNN(num_classes=3, in_channels=3)
        model.train()
        ONNXExporter(model, input_shape=(3, 32, 32))
        assert model.training is True, "构造导出器不得改变调用者模型的 train/eval 状态"

    @pytest.mark.skipif(not HAS_ONNX, reason="onnx 未安装")
    def test_export_restores_training_mode(self, tmp_path):
        model = SimpleCNN(num_classes=3, in_channels=3)
        model.train()
        exporter = ONNXExporter(model, input_shape=(3, 32, 32))
        exporter.export(os.path.join(str(tmp_path), "m.onnx"))
        assert model.training is True, "导出结束后必须恢复原有的 train/eval 状态"

    @pytest.mark.skipif(not HAS_ONNX, reason="onnx 未安装")
    def test_export_restores_state_even_on_failure(self, tmp_path):
        """try/finally：导出中途抛错也必须恢复状态。"""
        model = BoomModel()
        model.train()
        exporter = ONNXExporter(model, input_shape=(3, 8, 8))
        with pytest.raises(RuntimeError, match="boom"):
            exporter.export(os.path.join(str(tmp_path), "b.onnx"))
        assert model.training is True


# ---------------------------------------------------------------------------
# dynamo 参数契约
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dynamo_rejects_dynamic_axes(tmp_path):
    """dynamo 导出用 dynamic_shapes，必须显式拒绝而不是静默忽略 dynamic_axes。"""
    exporter = ONNXExporter(SimpleCNN(num_classes=3), input_shape=(3, 32, 32))
    with pytest.raises(ValueError, match="dynamic_axes"):
        exporter.export(os.path.join(str(tmp_path), "d.onnx"), backend="dynamo", dynamic_batch=True)


@pytest.mark.unit
def test_invalid_backend_raises(tmp_path):
    exporter = ONNXExporter(SimpleCNN(num_classes=3), input_shape=(3, 32, 32))
    with pytest.raises(ValueError, match="backend"):
        exporter.export(os.path.join(str(tmp_path), "x.onnx"), backend="nope")


# ---------------------------------------------------------------------------
# ORT 端到端验证
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime 未安装")
class TestOrtVerification:
    def test_classification_roundtrip(self, tmp_path):
        path = os.path.join(str(tmp_path), "cls.onnx")
        exporter = ONNXExporter(SimpleCNN(num_classes=4), input_shape=(3, 32, 32))
        exporter.export(path)
        assert exporter.verify(path, num_samples=2)

    def test_nested_detection_like_output_roundtrip(self, tmp_path):
        """嵌套输出（dict 含 list + 嵌套 dict）必须能导出并通过数值验证。"""
        path = os.path.join(str(tmp_path), "nested.onnx")
        exporter = ONNXExporter(NestedOutputModel(), input_shape=(3, 16, 16))
        exporter.export(path)
        assert exporter.verify(path, num_samples=2)

    @pytest.mark.skipif(not HAS_ONNX, reason="onnx 未安装")
    def test_output_path_map_is_written_to_metadata(self, tmp_path):
        import onnx

        path = os.path.join(str(tmp_path), "nested_meta.onnx")
        exporter = ONNXExporter(NestedOutputModel(), input_shape=(3, 16, 16))
        exporter.export(path)

        model = onnx.load(path)
        props = {p.key: p.value for p in model.metadata_props}
        assert "output_path_map" in props, sorted(props)
        mapping = json.loads(props["output_path_map"])
        assert mapping["cls_0"] == "cls[0]"
        assert mapping["box_main"] == "box.main"

    def test_segmentation_like_output_roundtrip(self, tmp_path):
        """分割风格的多输出（主 logits + 辅助 logits）也要能验证。"""

        class SegLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 5, 1)

            def forward(self, x):
                logits = self.conv(x)
                return {"logits": logits, "aux": logits.mean(dim=1, keepdim=True)}

        path = os.path.join(str(tmp_path), "seg.onnx")
        exporter = ONNXExporter(SegLike(), input_shape=(3, 16, 16))
        exporter.export(path)
        assert exporter.verify(path, num_samples=2)

    def test_dynamic_batch_axis_is_actually_dynamic(self, tmp_path):
        """dynamic_batch=True 导出的模型必须能接受不同 batch size（而不只是签名好看）。"""
        import numpy as np

        path = os.path.join(str(tmp_path), "dyn.onnx")
        exporter = ONNXExporter(SimpleCNN(num_classes=4), input_shape=(3, 32, 32))
        exporter.export(path, dynamic_batch=True)

        session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        for batch in (1, 3):
            feed = {
                session.get_inputs()[0].name: np.random.randn(batch, 3, 32, 32).astype("float32")
            }
            out = session.run(None, feed)
            assert out[0].shape[0] == batch
