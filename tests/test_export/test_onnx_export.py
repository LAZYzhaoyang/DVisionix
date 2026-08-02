# -*- coding: utf-8 -*-
"""ONNX 导出器测试：单/多输入、dict/多输出、verify、backend、元数据。"""

import os

import pytest
import torch
import torch.nn as nn

from dvisionix.export import ONNXExporter
from dvisionix.models import GridDetectionModel, SimpleCNN

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


class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, a, b):
        return self.fc2(self.fc1(a) + self.fc1(b))


class DictOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return {"logits": self.fc(x), "feat": x.mean(dim=1)}


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime 未安装")
def test_simple_cnn_export_and_verify(tmp_path):
    path = os.path.join(tmp_path, "simple_cnn.onnx")
    ex = ONNXExporter(SimpleCNN(num_classes=4), input_shape=(3, 32, 32), device="cpu")
    ex.export(path)
    assert os.path.exists(path) and os.path.getsize(path) > 0
    assert ex.verify(path, num_samples=2)


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime 未安装")
def test_grid_detection_export_and_verify(tmp_path):
    path = os.path.join(tmp_path, "grid_det.onnx")
    ex = ONNXExporter(GridDetectionModel(num_classes=3), input_shape=(3, 64, 64), device="cpu")
    ex.export(path)
    assert ex.verify(path, num_samples=2)


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime 未安装")
def test_multi_input_export_and_verify(tmp_path):
    path = os.path.join(tmp_path, "multi_input.onnx")
    ex = ONNXExporter(MultiInputModel(), input_shapes=[(8,), (8,)], device="cpu")
    ex.export(path)
    assert ex.verify(path, num_samples=2)


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime 未安装")
def test_dict_output_names_and_verify(tmp_path):
    path = os.path.join(tmp_path, "dict_out.onnx")
    ex = ONNXExporter(DictOutputModel(), input_shape=(8,), device="cpu")
    ex.export(path)
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]
    assert set(out_names) == {"logits", "feat"}  # 按键命名，不再出现 '4'
    assert ex.verify(path, num_samples=2)


def test_backend_validation(tmp_path):
    ex = ONNXExporter(SimpleCNN(num_classes=4), input_shape=(3, 32, 32), device="cpu")
    with pytest.raises(ValueError, match="backend"):
        ex.export(os.path.join(tmp_path, "x.onnx"), backend="foo")


def test_no_input_spec_raises():
    with pytest.raises(ValueError, match="input_shape|input_shapes|dummy_inputs"):
        ONNXExporter(SimpleCNN(num_classes=4), device="cpu")


def test_dummy_inputs_custom(tmp_path):
    ex = ONNXExporter(DictOutputModel(), dummy_inputs=torch.randn(2, 8), device="cpu")
    path = os.path.join(tmp_path, "custom.onnx")
    ex.export(path, dynamic_batch=False)
    assert os.path.exists(path)


def test_dynamo_backend_optional(tmp_path):
    onnxscript = pytest.importorskip("onnxscript")
    del onnxscript
    path = os.path.join(tmp_path, "dynamo.onnx")
    ex = ONNXExporter(SimpleCNN(num_classes=4), input_shape=(3, 32, 32), device="cpu")
    ex.export(path, backend="dynamo")
    assert os.path.exists(path)


@pytest.mark.skipif(not (HAS_ONNX and HAS_ORT), reason="需要 onnx 与 onnxruntime")
def test_metadata_written(tmp_path):
    path = os.path.join(tmp_path, "meta.onnx")
    ex = ONNXExporter(
        SimpleCNN(num_classes=4),
        input_shape=(3, 32, 32),
        device="cpu",
        task_type="classification",
        normalize={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "scale": 1.0 / 255.0,
        },
    )
    ex.export(path)
    import onnx

    model = onnx.load(path)
    keys = {prop.key for prop in model.metadata_props}
    assert "normalize_mean" in keys and "normalize_std" in keys
