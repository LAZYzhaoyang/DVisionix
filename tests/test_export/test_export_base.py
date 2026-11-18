# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: P5 导出核心契约测试：ExportResult / VerifyReport / 注册表 / 占位导出器。
"""P5 导出核心契约测试（契约 §6 的 A 部分）。

覆盖：

- :class:`ExportResult` 的 ``to_dict`` / ``from_dict`` / ``save`` / ``load`` 往返
  （含 :class:`~dvisionix.inference.preprocess.PreprocessSpec` 与动态轴）；
- 路径协议：``str(result)`` / ``os.fspath(result)`` / ``open(result)``；
- :class:`VerifyReport` 的布尔语义与 ``to_dict``；
- ``EXPORTERS`` 注册表、``build_exporter`` 构建与未知格式的 ``ConfigError``；
- ``TensorRTExporter`` 占位注册与 ``NotImplementedError``；
- ``ONNXExporter.export`` 返回结果字段、``preprocess`` 推断与 metadata 键；
- ``ONNXExporter.verify`` 返回报告并回填 ``parity_max_abs_diff``。
"""

import json
import os

import pytest

from dvisionix.errors import ConfigError
from dvisionix.export import (
    EXPORTERS,
    BaseExporter,
    ExportResult,
    ONNXExporter,
    TensorRTExporter,
    VerifyReport,
    build_exporter,
)
from dvisionix.inference import PreprocessSpec
from dvisionix.models import SimpleCNN

try:
    import onnx  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime  # noqa: F401

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def _make_result(path: str = "model.onnx") -> ExportResult:
    """构造一个字段齐全的导出结果，供序列化测试复用。"""
    return ExportResult(
        path=path,
        format="onnx",
        input_names=["input"],
        input_shapes=[[1, 3, 32, 32]],
        output_names=["logits", "feat"],
        opset=17,
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        preprocess=PreprocessSpec(
            task_type="classification",
            image_size=32,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            scale=1.0 / 255.0,
        ),
        parity_max_abs_diff=1.25e-6,
        extra={"backend": "trace", "备注": "中文元数据"},
    )


# ---------------------------------------------------------------------------
# ExportResult 序列化
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_export_result_dict_roundtrip_with_preprocess():
    """to_dict/from_dict 往返保留全部字段，preprocess 还原为 PreprocessSpec。"""
    result = _make_result()
    data = result.to_dict()

    assert data["preprocess"] == result.preprocess.to_dict()
    restored = ExportResult.from_dict(data)
    assert restored == result
    assert isinstance(restored.preprocess, PreprocessSpec)
    assert restored.dynamic_axes == {"input": {0: "batch_size"}, "logits": {0: "batch_size"}}


@pytest.mark.unit
def test_export_result_save_load_roundtrip(tmp_path):
    """save/load 往返；JSON 采用 indent=2、ensure_ascii=False。"""
    result = _make_result()
    sidecar = os.path.join(str(tmp_path), "model.onnx.json")
    result.save(sidecar)

    with open(sidecar, "r", encoding="utf-8") as handle:
        raw = handle.read()
    assert '\n  "path"' in raw, "sidecar 必须使用 indent=2 缩进"
    assert "中文元数据" in raw, "sidecar 必须 ensure_ascii=False（中文不转义）"

    loaded = ExportResult.load(sidecar)
    assert loaded == result
    # JSON 会把 dynamic_axes 的 int 轴键变成字符串，from_dict 必须还原为 int。
    assert loaded.dynamic_axes == {"input": {0: "batch_size"}, "logits": {0: "batch_size"}}


@pytest.mark.unit
def test_export_result_minimal_defaults():
    """可选字段缺省时保持契约默认值。"""
    result = ExportResult(
        path="m.onnx",
        format="onnx",
        input_names=["input"],
        input_shapes=[[1, 3, 8, 8]],
        output_names=["output"],
    )
    assert result.opset is None
    assert result.dynamic_axes is None
    assert result.preprocess is None
    assert result.parity_max_abs_diff is None
    assert result.extra == {}


@pytest.mark.unit
def test_export_result_path_protocol(tmp_path):
    """str / os.fspath / open / os.path.exists 均可直接使用结果对象。"""
    target = os.path.join(str(tmp_path), "model.onnx")
    result = _make_result(target)

    assert str(result) == target
    assert os.fspath(result) == target

    with open(result, "w", encoding="utf-8") as handle:
        handle.write("onnx-bytes")
    assert os.path.exists(result)
    with open(result, "r", encoding="utf-8") as handle:
        assert handle.read() == "onnx-bytes"


# ---------------------------------------------------------------------------
# VerifyReport
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_report_bool_semantics():
    """bool(report) 等价于 passed；to_dict 保留逐输出明细。"""
    passed = VerifyReport(
        passed=True,
        max_abs_diff=1e-6,
        outputs=[{"name": "output", "shape": [1, 4], "max_abs_diff": 1e-6, "passed": True}],
    )
    failed = VerifyReport(passed=False, max_abs_diff=1.0)

    assert bool(passed) is True
    assert bool(failed) is False
    assert passed.to_dict() == {
        "passed": True,
        "max_abs_diff": 1e-6,
        "outputs": [{"name": "output", "shape": [1, 4], "max_abs_diff": 1e-6, "passed": True}],
    }
    assert failed.to_dict()["outputs"] == []


# ---------------------------------------------------------------------------
# 注册表与 build_exporter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_builtin_exporters_registered():
    """onnx 与 tensorrt 均已注册，且注册项就是导出器类。"""
    assert "onnx" in EXPORTERS
    assert "tensorrt" in EXPORTERS
    assert EXPORTERS.get("onnx") is ONNXExporter
    assert EXPORTERS.get("tensorrt") is TensorRTExporter
    assert issubclass(ONNXExporter, BaseExporter)
    assert issubclass(TensorRTExporter, BaseExporter)
    assert ONNXExporter.format == "onnx"
    assert TensorRTExporter.format == "tensorrt"


@pytest.mark.unit
def test_build_exporter_onnx_forwards_kwargs():
    """build_exporter 按注册名构建并透传构造参数。"""
    exporter = build_exporter(
        "onnx",
        SimpleCNN(num_classes=4),
        input_shape=(3, 32, 32),
        device="cpu",
        task_type="classification",
    )
    assert isinstance(exporter, ONNXExporter)
    assert isinstance(exporter, BaseExporter)
    assert exporter.task_type == "classification"
    assert tuple(exporter.dummy_inputs[0].shape) == (1, 3, 32, 32)


@pytest.mark.unit
def test_build_exporter_unknown_format_raises_config_error():
    """未知格式必须抛 ConfigError，且消息列出全部已注册格式。"""
    with pytest.raises(ConfigError) as excinfo:
        build_exporter("not-a-format", SimpleCNN(num_classes=4))
    message = str(excinfo.value)
    assert "onnx" in message
    assert "tensorrt" in message


@pytest.mark.unit
def test_build_exporter_tensorrt_placeholder():
    """tensorrt 可被构建为占位导出器。"""
    exporter = build_exporter("tensorrt", SimpleCNN(num_classes=4))
    assert isinstance(exporter, TensorRTExporter)


@pytest.mark.unit
def test_tensorrt_export_not_implemented(tmp_path):
    """TensorRT 扩展点的 export 必须显式抛 NotImplementedError。"""
    exporter = build_exporter("tensorrt", SimpleCNN(num_classes=4))
    with pytest.raises(NotImplementedError, match="TensorRT"):
        exporter.export(os.path.join(str(tmp_path), "model.trt"))


@pytest.mark.unit
def test_base_exporter_verify_not_implemented():
    """BaseExporter.verify 默认抛 NotImplementedError（明确未实现，而非静默通过）。"""

    class StubExporter(BaseExporter):
        format = "stub"

        def export(self, output_path, **kwargs):
            raise NotImplementedError

    with pytest.raises(NotImplementedError, match="verify"):
        StubExporter(SimpleCNN(num_classes=4)).verify("stub.bin")


# ---------------------------------------------------------------------------
# ONNXExporter：导出结果字段 / preprocess / metadata
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_onnx_export_returns_export_result(tmp_path):
    """export 返回 ExportResult，字段与传参一致，且不写 verify 误差。"""
    output_path = os.path.join(str(tmp_path), "result.onnx")
    normalize = {
        "mean": [0.1, 0.2, 0.3],
        "std": [0.4, 0.5, 0.6],
        "scale": 1.0 / 255.0,
    }
    exporter = ONNXExporter(
        SimpleCNN(num_classes=4),
        input_shape=(3, 32, 32),
        device="cpu",
        task_type="classification",
        normalize=normalize,
    )
    result = exporter.export(output_path, dynamic_batch=True)

    assert isinstance(result, ExportResult)
    assert result.path == output_path
    assert result.format == "onnx"
    assert result.input_names == ["input"]
    assert result.input_shapes == [[1, 3, 32, 32]]
    assert result.output_names == ["output"]
    assert result.opset == 17
    assert result.dynamic_axes == {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    assert result.extra["backend"] == "trace"
    assert result.extra["simplify"] is False
    assert result.extra["output_path_map"] == {"output": "output"}
    assert result.parity_max_abs_diff is None

    # 由 task_type + normalize 推断出的预处理规格
    assert isinstance(result.preprocess, PreprocessSpec)
    assert result.preprocess.task_type == "classification"
    assert result.preprocess.image_size == 32
    assert result.preprocess.mean == (0.1, 0.2, 0.3)
    assert result.preprocess.std == (0.4, 0.5, 0.6)
    assert result.preprocess.scale == pytest.approx(1.0 / 255.0)


@pytest.mark.unit
def test_onnx_export_preprocess_defaults_to_imagenet(tmp_path):
    """未显式给 normalize 时，按契约使用 ImageNet 均值/标准差与 1/255 缩放。"""
    exporter = ONNXExporter(
        SimpleCNN(num_classes=3),
        input_shape=(3, 32, 32),
        device="cpu",
        task_type="classification",
    )
    result = exporter.export(os.path.join(str(tmp_path), "default.onnx"))
    assert result.preprocess is not None
    assert result.preprocess.mean == (0.485, 0.456, 0.406)
    assert result.preprocess.std == (0.229, 0.224, 0.225)
    assert result.preprocess.scale == pytest.approx(1.0 / 255.0)
    assert result.preprocess.image_size == 32


@pytest.mark.unit
def test_onnx_export_explicit_preprocess_wins(tmp_path):
    """显式传入的 preprocess 优先于 task_type 推断。"""
    spec = PreprocessSpec(task_type="detection", image_size=64)
    exporter = ONNXExporter(
        SimpleCNN(num_classes=3),
        input_shape=(3, 32, 32),
        device="cpu",
        task_type="classification",
        preprocess=spec,
    )
    result = exporter.export(os.path.join(str(tmp_path), "explicit.onnx"))
    assert result.preprocess is spec


@pytest.mark.unit
def test_onnx_export_without_task_type_has_no_preprocess(tmp_path):
    """既未显式传 preprocess、也没有 task_type 时，preprocess 为 None。"""
    exporter = ONNXExporter(SimpleCNN(num_classes=3), input_shape=(3, 32, 32), device="cpu")
    result = exporter.export(os.path.join(str(tmp_path), "no_prep.onnx"))
    assert result.preprocess is None


@pytest.mark.skipif(not HAS_ONNX, reason="onnx 未安装")
def test_onnx_metadata_contains_export_result(tmp_path):
    """metadata_props 含 dvisionix_export，且原有 normalize_* / output_path_map 保留。"""
    output_path = os.path.join(str(tmp_path), "metadata.onnx")
    exporter = ONNXExporter(
        SimpleCNN(num_classes=3),
        input_shape=(3, 32, 32),
        device="cpu",
        task_type="classification",
        normalize={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "scale": 1.0 / 255.0,
        },
    )
    result = exporter.export(output_path)
    assert result.path == output_path

    model = onnx.load(output_path)
    props = {prop.key: prop.value for prop in model.metadata_props}
    assert "dvisionix_export" in props, sorted(props)
    payload = json.loads(props["dvisionix_export"])
    assert payload["path"] == output_path
    assert payload["format"] == "onnx"
    assert payload["input_shapes"] == [[1, 3, 32, 32]]
    assert payload["preprocess"]["task_type"] == "classification"
    assert payload["preprocess"]["image_size"] == 32
    assert json.loads(props["output_path_map"]) == {"output": "output"}
    assert "normalize_mean" in props and "normalize_std" in props


# ---------------------------------------------------------------------------
# ONNXExporter：verify 报告
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (HAS_ONNX and HAS_ORT), reason="需要 onnx 与 onnxruntime")
def test_onnx_verify_returns_report_and_backfills_result(tmp_path):
    """verify 返回 VerifyReport，逐输出明细正确，并回填 parity_max_abs_diff。"""
    exporter = ONNXExporter(
        SimpleCNN(num_classes=4),
        input_shape=(3, 32, 32),
        device="cpu",
        task_type="classification",
    )
    result = exporter.export(os.path.join(str(tmp_path), "verify.onnx"))
    report = exporter.verify(result, num_samples=2)

    assert isinstance(report, VerifyReport)
    assert report.passed is True
    assert bool(report) is True
    assert report.max_abs_diff is not None
    assert report.max_abs_diff < 1e-4
    assert len(report.outputs) == 1
    entry = report.outputs[0]
    assert entry["name"] == "output"
    assert entry["shape"] == [1, 4]
    assert entry["max_abs_diff"] == report.max_abs_diff
    assert entry["passed"] is True
    assert result.parity_max_abs_diff == report.max_abs_diff

    # 旧调用方式（直接传路径）仍返回可当布尔用的报告
    assert exporter.verify(result.path, num_samples=1)
