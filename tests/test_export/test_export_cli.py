# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: tools/export.py 命令行入口测试（P5-B）。
"""``tools/export.py`` 命令行入口测试。

用 toy ``SimpleCNN`` + 纯 state_dict checkpoint + 最小配置，在 tmp 目录中端到端跑
``main()``：校验 ``.onnx`` 与 ``.onnx.json`` sidecar、``--verify`` 一致性校验、
``--dynamic-batch`` 的动态维度，以及完整 Trainer checkpoint 未加 ``--trusted`` 时
必须抛 ``CheckpointError``（加了则正常导出）。
"""

import json
import os

import numpy as np
import pytest
import torch

from dvisionix.errors import CheckpointError
from dvisionix.models import SimpleCNN
from tools.export import main

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

#: 测试用类别数（与最小配置一致）
_NUM_CLASSES = 3
#: 测试用输入边长（SimpleCNN 三次下采样后期望 4x4 特征图）
_IMAGE_SIZE = 32
#: 默认 opset（与 CLI 一致）
_DEFAULT_OPSET = 17

#: 最小导出配置：只保留模型结构与输入尺寸，不依赖仓库内任何数据
_CONFIG_TEXT = """\
task_type: classification
model:
  type: simple_cnn
  num_classes: 3
data:
  image_size: 32
"""


def _write_config(tmp_path):
    """写出最小导出配置，返回配置路径。"""
    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG_TEXT, encoding="utf-8")
    return str(config)


def _write_state_dict_checkpoint(tmp_path, model=None):
    """保存纯 state_dict checkpoint（默认与配置同构的 SimpleCNN）。"""
    checkpoint = tmp_path / "model.pt"
    torch.save((model or SimpleCNN(num_classes=_NUM_CLASSES)).state_dict(), checkpoint)
    return str(checkpoint)


def _write_wrapped_state_dict_checkpoint(tmp_path):
    """保存带 DDP ``module.`` 前缀的纯 state_dict（验证前缀剥离）。"""
    state = SimpleCNN(num_classes=_NUM_CLASSES).state_dict()
    checkpoint = tmp_path / "wrapped.pt"
    torch.save({f"module.{key}": value for key, value in state.items()}, checkpoint)
    return str(checkpoint)


def _write_trainer_checkpoint(tmp_path):
    """模拟完整 Trainer checkpoint：含 numpy RNG 等非张量状态。

    ``weights_only=True`` 无法反序列化 numpy RNG 状态，因此未加 ``--trusted``
    时必须在加载阶段失败（与真实 Trainer 产物的行为一致）。
    """
    checkpoint = tmp_path / "trainer.pt"
    torch.save(
        {
            "model_state_dict": SimpleCNN(num_classes=_NUM_CLASSES).state_dict(),
            "optimizer_state_dict": {},
            "epoch": 3,
            "global_step": 42,
            "numpy_rng_state": np.random.get_state(),
        },
        checkpoint,
    )
    return str(checkpoint)


def _export_args(config, checkpoint, output, *extra_args):
    """拼装 CLI 参数（main 接受 argv 列表，无需 monkeypatch sys.argv）。"""
    return [
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--output",
        output,
        "--device",
        "cpu",
        *extra_args,
    ]


def _load_sidecar(output):
    """读取 ``<output>.json`` sidecar 并返回解析后的字典。"""
    with open(f"{output}.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.unit
def test_export_cli_writes_onnx_and_sidecar(tmp_path):
    """默认参数：生成 .onnx 与 .onnx.json，sidecar 记录输入/输出/preprocess。"""
    config = _write_config(tmp_path)
    checkpoint = _write_state_dict_checkpoint(tmp_path)
    # 输出目录不存在：应由导出器自动创建
    output = str(tmp_path / "exports" / "model.onnx")

    main(_export_args(config, checkpoint, output))

    assert os.path.getsize(output) > 0
    payload = _load_sidecar(output)
    assert payload["path"] == output
    assert payload["format"] == "onnx"
    assert payload["opset"] == _DEFAULT_OPSET
    assert payload["input_names"] == ["input"]
    assert payload["input_shapes"] == [[1, 3, _IMAGE_SIZE, _IMAGE_SIZE]]
    assert payload["output_names"] == ["output"]

    preprocess = payload["preprocess"]
    assert preprocess["task_type"] == "classification"
    assert preprocess["image_size"] == _IMAGE_SIZE
    assert preprocess["mean"] == [0.485, 0.456, 0.406]
    assert preprocess["std"] == [0.229, 0.224, 0.225]
    assert preprocess["scale"] == pytest.approx(1.0 / 255.0)

    if HAS_ONNX:
        model = onnx.load(output)
        dims = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        assert dims == [1, 3, _IMAGE_SIZE, _IMAGE_SIZE]


@pytest.mark.unit
@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime 未安装")
def test_export_cli_verify_passes(tmp_path, capsys):
    """--verify：ORT 与 PyTorch 输出一致，打印全局 max_abs_diff 且不退出。"""
    config = _write_config(tmp_path)
    checkpoint = _write_state_dict_checkpoint(tmp_path)
    output = str(tmp_path / "model.onnx")

    main(_export_args(config, checkpoint, output, "--verify"))

    stdout = capsys.readouterr().out
    assert "一致性校验: 通过" in stdout
    assert "max_abs_diff" in stdout


@pytest.mark.unit
@pytest.mark.skipif(not HAS_ONNX, reason="onnx 未安装")
def test_export_cli_dynamic_batch_sets_dynamic_axis(tmp_path):
    """--dynamic-batch：ONNX 输入第 0 维为动态 batch_size。"""
    config = _write_config(tmp_path)
    checkpoint = _write_state_dict_checkpoint(tmp_path)
    output = str(tmp_path / "model.onnx")

    main(_export_args(config, checkpoint, output, "--dynamic-batch"))

    model = onnx.load(output)
    batch_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch_dim.dim_param == "batch_size"


@pytest.mark.unit
def test_export_cli_opset_is_applied(tmp_path):
    """--opset：导出与 sidecar 都记录显式指定的 opset。"""
    config = _write_config(tmp_path)
    checkpoint = _write_state_dict_checkpoint(tmp_path)
    output = str(tmp_path / "model.onnx")

    main(_export_args(config, checkpoint, output, "--opset", "18"))

    assert _load_sidecar(output)["opset"] == 18


@pytest.mark.unit
def test_export_cli_cfg_options_are_applied(tmp_path):
    """--cfg-options：命令行覆盖配置后仍能正常导出。"""
    config = _write_config(tmp_path)
    checkpoint = _write_state_dict_checkpoint(tmp_path)
    output = str(tmp_path / "model.onnx")

    main(
        _export_args(
            config,
            checkpoint,
            output,
            "--cfg-options",
            "model.num_classes=3",
            "data.image_size=32",
        )
    )

    assert _load_sidecar(output)["format"] == "onnx"


@pytest.mark.unit
def test_export_cli_strips_wrapper_prefixes(tmp_path):
    """带 ``module.`` 前缀的 state_dict 应被剥离后 strict 载入成功。"""
    config = _write_config(tmp_path)
    checkpoint = _write_wrapped_state_dict_checkpoint(tmp_path)
    output = str(tmp_path / "model.onnx")

    main(_export_args(config, checkpoint, output))

    assert os.path.getsize(output) > 0


@pytest.mark.unit
def test_export_cli_full_trainer_checkpoint_requires_trusted(tmp_path):
    """完整 Trainer checkpoint（含 numpy RNG）：不带 --trusted 抛 CheckpointError。"""
    config = _write_config(tmp_path)
    checkpoint = _write_trainer_checkpoint(tmp_path)
    output = str(tmp_path / "model.onnx")

    with pytest.raises(CheckpointError, match="--trusted"):
        main(_export_args(config, checkpoint, output))

    # 加载阶段即失败，不应留下半成品
    assert not os.path.exists(output)


@pytest.mark.unit
def test_export_cli_trusted_trainer_checkpoint_exports(tmp_path):
    """--trusted：完整 Trainer checkpoint 取出 model_state_dict 并正常导出。"""
    config = _write_config(tmp_path)
    checkpoint = _write_trainer_checkpoint(tmp_path)
    output = str(tmp_path / "model.onnx")

    main(_export_args(config, checkpoint, output, "--trusted"))

    assert os.path.getsize(output) > 0
    assert _load_sidecar(output)["input_shapes"] == [[1, 3, _IMAGE_SIZE, _IMAGE_SIZE]]


@pytest.mark.unit
def test_export_cli_state_dict_mismatch_raises_checkpoint_error(tmp_path):
    """checkpoint 与配置的模型结构不一致：同样包装为 CheckpointError。"""
    config = _write_config(tmp_path)
    checkpoint = _write_state_dict_checkpoint(
        tmp_path, model=SimpleCNN(num_classes=_NUM_CLASSES + 2)
    )
    output = str(tmp_path / "model.onnx")

    with pytest.raises(CheckpointError):
        main(_export_args(config, checkpoint, output))
