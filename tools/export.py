# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 配置驱动的模型导出命令行入口（P5）。
"""配置驱动的模型导出命令行入口（P5）。

用法::

    python tools/export.py --config configs/classification/demo_synthetic.yaml \\
        --checkpoint best.pt --output model.onnx
    python tools/export.py --config xxx.yaml --checkpoint best.pt --output model.onnx \\
        --dynamic-batch --verify
    python tools/export.py --config xxx.yaml --checkpoint best.pt --output model.onnx \\
        --format onnx --opset 17 --device cpu --cfg-options data.image_size=32

导出流程：读配置 -> 构建模型 -> 载入 checkpoint -> 按 ``--format`` 取导出器 -> 导出并
落盘 ``<output>.json`` sidecar；``--verify`` 时再用 onnxruntime 与 PyTorch 逐输出比对，
未通过以退出码 1 结束。

``--trusted`` 与推理命令行语义一致：默认 ``torch.load(weights_only=True)``，只允许
张量与基本容器；完整 Trainer checkpoint（含优化器 / RNG 等非张量状态）必须显式加
``--trusted`` 才能反序列化。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dvisionix.config import Config
from dvisionix.errors import CheckpointError, ConfigError
from dvisionix.export import build_exporter
from dvisionix.models import build_model
from dvisionix.training.checkpoint import strip_wrapper_prefixes

#: ImageNet 归一化均值（RGB，与训练/推理预设一致）
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
#: ImageNet 归一化标准差（RGB）
_IMAGENET_STD = [0.229, 0.224, 0.225]
#: 默认导出的 ONNX opset 版本
_DEFAULT_OPSET = 17
#: 配置缺少 ``data.image_size`` 时的默认输入边长（与 toy SimpleCNN 对齐）
_DEFAULT_IMAGE_SIZE = 32


def parse_args(argv=None):
    """解析命令行参数。

    Args:
        argv: 参数列表；为 None 时读取 ``sys.argv[1:]``（便于测试注入）。

    Returns:
        argparse.Namespace。
    """
    parser = argparse.ArgumentParser(description="DVisionix config-driven model export")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", required=True, help="model checkpoint path (.pt/.pth)")
    parser.add_argument("--output", required=True, help="output model path, e.g. model.onnx")
    parser.add_argument("--format", default="onnx", help="export format (default: onnx)")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="export dynamic batch dimension (default: static batch)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=_DEFAULT_OPSET,
        help=f"ONNX opset version (default: {_DEFAULT_OPSET})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify ONNX outputs against PyTorch (exit code 1 on mismatch)",
    )
    parser.add_argument(
        "--trusted",
        action="store_true",
        help="allow loading a full Trainer checkpoint (torch.load weights_only=False)",
    )
    parser.add_argument("--device", default="cpu", help="export device: cpu / cuda")
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=[],
        help="override config, e.g. data.image_size=32",
    )
    return parser.parse_args(argv)


def load_checkpoint(model, checkpoint, *, trusted=False):
    """把 checkpoint 中的模型权重载入 ``model``。

    checkpoint 兼容两种格式：纯 state_dict，或含 ``model_state_dict`` 的完整
    Trainer checkpoint（后者携带非张量状态，只有 ``trusted=True`` 才能反序列化）。
    键上的 DDP / torch.compile 包装前缀会先被 ``strip_wrapper_prefixes`` 剥离。

    Args:
        model: 已按配置构建的模型（``load_state_dict(strict=True)`` 校验结构）。
        checkpoint: checkpoint 路径。
        trusted: **安全开关**。默认 ``False``，即 ``torch.load(weights_only=True)``；
            仅当你确认文件来源可信、且它确实携带非张量状态时才置为 ``True``。

    Returns:
        已剥离包装前缀的 state_dict（便于调用方回溯实际加载的键）。

    Raises:
        CheckpointError: 文件无法反序列化、内容不是 state_dict，或与模型结构不匹配。
    """
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=not trusted)
    except Exception as exc:
        if trusted:
            raise CheckpointError(
                f"无法加载 checkpoint: {checkpoint}（{type(exc).__name__}: {exc}）",
                hint="确认文件是 torch.save 产物且未损坏；只需权重时建议保存纯 state_dict。",
            ) from exc
        raise CheckpointError(
            f"以安全模式（weights_only=True）加载 checkpoint 失败: {checkpoint}"
            f"（{type(exc).__name__}: {exc}）",
            hint="该文件可能携带非张量状态（完整 Trainer checkpoint 含优化器 / RNG "
            "状态）。确认文件来源可信后请加 --trusted 重新导出。",
        ) from exc

    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise CheckpointError(
            f"checkpoint 内容不是 state_dict: {checkpoint}（实际类型 {type(state).__name__}）",
            hint="checkpoint 应为纯 state_dict，或含 model_state_dict 字段的完整 Trainer "
            "checkpoint（后者需 --trusted）。",
        )

    try:
        state = strip_wrapper_prefixes(state)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise CheckpointError(
            f"checkpoint 与模型结构不匹配: {checkpoint}（{type(exc).__name__}: {exc}）",
            hint="确认 --config 的 model 段与训练时一致（层名 / 类别数 / in_channels）；"
            "若该文件是完整 Trainer checkpoint 且报错来自安全模式，请加 --trusted。",
        ) from exc
    return state


def build_input_spec(cfg):
    """从配置提取导出所需的输入形状与归一化元数据。

    Args:
        cfg: 完整的 :class:`~dvisionix.config.Config`。

    Returns:
        ``(input_shape, normalize)``。``input_shape`` 为不含 batch 的 ``(C, H, W)``；
        ``normalize`` 为 ``{"mean", "std", "scale"}`` 形式的归一化元数据。
    """
    model_cfg = cfg.get("model") or {}
    data_cfg = cfg.get("data") or {}
    in_channels = int(model_cfg.get("in_channels", 3))
    image_size = int(data_cfg.get("image_size", _DEFAULT_IMAGE_SIZE))
    normalize = {
        "mean": list(data_cfg.get("mean", _IMAGENET_MEAN)),
        "std": list(data_cfg.get("std", _IMAGENET_STD)),
        "scale": 1.0 / 255.0,
    }
    return (in_channels, image_size, image_size), normalize


def print_summary(result, sidecar_path, dynamic_batch):
    """打印导出摘要：路径 / 输入输出名 / 预处理规格。

    Args:
        result: 导出器返回的 :class:`~dvisionix.export.ExportResult`。
        sidecar_path: ``result.save`` 落盘的 JSON sidecar 路径。
        dynamic_batch: 是否导出了动态 batch 维度。
    """
    print(f"导出完成: {result.path}")
    print(f"元数据: {sidecar_path}")
    print(f"格式: {result.format}（opset={result.opset}，动态 batch={dynamic_batch}）")
    print(f"输入: {result.input_names} 形状: {result.input_shapes}")
    print(f"输出: {result.output_names}")
    preprocess = result.preprocess
    if preprocess is None:
        print("预处理: 无（配置未提供 task_type / normalize）")
    else:
        print(f"预处理: {json.dumps(preprocess.to_dict(), ensure_ascii=False)}")


def run_verify(exporter, result):
    """执行 ONNX Runtime 一致性校验并打印逐输出明细。

    Args:
        exporter: 导出器实例。
        result: 导出结果（``ExportResult`` 或路径，由导出器自行接受）。

    Returns:
        是否全部输出在所有样本上通过。
    """
    report = exporter.verify(result)
    max_diff = report.max_abs_diff
    max_diff_text = "n/a" if max_diff is None else f"{max_diff:.3e}"
    status = "通过" if report.passed else "未通过"
    print(f"一致性校验: {status}，全局 max_abs_diff={max_diff_text}")
    for item in report.outputs:
        item_diff = item.get("max_abs_diff")
        item_diff_text = "n/a" if item_diff is None else f"{item_diff:.3e}"
        print(
            f"  - {item.get('name')}: shape={item.get('shape')} "
            f"max_abs_diff={item_diff_text} passed={item.get('passed')}"
        )
    return report.passed


def main(argv=None):
    """导出入口：解析参数 -> 构建模型/载入权重 -> 导出 -> 摘要 -> 可选校验。

    Args:
        argv: 参数列表；为 None 时读取 ``sys.argv[1:]``。

    Raises:
        ConfigError: 配置缺少 ``model`` 段，或 ``--format`` 未注册。
        CheckpointError: checkpoint 无法加载或与模型结构不匹配。
    """
    args = parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.cfg_options:
        cfg.update_from_cli(args.cfg_options)

    if "model" not in cfg:
        raise ConfigError(
            "配置缺少 model 段",
            hint="在 YAML 中提供 model: {type: ..., num_classes: ...}，"
            "例如 model: {type: simple_cnn, num_classes: 3}",
        )

    model = build_model(cfg.model.to_dict())
    load_checkpoint(model, args.checkpoint, trusted=args.trusted)

    input_shape, normalize = build_input_spec(cfg)
    exporter = build_exporter(
        args.format,
        model,
        input_shape=input_shape,
        device=args.device,
        task_type=cfg.get("task_type"),
        normalize=normalize,
    )
    result = exporter.export(
        args.output,
        dynamic_batch=args.dynamic_batch,
        opset_version=args.opset,
    )

    sidecar_path = f"{args.output}.json"
    verified = True
    if args.verify:
        verified = run_verify(exporter, result)
    # sidecar 必须在校验之后落盘：``verify`` 会把 ``parity_max_abs_diff`` 回填到
    # ``ExportResult``，先保存会得到 parity=null 的元数据（与导出产物不一致）。
    result.save(sidecar_path)
    print_summary(result, sidecar_path, args.dynamic_batch)
    if not verified:
        sys.exit(1)


if __name__ == "__main__":
    main()
