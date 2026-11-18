# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 配置驱动的推理命令行入口（P4）。
"""配置驱动的推理命令行入口（P4）。

用法::

    python tools/infer.py --config configs/classification/demo_synthetic.yaml \
        --checkpoint best.pt --input imgs/ --output outputs/
    python tools/infer.py --config xxx.yaml --checkpoint best.pt --input a.jpg \
        --output out --task-type classification --device cpu --save-vis
    python tools/infer.py --config xxx.yaml --checkpoint best.pt --input imgs/ \
        --output out --cfg-options data.image_size=64

输入可以是单张图片，也可以是目录（递归收集 ``.jpg/.jpeg/.png/.bmp``，按路径排序）。
每张图片输出 ``<stem>.json``；``--save-vis`` 时额外输出 ``<stem>_vis.jpg`` 可视化。
预处理 / 前向 / 后处理与反变换全部复用 ``dvisionix.inference.predictors``，
本入口只负责装配与落盘。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from dvisionix.config import Config
from dvisionix.errors import ConfigError
from dvisionix.inference.predictors import build_predictor
from dvisionix.models import build_model

#: 支持推理的图片扩展名（小写，含点）
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args(argv=None):
    """解析命令行参数。

    Args:
        argv: 参数列表；为 None 时读取 ``sys.argv[1:]``（便于测试注入）。

    Returns:
        argparse.Namespace。
    """
    parser = argparse.ArgumentParser(description="DVisionix config-driven inference")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", required=True, help="model checkpoint path (.pt/.pth)")
    parser.add_argument("--input", required=True, help="image file or directory")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument(
        "--task-type",
        default=None,
        help="task type (default: config's task_type), e.g. classification",
    )
    parser.add_argument("--device", default="auto", help="inference device: auto / cpu / cuda")
    parser.add_argument(
        "--trusted",
        action="store_true",
        help="allow loading a full Trainer checkpoint (torch.load weights_only=False)",
    )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save <stem>.json per image (default: enabled)",
    )
    parser.add_argument("--save-vis", action="store_true", help="save <stem>_vis.jpg per image")
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=[],
        help="override config, e.g. data.image_size=64",
    )
    return parser.parse_args(argv)


def collect_images(input_path):
    """收集待推理图片路径。

    Args:
        input_path: 单张图片路径，或包含图片的目录（递归扫描）。

    Returns:
        图片路径列表；目录输入按路径排序以保证可复现。

    Raises:
        FileNotFoundError: 路径不存在，或目录中没有任何支持的图片。
    """
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        images = []
        for root, _dirs, files in os.walk(input_path):
            for name in files:
                if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS:
                    images.append(os.path.join(root, name))
        if not images:
            raise FileNotFoundError(
                f"输入目录中没有找到支持的图片（{', '.join(_IMAGE_EXTENSIONS)}）: {input_path}"
            )
        return sorted(images)

    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def build_predictor_from_checkpoint(
    model, task_type, checkpoint, *, trusted, device, image_size, class_names
):
    """构建任务预测器并载入 checkpoint 权重。

    先用 ``build_predictor`` 取得任务对应的具体预测器子类（分类/检测/分割），
    再通过其 ``from_checkpoint`` 载入权重 —— 预处理与后处理只依赖预测器实现，
    本入口不重复实现任何前后处理逻辑。

    Args:
        model: 已构建的模型（结构由 ``model.load_state_dict`` 校验）。
        task_type: 任务类型。
        checkpoint: checkpoint 路径。
        trusted: 是否信任 checkpoint（完整 Trainer checkpoint 需要 True）。
        device: 推理设备。
        image_size: 预处理输入边长。
        class_names: 类别名列表；无则为 None。

    Returns:
        对应的 ``BasePredictor`` 子类实例。
    """
    kwargs = {
        "device": device,
        "image_size": image_size,
        "class_names": class_names,
    }
    predictor = build_predictor(model, task_type, **kwargs)
    return type(predictor).from_checkpoint(
        model,
        checkpoint,
        task_type,
        trusted=trusted,
        **kwargs,
    )


def main(argv=None):
    """推理入口：解析参数 -> 构建模型/预测器 -> 逐图推理 -> JSON/可视化落盘。

    Args:
        argv: 参数列表；为 None 时读取 ``sys.argv[1:]``。

    Raises:
        ConfigError: 配置缺少 ``task_type`` 或 ``model``。
        FileNotFoundError: checkpoint / 输入路径不存在。
    """
    args = parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.cfg_options:
        cfg.update_from_cli(args.cfg_options)

    task_type = args.task_type or cfg.get("task_type")
    if not task_type:
        raise ConfigError(
            "配置缺少 task_type，且命令行未提供 --task-type",
            hint="在 YAML 中设置 task_type: classification/detection/segmentation，"
            "或使用 --task-type 指定",
        )
    if "model" not in cfg:
        raise ConfigError(
            "配置缺少 model 段",
            hint="在 YAML 中提供 model: {type: ..., num_classes: ...}，"
            "例如 model: {type: simple_cnn, num_classes: 3}",
        )
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    data_cfg = cfg.get("data") or {}
    image_size = data_cfg.get("image_size", 224)
    class_names = data_cfg.get("classes")

    model = build_model(cfg.model.to_dict())
    predictor = build_predictor_from_checkpoint(
        model,
        task_type,
        args.checkpoint,
        trusted=args.trusted,
        device=args.device,
        image_size=image_size,
        class_names=class_names,
    )

    inputs = collect_images(args.input)
    os.makedirs(args.output, exist_ok=True)

    start = time.perf_counter()
    for index, image_path in enumerate(inputs, start=1):
        results = predictor.predict(image_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]

        if args.save_json:
            results.save(os.path.join(args.output, f"{stem}.json"))
        if args.save_vis:
            vis = results.visualize(
                image=cv2.imread(image_path),
                class_names=class_names,
            )
            cv2.imwrite(os.path.join(args.output, f"{stem}_vis.jpg"), vis)
        print(f"[{index}/{len(inputs)}] {image_path}")

    elapsed = time.perf_counter() - start
    print(
        f"完成：任务 {task_type}，共处理 {len(inputs)} 张图片，"
        f"耗时 {elapsed:.3f}s，输出目录 {os.path.abspath(args.output)}"
    )


if __name__ == "__main__":
    main()
