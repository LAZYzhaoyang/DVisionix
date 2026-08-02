# -*- coding: utf-8 -*-
"""配置驱动的统一训练入口（v0.3.0）。

用法::

    python tools/train.py --config configs/classification/demo_synthetic.yaml
    python tools/train.py --config xxx.yaml --cfg-options training.learning_rate=0.01
    python tools/train.py --config xxx.yaml --resume auto            # 续最近一次任务
    python tools/train.py --config xxx.yaml --work-dir /path/to/run  # 指定工作目录
    torchrun --nproc_per_node=2 tools/train.py --config xxx.yaml --devices 0,1

配置需包含 task_type / model / data / training 等字段，详见 CodePlan.md。
训练产物（日志 / TensorBoard / 检查点 / 最终配置）全部落在 work_dir（默认在代码库外）。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader

from dvisionix.config import Config
from dvisionix.data import CustomDataset, build_dataset
from dvisionix.data.transforms import (
    ClassificationTransforms,
    DetectionTransforms,
    SegmentationTransforms,
    SimCLRTransforms,
)
from dvisionix.models import build_model
from dvisionix.training import (
    build_task,
    build_trainer,
    dump_config,
    find_checkpoint,
    load_backbone,
    resolve_work_dir,
)
from dvisionix.utils import get_logger, set_seed

_TASK_TYPES = ("classification", "detection", "segmentation", "simclr")
_TASK_MAPPING = {
    "classification": "ClassificationTask",
    "detection": "DetectionTask",
    "segmentation": "SegmentationTask",
    "simclr": "SimCLRTask",
}


def build_transforms(task_type, image_size, train):
    if task_type == "classification":
        return ClassificationTransforms(train=train, image_size=image_size)
    if task_type == "detection":
        return DetectionTransforms(train=train, image_size=image_size)
    if task_type == "segmentation":
        return SegmentationTransforms(train=train, image_size=image_size)
    if task_type == "simclr":
        return SimCLRTransforms(train=train, image_size=image_size)
    raise ValueError(f"Unknown task_type: {task_type}")


def build_synthetic_dataset(
    task_type, num_samples, num_classes, image_size, transforms, cache_dir=None
):
    """生成内存合成数据集（便于无网络环境快速验证）。"""
    import cv2

    tmp_dir = cache_dir or os.path.join(".cache", "synthetic", task_type)
    os.makedirs(tmp_dir, exist_ok=True)
    samples = []
    for i in range(num_samples):
        img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        path = os.path.join(tmp_dir, f"img_{i:04d}.png")
        if not os.path.exists(path):
            cv2.imwrite(path, img)
        if task_type == "classification":
            samples.append({"image": path, "label": i % num_classes})
        elif task_type == "detection":
            x1, y1 = np.random.randint(0, image_size // 2, 2)
            x2 = x1 + np.random.randint(10, image_size // 2)
            y2 = y1 + np.random.randint(10, image_size // 2)
            samples.append(
                {
                    "image": path,
                    "boxes": np.array(
                        [[float(x1), float(y1), float(x2), float(y2)]], dtype=np.float32
                    ),
                    "labels": np.array([i % num_classes], dtype=np.int64),
                }
            )
        elif task_type == "segmentation":
            mask = (np.random.rand(image_size, image_size) * num_classes).astype(np.uint8)
            mask_path = os.path.join(tmp_dir, f"mask_{i:04d}.png")
            cv2.imwrite(mask_path, mask)
            samples.append({"image": path, "mask": mask_path})
    return CustomDataset(samples=samples, task_type=task_type, transforms=transforms)


def build_data(cfg, work_dir=None):
    """根据配置构建 (train_loader, val_loader)。

    Args:
        work_dir: 工作目录；合成数据缓存写到 work_dir/.cache/synthetic（隔离代码库）。
    """
    task_type = cfg.task_type
    image_size = cfg.data.image_size
    num_classes = cfg.model.num_classes

    train_tf = build_transforms(task_type, image_size, train=True)
    val_tf = build_transforms(task_type, image_size, train=False)

    data_cfg = cfg.data
    dataset_name = data_cfg.get("dataset", "custom")
    if dataset_name in ("custom", "synthetic") or data_cfg.get("synthetic", False):
        n_train = data_cfg.get("num_samples", 64)
        n_val = data_cfg.get("val_samples", 16)
        cache_dir = os.path.join(work_dir, ".cache", "synthetic", task_type) if work_dir else None
        train_ds = build_synthetic_dataset(
            task_type, n_train, num_classes, image_size, train_tf, cache_dir
        )
        val_ds = build_synthetic_dataset(
            task_type, n_val, num_classes, image_size, val_tf, cache_dir
        )
    else:
        root = data_cfg.get("root", "./data")
        train_kwargs = dict(
            root=root, train=True, transforms=train_tf, download=data_cfg.get("download", False)
        )
        val_kwargs = dict(
            root=root, train=False, transforms=val_tf, download=data_cfg.get("download", False)
        )
        for extra_key in ("year", "image_set", "split"):
            if extra_key in data_cfg:
                train_kwargs[extra_key] = data_cfg[extra_key]
                val_kwargs[extra_key] = data_cfg[extra_key]
        train_ds = build_dataset({"type": dataset_name, **train_kwargs})
        val_ds = build_dataset({"type": dataset_name, **val_kwargs})

    batch_size = cfg.training.batch_size
    num_workers = cfg.training.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=getattr(train_ds, "collate_fn", None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=getattr(val_ds, "collate_fn", None),
    )
    return train_loader, val_loader


def build_task_cfg(cfg):
    """从配置组装 Task 构建参数（optimizer/scheduler/loss/metrics 全部配置驱动）。"""
    task_cfg = dict(cfg.get("task", {}) or {})
    if "type" not in task_cfg:
        task_cfg["type"] = _TASK_MAPPING[cfg.task_type]
    task_cfg.setdefault("num_classes", cfg.model.num_classes)

    training = cfg.get("training", {}) or {}
    optimizer_cfg = dict(training.get("optimizer", {}) or {})
    optimizer_cfg.setdefault("lr", training.get("learning_rate", 1e-3))
    optimizer_cfg.setdefault("weight_decay", training.get("weight_decay", 1e-4))
    task_cfg.setdefault("optimizer_cfg", optimizer_cfg)

    scheduler_cfg = dict(training.get("scheduler", {}) or {})
    task_cfg.setdefault("scheduler_cfg", scheduler_cfg)

    loss = cfg.get("loss")
    if loss is not None:
        loss = loss.to_dict() if isinstance(loss, Config) else dict(loss)
        if cfg.task_type == "detection" and "num_classes" not in loss:
            loss["num_classes"] = cfg.model.num_classes
        task_cfg.setdefault("loss", loss)

    metrics = cfg.get("metrics")
    if metrics is not None:
        metrics = metrics.to_dict() if isinstance(metrics, Config) else dict(metrics)
        task_cfg.setdefault("metrics", metrics)

    return task_cfg


def parse_devices(value):
    """'0,1' -> [0, 1]。"""
    if value is None:
        return None
    return [int(x) for x in str(value).split(",") if x.strip() != ""]


def export_best_onnx(cfg, work_dir: str, logger=None):
    """把最优 checkpoint（work_dir/checkpoints/best.pt）导出为 ONNX（best-effort）。

    仅当 ``training.export_best_onnx`` 为 true 时调用；模型不可 trace 时告警降级，
    不影响训练主流程。
    """
    best_pt = os.path.join(work_dir, "checkpoints", "best.pt")
    if not os.path.exists(best_pt):
        if logger:
            logger.warning(f"export_best_onnx 已开启但未找到 {best_pt}")
        return None
    try:
        import torch

        from dvisionix.export import ONNXExporter

        model = build_model(cfg.model.to_dict())
        state = torch.load(best_pt, map_location="cpu", weights_only=False)
        if state.get("model_state_dict"):
            model.load_state_dict(state["model_state_dict"])
        in_channels = cfg.model.get("in_channels", 3)
        image_size = cfg.data.get("image_size", 32)
        exporter = ONNXExporter(
            model,
            input_shape=(in_channels, image_size, image_size),
            device="cpu",
            task_type=cfg.get("task_type"),
        )
        out_path = os.path.join(work_dir, "best.onnx")
        exporter.export(out_path, dynamic_batch=True)
        if logger:
            logger.info(f"Best model exported to ONNX: {out_path}")
        return out_path
    except Exception as exc:  # pragma: no cover - 依赖具体模型可 trace 性
        if logger:
            logger.warning(f"导出 best.onnx 失败（已忽略）：{exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="DVisionix config-driven training")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=[],
        help="override config, e.g. training.learning_rate=0.01",
    )
    parser.add_argument(
        "--resume", default=None, help="resume mode: auto / latest / <checkpoint path>"
    )
    parser.add_argument(
        "--work-dir", default=None, help="explicit work dir (default: ~/dvisionix_runs/<exp>/<ts>)"
    )
    parser.add_argument("--devices", default=None, help="devices for DDP, e.g. '0,1'")
    parser.add_argument("--strategy", default=None, help="auto / ddp / none")
    parser.add_argument("--force", action="store_true", help="force fresh run (ignore resume)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.update_from_cli(args.cfg_options)
    cfg.validate(["task_type", "model.num_classes", "training.num_epochs"])
    schema_warnings = cfg.validate_schema(cfg.task_type)

    resume = None if args.force else (args.resume or cfg.get("resume", False))
    work_dir = resolve_work_dir(cfg, cli_work_dir=args.work_dir, resume=resume)
    resume_path = find_checkpoint(work_dir, resume)
    os.makedirs(work_dir, exist_ok=True)
    dump_config(cfg, os.path.join(work_dir, "config.resolved.yaml"))

    seed = cfg.training.get("seed", 42)
    set_seed(seed)
    logger = get_logger("dvisionix.train", level="info", log_dir=os.path.join(work_dir, "logs"))
    logger.info(f"Config: {args.config}")
    logger.info(f"Task: {cfg.task_type}, model: {cfg.model.get('type') or cfg.model.get('name')}")
    logger.info(f"Work dir: {work_dir}")
    if resume_path:
        logger.info(f"Resume from: {resume_path}")
    for w in schema_warnings:
        logger.warning(w)

    train_loader, val_loader = build_data(cfg, work_dir=work_dir)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(cfg.model.to_dict())
    logger.info(f"Model params: {model.count_parameters():,}")

    pretrained_backbone = cfg.model.get("pretrained_backbone") or cfg.training.get(
        "pretrained_backbone"
    )
    if pretrained_backbone:
        load_backbone(model, pretrained_backbone)
        logger.info(f"Backbone pretrained weights loaded from: {pretrained_backbone}")

    task = build_task(build_task_cfg(cfg))

    trainer = build_trainer(
        cfg,
        task,
        train_loader,
        val_loader,
        work_dir=work_dir,
        resume_from=resume_path,
        devices=parse_devices(args.devices),
        strategy=args.strategy,
    )
    trainer.fit(model)

    if cfg.training.get("export_best_onnx", False):
        export_best_onnx(cfg, work_dir, logger=logger)


if __name__ == "__main__":
    main()
