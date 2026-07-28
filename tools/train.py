# -*- coding: utf-8 -*-
"""配置驱动的统一训练入口。

用法::

    python tools/train.py --config configs/classification/demo_synthetic.yaml
    python tools/train.py --config xxx.yaml --cfg-options training.learning_rate=0.01

配置需包含 task_type / model / data / training 等字段，详见 CodePlan.md。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dvisionix.config import Config
from dvisionix.data import DatasetFactory, CustomDataset, TaskType
from dvisionix.data.transforms import (
    ClassificationTransforms,
    DetectionTransforms,
    SegmentationTransforms,
)
from dvisionix.models import build_model
from dvisionix.training import (
    Trainer,
    build_task,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoardLogger,
)
from dvisionix.utils import get_logger, set_seed


def build_transforms(task_type, image_size, train, mean=None, std=None):
    """根据任务类型构建默认变换。"""
    if task_type == "classification":
        return ClassificationTransforms(train=train, image_size=image_size)
    if task_type == "detection":
        return DetectionTransforms(train=train, image_size=image_size)
    if task_type == "segmentation":
        return SegmentationTransforms(train=train, image_size=image_size)
    raise ValueError(f"Unknown task_type: {task_type}")


def build_synthetic_dataset(task_type, num_samples, num_classes, image_size, transforms):
    """生成内存合成数据集（便于无网络环境快速验证）。"""
    import cv2

    tmp_dir = os.path.join(".cache", "synthetic", task_type)
    os.makedirs(tmp_dir, exist_ok=True)
    samples = []
    for i in range(num_samples):
        img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        path = os.path.join(tmp_dir, f"img_{i:04d}.png")
        if not os.path.exists(path):
            cv2.imwrite(path, img)
        if task_type == "classification":
            samples.append({"image_path": path, "label": i % num_classes})
        elif task_type == "detection":
            x1, y1 = np.random.randint(0, image_size // 2, 2)
            x2 = x1 + np.random.randint(10, image_size // 2)
            y2 = y1 + np.random.randint(10, image_size // 2)
            samples.append({
                "image_path": path,
                "boxes": [[float(x1), float(y1), float(x2), float(y2)]],
                "labels": [i % num_classes],
            })
        elif task_type == "segmentation":
            mask = (np.random.rand(image_size, image_size) * num_classes).astype(np.uint8)
            mask_path = os.path.join(tmp_dir, f"mask_{i:04d}.png")
            cv2.imwrite(mask_path, mask)
            samples.append({"image_path": path, "mask_path": mask_path})
    return CustomDataset(
        task_type=task_type,
        samples=samples,
        num_classes=num_classes,
        transforms=transforms,
    )


def build_data(cfg):
    """根据配置构建 (train_loader, val_loader)。"""
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
        train_ds = build_synthetic_dataset(task_type, n_train, num_classes, image_size, train_tf)
        val_ds = build_synthetic_dataset(task_type, n_val, num_classes, image_size, val_tf)
    else:
        root = data_cfg.get("root", "./data")
        train_ds = DatasetFactory.create(name=dataset_name, root=root, train=True,
                                         transforms=train_tf, download=True)
        val_ds = DatasetFactory.create(name=dataset_name, root=root, train=False,
                                       transforms=val_tf, download=True)

    batch_size = cfg.training.batch_size
    num_workers = cfg.training.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def build_callbacks(cfg):
    callbacks = []
    ckpt = cfg.get("checkpoint", {})
    callbacks.append(ModelCheckpoint(
        save_dir=ckpt.get("save_dir", "./checkpoints"),
        monitor=ckpt.get("monitor", "val_loss"),
        mode=ckpt.get("mode", "min"),
        save_best_only=ckpt.get("save_best_only", True),
    ))
    log_cfg = cfg.get("logging", {})
    if log_cfg.get("tensorboard", True):
        callbacks.append(TensorBoardLogger(log_dir=log_cfg.get("log_dir", "./logs")))
    es = cfg.training.get("early_stopping", {})
    if es and es.get("enabled", False):
        callbacks.append(EarlyStopping(
            monitor=es.get("monitor", "val_loss"),
            mode=es.get("mode", "min"),
            patience=es.get("patience", 5),
        ))
    return callbacks


def _default_task_cfg(cfg):
    """若配置未显式给出 task 段，按 task_type 推断默认任务类型。"""
    mapping = {
        "classification": "ClassificationTask",
        "detection": "DetectionTask",
        "segmentation": "SegmentationTask",
    }
    type_name = mapping.get(cfg.task_type)
    if type_name is None:
        raise ValueError(f"Cannot infer task for task_type={cfg.task_type}")
    return {
        "type": type_name,
        "num_classes": cfg.model.num_classes,
        "learning_rate": cfg.training.get("learning_rate", 1e-3),
        "weight_decay": cfg.training.get("weight_decay", 1e-4),
    }


def main():
    parser = argparse.ArgumentParser(description="DVisionix config-driven training")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--cfg-options", nargs="*", default=[],
                        help="override config, e.g. training.learning_rate=0.01")
    parser.add_argument("--resume", default=None,
                        help="path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.update_from_cli(args.cfg_options)
    cfg.validate(["task_type", "model.num_classes", "training.num_epochs"])

    seed = cfg.training.get("seed", 42)
    set_seed(seed)

    logger = get_logger("dvisionix.train", level="info",
                        log_dir=cfg.get("logging", {}).get("log_dir", "./logs"))
    logger.info(f"Config: {args.config}")
    model_label = cfg.model.get("type") or cfg.model.get("name")
    logger.info(f"Task: {cfg.task_type}, model: {model_label}")

    train_loader, val_loader = build_data(cfg)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(cfg.model.to_dict())
    logger.info(f"Model params: {model.count_parameters():,}")

    task = build_task(cfg.task.to_dict() if "task" in cfg else _default_task_cfg(cfg))

    trainer = Trainer(
        task=task,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=build_callbacks(cfg),
        device=cfg.training.get("device", "auto"),
        max_epochs=cfg.training.num_epochs,
        gradient_clip_val=cfg.training.get("gradient_clip_val", None),
        amp=cfg.training.get("amp", False),
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        seed=seed,
        resume_from=args.resume or cfg.training.get("resume_from"),
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
