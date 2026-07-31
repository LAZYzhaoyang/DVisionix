# -*- coding: utf-8 -*-
# NOTE: v0.2.0 起推荐使用 config 驱动入口:
#   python tools/train.py --config configs/<task>/demo_synthetic.yaml
# 该脚本作为教学参考保留，功能上等价的现代用法请参考 tools/train.py + configs/。
"""
Config 驱动的端到端训练 Demo

演示如何仅通过一个 YAML 配置文件驱动整个训练流程：
    配置加载 -> 构建数据集/模型/任务/回调 -> 训练。

为便于快速验证，这里使用内存中的合成数据集（无需下载）。
把 model.name / data.dataset 换成真实值即可用于真实任务。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dvisionix.config import Config
from dvisionix.data import CustomDataset, TaskType
from dvisionix.data.transforms import ClassificationTransforms
from dvisionix.models import SimpleCNN, TimmClassifier
from dvisionix.training import (
    Trainer,
    ClassificationTask,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoardLogger,
)
from dvisionix.utils import get_logger, log_metrics


def build_synthetic_dataset(tmp_dir, num_samples, num_classes, image_size, transforms):
    """生成一批随机图片并构造 CustomDataset。"""
    import cv2

    os.makedirs(tmp_dir, exist_ok=True)
    samples = []
    for i in range(num_samples):
        img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        path = os.path.join(tmp_dir, f"img_{i:04d}.png")
        if not os.path.exists(path):
            cv2.imwrite(path, img)
        samples.append({"image": path, "label": i % num_classes})
    return CustomDataset(
        task_type="classification",
        samples=samples,
        num_classes=num_classes,
        transforms=transforms,
    )


def build_model(cfg: Config):
    """根据配置构建模型（simple_cnn 或 timm 模型名）。"""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, in_channels=cfg.model.get("in_channels", 3))
    # 其它名称一律走 timm
    return TimmClassifier(name=name, num_classes=num_classes, pretrained=False)


def main(config_path: str) -> None:
    cfg = Config.from_yaml(config_path)
    cfg.validate(["task_type", "model.name", "model.num_classes", "training.num_epochs"])

    logger = get_logger(
        "dvisionix.demo",
        level="info",
        log_dir=cfg.logging.get("log_dir", "./logs"),
    )
    logger.info(f"Loaded config: {config_path}")
    logger.info(f"Experiment: {cfg.get('experiment_name', 'unnamed')}")

    image_size = cfg.data.image_size
    num_classes = cfg.model.num_classes

    train_tf = ClassificationTransforms(train=True, image_size=image_size)
    val_tf = ClassificationTransforms(train=False, image_size=image_size)

    data_cache = os.path.join(".cache", "synthetic")
    train_ds = build_synthetic_dataset(
        os.path.join(data_cache, "train"),
        cfg.data.get("num_samples", 128),
        num_classes, image_size, train_tf,
    )
    val_ds = build_synthetic_dataset(
        os.path.join(data_cache, "val"),
        cfg.data.get("val_samples", 32),
        num_classes, image_size, val_tf,
    )
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    batch_size = cfg.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg)
    logger.info(f"Model: {cfg.model.name}, params: {model.count_parameters():,}")

    task = ClassificationTask(
        num_classes=num_classes,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.get("weight_decay", 1e-4),
    )

    callbacks = [
        ModelCheckpoint(
            save_dir=cfg.checkpoint.get("save_dir", "./checkpoints"),
            monitor=cfg.checkpoint.get("monitor", "val_loss"),
            mode=cfg.checkpoint.get("mode", "min"),
        ),
        TensorBoardLogger(log_dir=cfg.logging.get("log_dir", "./logs")),
    ]
    es = cfg.training.get("early_stopping", {})
    if es and es.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=es.get("monitor", "val_loss"),
                mode=es.get("mode", "min"),
                patience=es.get("patience", 5),
            )
        )

    trainer = Trainer(
        task=task,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks,
        device=cfg.training.device,
        max_epochs=cfg.training.num_epochs,
    )

    logger.info("Start training ...")
    history = trainer.fit(model)

    # 记录最终指标
    final = {}
    for k, v in history.items():
        if isinstance(v, list) and v and isinstance(v[-1], (int, float)):
            final[k] = float(v[-1])
    if final:
        log_metrics(logger, final, stage="final")
    logger.info("Training finished.")


if __name__ == "__main__":
    default_cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "classification", "demo_synthetic.yaml",
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default_cfg
    main(path)
