# -*- coding: utf-8 -*-
# NOTE: v0.2.0 起推荐使用 config 驱动入口:
#   python tools/train.py --config configs/<task>/demo_synthetic.yaml
# 该脚本作为教学参考保留，功能上等价的现代用法请参考 tools/train.py + configs/。
"""
语义分割端到端 Demo（Config 驱动）

演示分割任务的完整流程：配置加载 -> 合成数据(image+mask) ->
SimpleSegmentationModel + SegmentationTask + 回调 -> 训练。

使用内存中的合成数据（无需下载）。把 data.dataset 换成 cityscapes/ade20k
并提供真实 mask_path 即可用于真实数据。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader

from dvisionix.config import Config
from dvisionix.data import CustomDataset
from dvisionix.models import SimpleSegmentationModel
from dvisionix.training import (
    Trainer,
    SegmentationTask,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoardLogger,
)
from dvisionix.utils import get_logger


def build_seg_dataset(tmp_dir, num_samples, num_classes, image_size, transforms=None):
    """生成固定尺寸的随机图像与掩码，构造分割用 CustomDataset。"""
    import cv2

    img_dir = os.path.join(tmp_dir, "images")
    mask_dir = os.path.join(tmp_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    samples = []
    for i in range(num_samples):
        img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        mask = np.random.randint(0, num_classes, (image_size, image_size), dtype=np.uint8)
        img_path = os.path.join(img_dir, f"img_{i:04d}.png")
        mask_path = os.path.join(mask_dir, f"mask_{i:04d}.png")
        if not os.path.exists(img_path):
            cv2.imwrite(img_path, img)
        if not os.path.exists(mask_path):
            cv2.imwrite(mask_path, mask)
        samples.append({"image_path": img_path, "mask_path": mask_path})

    return CustomDataset(
        task_type="segmentation",
        samples=samples,
        num_classes=num_classes,
        transforms=transforms,
    )


def main(config_path: str) -> None:
    cfg = Config.from_yaml(config_path)
    cfg.validate(["task_type", "model.name", "model.num_classes", "training.num_epochs"])

    logger = get_logger(
        "dvisionix.seg_demo",
        level="info",
        log_dir=cfg.logging.get("log_dir", "./logs"),
    )
    logger.info(f"Loaded config: {config_path}")
    logger.info(f"Experiment: {cfg.get('experiment_name', 'unnamed')}")

    image_size = cfg.data.image_size
    num_classes = cfg.model.num_classes

    data_cache = os.path.join(".cache", "synthetic_seg")
    train_ds = build_seg_dataset(
        os.path.join(data_cache, "train"),
        cfg.data.get("num_samples", 64), num_classes, image_size,
    )
    val_ds = build_seg_dataset(
        os.path.join(data_cache, "val"),
        cfg.data.get("val_samples", 16), num_classes, image_size,
    )
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    batch_size = cfg.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleSegmentationModel(num_classes=num_classes, in_channels=cfg.model.get("in_channels", 3))
    logger.info(f"Model: {cfg.model.name}, params: {model.count_parameters():,}")

    task = SegmentationTask(
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
    trainer.fit(model)
    logger.info("Training finished.")


if __name__ == "__main__":
    default_cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "segmentation", "demo_synthetic.yaml",
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default_cfg
    main(path)
