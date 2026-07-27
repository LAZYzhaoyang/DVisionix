# -*- coding: utf-8 -*-
"""
目标检测端到端 Demo（Config 驱动，训练 + 解码/NMS/mAP 评估）

流程：配置加载 -> 合成带框数据(image+boxes+labels) ->
GridDetectionModel + DetectionTask + detection_collate + 回调 -> 训练 ->
解码 + NMS + mAP 评估。

使用内存中的合成数据（无需下载）。真实数据可用 DatasetFactory 加载 COCO/VOC，
或用 CustomDataset 提供 boxes/labels，再配合 detection_collate。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dvisionix.config import Config
from dvisionix.data import CustomDataset, detection_collate
from dvisionix.models import GridDetectionModel
from dvisionix.training import (
    Trainer,
    DetectionTask,
    evaluate_detection,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoardLogger,
)
from dvisionix.utils import get_logger


def build_det_dataset(tmp_dir, num_samples, num_classes, image_size, max_boxes):
    """生成随机图像及随机边界框/标签，构造检测用 CustomDataset。"""
    import cv2

    os.makedirs(tmp_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    # 每个类别用一种固定颜色画实心矩形，使目标可从视觉学习（mAP 才有意义）
    class_colors = [(220, 40, 40), (40, 200, 40), (40, 40, 220),
                    (220, 200, 40), (200, 40, 200)]
    samples = []
    for i in range(num_samples):
        # 低噪声背景
        img = rng.integers(0, 40, (image_size, image_size, 3), dtype=np.uint8)
        n = int(rng.integers(1, max_boxes + 1))
        boxes = []
        labels = []
        for _ in range(n):
            bw = int(rng.integers(12, image_size // 2))
            bh = int(rng.integers(12, image_size // 2))
            x1 = int(rng.integers(0, image_size - bw))
            y1 = int(rng.integers(0, image_size - bh))
            cls = int(rng.integers(0, num_classes))
            color = class_colors[cls % len(class_colors)]
            # OpenCV 用 BGR，这里直接写入（顺序一致即可，模型自行学习）
            cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), color, thickness=-1)
            boxes.append([float(x1), float(y1), float(x1 + bw), float(y1 + bh)])
            labels.append(cls)
        path = os.path.join(tmp_dir, f"img_{i:04d}.png")
        cv2.imwrite(path, img)
        samples.append({"image_path": path, "boxes": boxes, "labels": labels})

    return CustomDataset(
        task_type="detection",
        samples=samples,
        num_classes=num_classes,
        max_boxes=max_boxes,
    )


def main(config_path: str) -> None:
    cfg = Config.from_yaml(config_path)
    cfg.validate(["task_type", "model.name", "model.num_classes", "training.num_epochs"])

    logger = get_logger(
        "dvisionix.det_demo",
        level="info",
        log_dir=cfg.logging.get("log_dir", "./logs"),
    )
    logger.info(f"Loaded config: {config_path}")
    logger.info(f"Experiment: {cfg.get('experiment_name', 'unnamed')}")

    image_size = cfg.data.image_size
    num_classes = cfg.model.num_classes
    max_boxes = cfg.data.get("max_boxes", 3)

    data_cache = os.path.join(".cache", "synthetic_det")
    train_ds = build_det_dataset(
        os.path.join(data_cache, "train"),
        cfg.data.get("num_samples", 64), num_classes, image_size, max_boxes,
    )
    val_ds = build_det_dataset(
        os.path.join(data_cache, "val"),
        cfg.data.get("val_samples", 16), num_classes, image_size, max_boxes,
    )
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    batch_size = cfg.training.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=detection_collate,
    )

    model = GridDetectionModel(num_classes=num_classes, in_channels=cfg.model.get("in_channels", 3))
    logger.info(f"Model: {cfg.model.name}, params: {model.count_parameters():,}")

    task = DetectionTask(
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

    # 训练后：解码 + NMS + mAP 评估
    logger.info("Evaluating mAP on val set ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_detection(
        model, val_loader, num_classes=num_classes, device=device,
        score_threshold=0.3, iou_threshold=0.5,
    )
    logger.info(
        "mAP: {:.4f} | mAP@50: {:.4f} | mAP@75: {:.4f}".format(
            metrics["mAP"], metrics["mAP_50"], metrics["mAP_75"]
        )
    )


if __name__ == "__main__":
    default_cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "detection", "demo_synthetic.yaml",
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default_cfg
    main(path)
