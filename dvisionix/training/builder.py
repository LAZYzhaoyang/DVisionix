# -*- coding: utf-8 -*-
"""训练装配层：从配置构建 callbacks 与 Trainer。"""

import os
from typing import List, Optional

from ..utils.logging import TrainingLogger
from .callbacks import Callback, EarlyStopping, ModelCheckpoint
from .trainer import Trainer


def build_callbacks(cfg, work_dir: Optional[str] = None) -> List[Callback]:
    """根据配置构建回调列表（ModelCheckpoint / EarlyStopping）。"""
    callbacks: List[Callback] = []

    ckpt = cfg.get("checkpoint", {}) or {}
    ckpt_dir = ckpt.get("save_dir", "checkpoints")
    if work_dir and not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(work_dir, ckpt_dir)
    callbacks.append(
        ModelCheckpoint(
            save_dir=ckpt_dir,
            monitor=ckpt.get("monitor", "val_loss"),
            mode=ckpt.get("mode", "min"),
            save_best_only=ckpt.get("save_best_only", True),
            save_last=ckpt.get("save_last", True),
        )
    )

    training = cfg.get("training", {}) or {}
    es = training.get("early_stopping", {}) or {}
    if es.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=es.get("monitor", "val_loss"),
                mode=es.get("mode", "min"),
                patience=es.get("patience", 10),
                min_delta=es.get("min_delta", 0.0),
            )
        )
    return callbacks


def build_trainer(
    cfg,
    task,
    train_loader,
    val_loader,
    work_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    devices: Optional[List[int]] = None,
    strategy: Optional[str] = None,
    logger: Optional[TrainingLogger] = None,
) -> Trainer:
    """从配置构建统一 Trainer。"""
    training = cfg.get("training", {}) or {}
    callbacks = build_callbacks(cfg, work_dir)
    return Trainer(
        task=task,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks,
        work_dir=work_dir,
        device=training.get("device", "auto"),
        max_epochs=training.get("num_epochs", 10),
        gradient_clip_val=training.get("gradient_clip_val"),
        log_interval=training.get("log_interval", 50),
        amp=training.get("amp", False),
        accumulate_grad_batches=training.get("accumulate_grad_batches", 1),
        seed=training.get("seed"),
        resume_from=resume_from or training.get("resume_from"),
        strategy=strategy or training.get("strategy", "auto"),
        devices=devices or training.get("devices"),
        find_unused_parameters=training.get("find_unused_parameters", False),
        compile=training.get("compile", False),
        channels_last=training.get("channels_last", False),
        logger=logger,
    )


__all__ = ["build_callbacks", "build_trainer"]
