# -*- coding: utf-8 -*-
"""
统一训练引擎（Trainer）

纯执行引擎，只负责循环流程，不包含任何任务特定逻辑。
所有任务逻辑通过 BaseTask 组件注入（分类 / 检测 / 分割 / 自定义任务）。

能力：
- AMP（fp16 GradScaler）、梯度累积（含 epoch 末 flush）
- 验证循环接入任务 MetricCollection（epoch 级指标）
- 多卡训练（DDP，strategy="ddp"），rank0 专属日志/保存
- 工作目录（work_dir）隔离、完整断点续训（model/optimizer/scheduler/scaler/callbacks/rng）
- 统一日志：utils.logging.TrainingLogger（console + file + JSONL + TensorBoard）
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import get_device, set_seed
from ..utils.logging import TrainingLogger
from .callbacks import Callback, CallbackList, ProgressBar
from .tasks import BaseTask


def _make_scaler(amp: bool, device: torch.device):
    """创建 AMP GradScaler（仅 CUDA 启用）。"""
    if not amp or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:  # pragma: no cover
        try:
            return torch.cuda.amp.GradScaler()
        except Exception:
            return None


def _infer_batch_size(step_result, batch) -> int:
    try:
        return int(batch["image"].shape[0])
    except Exception:
        return 1


def _concat_objects(objs: List[Any]) -> Any:
    """把 DDP all_gather 得到的多个对象拼接成一份（built-in 任务格式）。"""
    if isinstance(objs[0], torch.Tensor):
        return torch.cat(objs, dim=0)
    if isinstance(objs[0], (list, tuple)):
        merged = []
        for o in objs:
            merged.extend(o)
        return type(objs[0])(merged)
    return objs[0]


def _gather_tuple(seq: List[Any]) -> Any:
    """seq: 各 rank 的 (preds, targets) 元组列表 -> (合并 preds, 合并 targets)。"""
    preds = [item[0] for item in seq]
    targets = [item[1] for item in seq]
    return _concat_objects(preds), _concat_objects(targets)


class Trainer:
    """统一训练引擎。

    Args:
        task: BaseTask 实例（任务逻辑）。
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器（可选）。
        callbacks: 回调列表。
        work_dir: 工作目录（可选；提供后日志/检查点均在其中）。
        device: 设备（'auto' / 'cpu' / 'cuda' / 'cuda:0' 等）。
        max_epochs: 最大训练轮数。
        gradient_clip_val: 梯度裁剪范数阈值（None 表示不裁剪）。
        log_interval: 日志打印间隔（batch 数）。
        amp: 是否启用自动混合精度（仅 CUDA 生效）。
        accumulate_grad_batches: 梯度累积步数。
        seed: 随机种子。
        resume_from: 检查点路径（None 不恢复）。
        strategy: 'auto' / 'ddp' / 'none'。
        devices: DDP 使用的设备列表（如 [0, 1]；None 时使用 LOCAL_RANK）。
        find_unused_parameters: DDP 是否查找未使用参数。
        logger: 自定义 TrainingLogger（默认自动创建）。
    """

    def __init__(
        self,
        task: BaseTask,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callback]] = None,
        work_dir: Optional[str] = None,
        device: str = "auto",
        max_epochs: int = 10,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_value: Optional[float] = None,
        log_interval: int = 50,
        amp: bool = False,
        accumulate_grad_batches: int = 1,
        seed: Optional[int] = None,
        resume_from: Optional[str] = None,
        strategy: str = "auto",
        devices: Optional[List[int]] = None,
        find_unused_parameters: bool = False,
        logger: Optional[TrainingLogger] = None,
    ):
        self.task = task
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.work_dir = work_dir
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_value = gradient_clip_value
        self.log_interval = log_interval
        self.amp = amp
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.seed = seed
        self.resume_from = resume_from
        self.find_unused_parameters = find_unused_parameters

        # 分布式状态
        self.strategy = strategy
        self.devices = devices
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self._init_distributed()

        # 设备设置
        if self.is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = get_device(device)
        self.scaler = _make_scaler(self.amp, self.device)

        # 日志
        log_dir = os.path.join(work_dir, "logs") if work_dir else None
        tb_dir = os.path.join(work_dir, "tb") if work_dir else None
        self.logger = logger or TrainingLogger("dvisionix.trainer", log_dir=log_dir, tb_dir=tb_dir)
        self.logger.info(
            f"Using device: {self.device}, amp: {bool(self.scaler)}, strategy: {self.strategy}"
        )

        # 回调系统
        default_callbacks = [ProgressBar(log_interval=log_interval)]
        if callbacks:
            self.callbacks = CallbackList(default_callbacks + callbacks)
        else:
            self.callbacks = CallbackList(default_callbacks)

        # 训练状态
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scheduler_monitor: Optional[str] = None
        self.current_epoch = 0
        self.global_step = 0
        self.stop_training = False
        self.history: List[Dict[str, float]] = []
        self.teacher_logits = None  # DistillCallback 使用

    # ------------------------------------------------------------------
    # 分布式
    # ------------------------------------------------------------------
    def _init_distributed(self) -> None:
        import torch.distributed as dist

        if self.strategy == "auto":
            self.strategy = (
                "ddp"
                if dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
                and torch.cuda.is_available()
                else "none"
            )
        if self.strategy == "ddp":
            if not dist.is_initialized():
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "strategy='ddp' requires CUDA. Use 'none' or 'auto' on CPU, "
                        "or launch with torchrun on a multi-GPU machine."
                    )
                dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_distributed = True
        else:
            self.is_distributed = False
            self.rank = 0
            self.world_size = 1

    def _is_rank0(self) -> bool:
        return not self.is_distributed or self.rank == 0

    def _make_distributed_loader(self, loader: DataLoader, shuffle: bool) -> DataLoader:
        if not self.is_distributed:
            return loader
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            loader.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )
        # DDP 下要求各 rank 批数一致，drop_last=True 避免 all_gather 死锁
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=True,
        )

    def _set_sampler_epoch(self, epoch: int) -> None:
        if not self.is_distributed:
            return
        for loader in (self.train_loader, self.val_loader):
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        if self.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=self.find_unused_parameters,
            )
        return model

    # ------------------------------------------------------------------
    # 训练主流程
    # ------------------------------------------------------------------
    def fit(self, model: nn.Module) -> Dict[str, Any]:
        if self.seed is not None:
            set_seed(self.seed + self.rank)
        self.model = self._wrap_model(model)

        # 配置优化器和学习率调度器
        opt_config = self.task.configure_optimizers(self.model)
        if isinstance(opt_config, dict):
            self.optimizer = opt_config["optimizer"]
            self.scheduler = opt_config.get("lr_scheduler")
            self.scheduler_monitor = opt_config.get("monitor")
        elif isinstance(opt_config, tuple) and len(opt_config) == 2:
            self.optimizer, self.scheduler = opt_config
        else:
            self.optimizer = opt_config
            self.scheduler = None

        # 自动 resume（在优化器/调度器就绪后再加载状态）
        if self.resume_from is not None:
            self.load_checkpoint(self.resume_from, self.model)

        # DDP 数据加载器（DistributedSampler）
        self.train_loader = self._make_distributed_loader(self.train_loader, shuffle=True)
        if self.val_loader is not None:
            self.val_loader = self._make_distributed_loader(self.val_loader, shuffle=False)

        self.callbacks.on_train_begin(self)
        self.logger.info(f"Start training for {self.max_epochs} epochs")
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        if self.val_loader is not None:
            self.logger.info(f"Val batches: {len(self.val_loader)}")

        for epoch in range(self.current_epoch, self.max_epochs):
            if self.stop_training:
                break
            self.current_epoch = epoch
            self._set_sampler_epoch(epoch)

            self.callbacks.on_epoch_begin(self, epoch)
            train_logs = self._run_epoch("train")

            val_logs: Dict[str, float] = {}
            if self.val_loader is not None:
                val_logs = self._run_epoch("val")

            epoch_logs = {
                **{f"train_{k}": v for k, v in train_logs.items()},
                **{f"val_{k}": v for k, v in val_logs.items()},
            }

            # 任务级 epoch 指标（MetricCollection）
            if self.val_loader is not None and not self.is_distributed:
                epoch_logs.update(self.task.on_validation_epoch_end())
            elif self.is_distributed and self.rank == 0:
                epoch_logs.update(self.task.on_validation_epoch_end())

            # 学习率调度（epoch 级）
            if self.scheduler is not None:
                if self.scheduler_monitor is not None:
                    metric = epoch_logs.get(self.scheduler_monitor)
                    if metric is not None:
                        self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            self.history.append(epoch_logs)
            self.callbacks.on_epoch_end(self, epoch, epoch_logs)

        self.callbacks.on_train_end(self)
        self._write_history_csv()
        self.logger.info("Training finished!")
        if self._is_rank0():
            self.logger.log_event(
                "train_end", epochs=self.current_epoch, global_step=self.global_step
            )

        return {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "history": self.history,
        }

    # ------------------------------------------------------------------
    # 单 epoch
    # ------------------------------------------------------------------
    def _optimizer_step(self) -> None:
        if self.gradient_clip_value is not None:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.gradient_clip_value)
        elif self.gradient_clip_val is not None:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    def _run_epoch(self, mode: str) -> Dict[str, float]:
        if mode == "train":
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader
            self.callbacks.on_validation_begin(self)

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            self.callbacks.on_batch_begin(self, batch_idx, mode, batch)

            if mode == "train":
                with torch.autocast(device_type=self.device.type, enabled=bool(self.scaler)):
                    step_result = self.task.training_step(self.model, batch, self.device)
                loss = step_result["loss"] / self.accumulate_grad_batches
                if self.scaler is not None:
                    loss = self.scaler.scale(loss)
                loss.backward()
                if (batch_idx + 1) % self.accumulate_grad_batches == 0 or (
                    batch_idx + 1
                ) == total_batches:
                    self._optimizer_step()
                    self.global_step += 1
            else:
                with torch.no_grad():
                    step_result = self.task.validation_step(self.model, batch, self.device)
                if not self.is_distributed:
                    preds = step_result.get("preds")
                    targets = step_result.get("targets")
                    if preds is not None and targets is not None:
                        self.task.update_metrics(preds, targets)
                else:
                    self._gather_and_update_metrics(step_result)

            # 分离张量，转换为 Python float（跳过 preds/targets）
            step_logs: Dict[str, float] = {}
            for k, v in step_result.items():
                if k in ("preds", "targets"):
                    continue
                if isinstance(v, torch.Tensor):
                    step_logs[k] = v.detach().cpu().item()
                else:
                    step_logs[k] = float(v)

            # 按批次大小加权累积
            batch_size = _infer_batch_size(step_result, batch)
            for k, v in step_logs.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v * batch_size
                metric_counts[k] = metric_counts.get(k, 0) + batch_size

            self.callbacks.on_batch_end(self, batch_idx, step_logs, mode, batch)

        if mode == "val":
            self.callbacks.on_validation_end(self)
        avg_metrics = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
        return avg_metrics

    def _gather_and_update_metrics(self, step_result: Dict[str, Any]) -> None:
        import torch.distributed as dist

        preds = step_result.get("preds")
        targets = step_result.get("targets")
        if preds is None or targets is None:
            return
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, (preds, targets))
        if self.rank == 0:
            all_preds, all_targets = _gather_tuple(gathered)
            self.task.update_metrics(all_preds, all_targets)

    # ------------------------------------------------------------------
    # 独立验证
    # ------------------------------------------------------------------
    def validate(
        self, model: nn.Module, val_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        self.model = model.to(self.device)
        loader = val_loader or self.val_loader
        if loader is None:
            raise ValueError("No validation loader provided")

        self.model.eval()
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        self.task.reset_metrics()

        with torch.no_grad():
            for batch in loader:
                step_result = self.task.validation_step(self.model, batch, self.device)
                preds = step_result.get("preds")
                targets = step_result.get("targets")
                if preds is not None and targets is not None:
                    self.task.update_metrics(preds, targets)
                for k, v in step_result.items():
                    if k in ("preds", "targets"):
                        continue
                    value = v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                    metric_sums[k] = metric_sums.get(k, 0.0) + value
                    metric_counts[k] = metric_counts.get(k, 0) + 1

        avg = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
        avg.update(self.task.on_validation_epoch_end())
        return avg

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def predict(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        model = model.to(self.device)
        model.eval()
        images = batch["image"].to(self.device)
        with torch.no_grad():
            return model(images)

    # ------------------------------------------------------------------
    # 检查点
    # ------------------------------------------------------------------
    def _collect_rng_state(self) -> Dict[str, Any]:
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
        return {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": cuda_state,
        }

    def _apply_rng_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        random.setstate(state.get("random", random.getstate()))
        np.random.set_state(state.get("numpy", np.random.get_state()))
        torch.set_rng_state(state.get("torch", torch.get_rng_state()))
        cuda_state = state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    def save_checkpoint(self, path: str) -> None:
        if not self._is_rank0():
            return
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict() if self.model else None,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "rng_state": self._collect_rng_state(),
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        cb_state = self.callbacks.state_dict()
        if cb_state:
            checkpoint["callbacks_state_dict"] = cb_state
        task_state = getattr(self.task, "state_dict", None)
        if callable(task_state):
            ts = task_state()
            if ts:
                checkpoint["task_state_dict"] = ts

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str, model: nn.Module, strict: bool = True) -> None:
        # torch 2.6 起默认 weights_only=True，导致完整 checkpoint 无法反序列化
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:  # pragma: no cover
            checkpoint = torch.load(path, map_location=self.device)

        if self.model is None:
            self.model = model.to(self.device)

        if checkpoint.get("model_state_dict"):
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if checkpoint.get("optimizer_state_dict") and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if checkpoint.get("callbacks_state_dict"):
            self.callbacks.load_state_dict(checkpoint["callbacks_state_dict"])
        if checkpoint.get("task_state_dict"):
            load_task = getattr(self.task, "load_state_dict", None)
            if callable(load_task):
                load_task(checkpoint["task_state_dict"])
        if checkpoint.get("rng_state"):
            self._apply_rng_state(checkpoint["rng_state"])

        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.logger.info(f"Checkpoint loaded from: {path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def _write_history_csv(self) -> None:
        """将训练 history 导出到 work_dir/history.csv（可选，无 work_dir 时跳过）。"""
        if not self.work_dir or not self.history:
            return
        try:
            import csv

            keys = sorted({k for epoch in self.history for k in epoch.keys()})
            path = os.path.join(self.work_dir, "history.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for epoch in self.history:
                    writer.writerow({k: epoch.get(k, "") for k in keys})
            self.logger.info(f"History exported to: {path}")
        except Exception as exc:  # pragma: no cover
            self.logger.warning(f"Failed to export history.csv: {exc}")


__all__ = ["Trainer"]
