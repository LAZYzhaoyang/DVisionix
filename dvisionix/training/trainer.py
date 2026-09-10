# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 统一训练引擎（Trainer）
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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import get_device, set_seed
from ..utils.logging import TrainingLogger
from .callbacks import Callback, CallbackList, ModelCheckpoint, ProgressBar
from .tasks import BaseTask

#: 完整 checkpoint 的结构版本。
#: 只要 checkpoint 的字段结构发生变化就必须递增，使旧文件被**明确拒绝**，
#: 而不是按错误的结构解出一份看似正常的训练状态（CodePlan 7.5 步骤 3-3）。
CHECKPOINT_SCHEMA_VERSION = 1


def _unwrap_module(model: Any) -> Any:
    """剥掉 DDP / torch.compile 的包装，拿到真正的模型对象。

    ``save_checkpoint`` 时 ``self.model`` 可能已被 DDP 或 ``torch.compile`` 包装，
    直接取类名会得到 ``DistributedDataParallel`` / ``OptimizedModule``，
    让「模型类型一致性」校验在单卡/多卡之间误报。
    """
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        inner = getattr(model, "module", None) or getattr(model, "_orig_mod", None)
        if inner is None or inner is model:
            break
        model = inner
    return model


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


def _infer_batch_size(batch: Dict[str, Any]) -> int:
    """从 batch 取样本数，用于按样本数加权指标的均值。

    只认明确的图像键（``image`` / SimCLR 的 ``image1``），取不到就**显式报错**。
    v1.0.0 用的是「遍历 step_result 与 batch 的所有值、取第一个带 batch 维的张量」，
    当 ``preds``/``targets`` 是 ``(Tensor, Tensor)`` 形态时会返回 **tuple 的长度 2**
    作为 batch size；``except: return 1`` 也会把错误静默成「每个 batch 记 1 个样本」。
    """
    if isinstance(batch, dict):
        for key in ("image", "image1", "image2"):
            value = batch.get(key)
            if value is not None and hasattr(value, "shape") and value.dim() > 0:
                return int(value.shape[0])
    raise ValueError(
        "无法从 batch 推断 batch size：batch 必须含形如 (B, ...) 的 'image' 键"
        "（SimCLR 可用 'image1'/'image2'）。自定义任务请确保 batch 携带 image。"
    )


def _values_equal(a: Any, b: Any) -> bool:
    """安全比较两个标量/数组是否相等（避免 numpy 数组触发二义性真值错误）。"""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:
            return False
    try:
        return bool(a == b)
    except Exception:
        return a is b


def _concat_objects(objs: List[Any]) -> Any:
    """递归合并 DDP ``all_gather`` 得到的各 rank 对象。

    容器类型决定合并语义（与内置任务的返回约定一一对应）：

    - ``Tensor``：沿 dim=0 拼接（batch 维）；
    - ``list``：**按样本排列的变长结果**，逐元素 extend；
    - ``tuple``：**固定字段结构**，按位置递归合并；
    - ``dict``：按 key 递归合并；
    - 其它：要求各 rank 完全一致。

    注意 tuple 不能像 list 那样 extend：检测任务的
    ``preds = (boxes_list, scores_list, labels_list)`` 一旦被拍平成 6 元组，
    ``task.update_metrics`` 的三元解包就会出错（CodePlan 7.1.2 D5）。
    """
    if not objs:
        return objs
    first = objs[0]

    if isinstance(first, torch.Tensor):
        if not all(isinstance(o, torch.Tensor) for o in objs):
            raise TypeError("DDP gathered objects have inconsistent tensor structure")
        return torch.cat(objs, dim=0)

    if isinstance(first, list):
        if not all(isinstance(o, list) for o in objs):
            raise TypeError("DDP gathered objects have inconsistent list structure")
        merged: List[Any] = []
        for obj in objs:
            merged.extend(obj)
        return merged

    if isinstance(first, tuple):
        if not all(isinstance(o, tuple) and len(o) == len(first) for o in objs):
            raise ValueError("DDP gathered tuples have inconsistent structure")
        parts = [_concat_objects([obj[i] for obj in objs]) for i in range(len(first))]
        # 保留 namedtuple 的类型语义
        if hasattr(first, "_fields"):
            return type(first)(*parts)
        return tuple(parts)

    if isinstance(first, dict):
        keys = set(first)
        if not all(isinstance(o, dict) and set(o) == keys for o in objs):
            raise ValueError("DDP gathered dictionaries have inconsistent keys")
        return {key: _concat_objects([obj[key] for obj in objs]) for key in first}

    if any(not _values_equal(o, first) for o in objs[1:]):
        raise ValueError("DDP gathered scalar objects have inconsistent values")
    return first


def _gather_tuple(seq: List[Any]) -> Any:
    """seq: 各 rank 的 (preds, targets) 元组列表 -> (合并 preds, 合并 targets)。"""
    preds = [item[0] for item in seq]
    targets = [item[1] for item in seq]
    return _concat_objects(preds), _concat_objects(targets)


def _gather_preds_targets(
    step_result: Dict[str, Any], world_size: int, rank: int
) -> Optional[Tuple[Any, Any]]:
    """``all_gather`` 各 rank 的 (preds, targets) 并递归合并。

    训练中验证与独立 ``Trainer.validate()`` **共用本函数**，以保证两条路径的全局
    指标口径一致（CodePlan 7.4 步骤 2-1 / D8）。

    Returns:
        rank0 上返回合并后的 ``(preds, targets)``；非 rank0 或 step_result 不含
        preds/targets 时返回 ``None``。
    """
    import torch.distributed as dist

    preds = step_result.get("preds")
    targets = step_result.get("targets")
    if preds is None or targets is None:
        return None
    gathered: List[Any] = [None] * world_size
    dist.all_gather_object(gathered, (preds, targets))
    if rank != 0:
        return None
    return _gather_tuple(gathered)


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
        compile: 是否启用 torch.compile（DDP wrap 前编译；失败自动降级并告警）。
        channels_last: 是否将模型转为 channels_last 内存格式（卷积网络友好）。
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
        compile: bool = False,
        channels_last: bool = False,
        logger: Optional[TrainingLogger] = None,
        config_hash: Optional[str] = None,
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
        self.compile = compile
        self.channels_last = channels_last
        # 解析后配置的哈希：写入 checkpoint 并在 resume 时校验配置一致性
        self.config_hash = config_hash

        # 分布式状态
        self.strategy = strategy
        self.devices = devices
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self._init_distributed()

        # 设备设置
        if self.is_distributed and torch.cuda.is_available():
            # CUDA DDP：每个进程绑定自己的 LOCAL_RANK 卡
            local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            # 单卡、或 CPU gloo 分布式：走通用设备解析（不能强制 cuda:*）
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
            # 进程组已初始化即意味着处于分布式启动中（torchrun / mp.spawn），
            # 与是否 CUDA 无关 —— CPU gloo 是官方的回归路径（CodePlan 7.4 步骤 2-1）。
            self.strategy = (
                "ddp"
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
                else "none"
            )
        if self.strategy == "ddp":
            if not dist.is_initialized():
                if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
                    raise RuntimeError(
                        "strategy='ddp' 要求进程组已初始化。请用 torchrun 启动"
                        "（会注入 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT），"
                        "或先自行调用 dist.init_process_group(backend='gloo')"
                        "（CPU 双进程一致性测试见 tests/test_training/test_ddp_cpu.py）。"
                    )
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
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
        if self.channels_last:
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception as exc:  # pragma: no cover
                self.logger.warning(f"channels_last 转换失败，已忽略：{exc}")
        if self.compile:
            try:
                model = torch.compile(model)
            except Exception as exc:
                self.logger.warning(f"torch.compile 不可用，已降级为普通模型：{exc}")
        if self.is_distributed:
            ddp_kwargs: Dict[str, Any] = {
                "find_unused_parameters": self.find_unused_parameters,
            }
            # device_ids/output_device 只对 CUDA 有意义；CPU gloo 下传 None
            # （传 [self.device.index] 会变成 [None] 并直接报错）。
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.device.index]
                ddp_kwargs["output_device"] = self.device.index
            model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        return model

    # ------------------------------------------------------------------
    # 训练主流程
    # ------------------------------------------------------------------
    def fit(self, model: nn.Module) -> Dict[str, Any]:
        """训练主循环：多 epoch 训练 + 验证 + 回调 + history 导出。"""
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

            # 任务级 epoch 指标（accuracy / f1 / mAP ...）已在 _evaluate 内由
            # `on_validation_epoch_end()` 计算并 reset，并以 val_ 前缀并入 val_logs。
            # 这里**不能**再调用一次：它在已重置的累加器上会算出全 0
            # 并覆盖上面的正确值（v1.0.0 的 history.csv 里那列裸 accuracy 正是这么来的）。

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
        self._write_best_metrics_csv()
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
        if mode != "train":
            # 验证一律走 _evaluate —— 与独立 validate() 同一实现
            if self.val_loader is None:
                raise ValueError("No validation loader provided")
            return self._evaluate(self.val_loader)

        self.model.train()
        loader = self.train_loader

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            self.callbacks.on_batch_begin(self, batch_idx, mode, batch)

            with torch.autocast(device_type=self.device.type, enabled=bool(self.scaler)):
                step_result = self.task.training_step(self.model, batch, self.device)
            loss = step_result["loss"] / self._accumulation_divisor(batch_idx, total_batches)
            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            loss.backward()
            if (batch_idx + 1) % self.accumulate_grad_batches == 0 or (
                batch_idx + 1
            ) == total_batches:
                self._optimizer_step()
                self.global_step += 1

            step_logs = self._accumulate_step_logs(
                step_result, _infer_batch_size(batch), metric_sums, metric_counts
            )
            self.callbacks.on_batch_end(self, batch_idx, step_logs, mode, batch)

        return {k: metric_sums[k] / metric_counts[k] for k in metric_sums}

    def _accumulation_divisor(self, batch_idx: int, total_batches: int) -> int:
        """当前 micro-batch 所在累积窗口的**实际长度**，作为梯度累积分母。

        v1.0.0 一律除以 ``accumulate_grad_batches``，于是末尾不足一个完整窗口时
        梯度被系统性低估：例如 5 个 batch + accum=2 的窗口是 2/2/1，
        尾窗只有 1 个 batch 却仍除以 2（CodePlan 7.5 步骤 3-3）。

        分母必须是**所在窗口**的长度而不是「从当前位置到末尾的剩余数」——
        否则 4 batch + accum=2 时最后一个 batch 会被误算成 /1。
        """
        accum = self.accumulate_grad_batches
        if accum <= 1:
            return 1
        window_start = (batch_idx // accum) * accum
        return max(1, min(accum, total_batches - window_start))

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """在给定 loader 上评估，返回按 batch size 加权的指标与 epoch 级指标。

        **训练中验证与独立 ``validate()`` 共用本实现**，因此两条路径口径完全一致，
        并且独立 ``validate()`` 也会触发 ``on_validation_begin`` / ``on_validation_end``
        —— EMA 回调在这两个钩子里交换/恢复权重，v1.0.0 的 ``validate()`` 完全不走回调，
        所以它报的是**未交换 EMA 权重**的指标（CodePlan 7.1.2 D8）。
        """
        self.model.eval()
        self.task.reset_metrics()
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        self.callbacks.on_validation_begin(self)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                self.callbacks.on_batch_begin(self, batch_idx, "val", batch)
                step_result = self.task.validation_step(self.model, batch, self.device)
                self._update_metrics_for_step(step_result)
                step_logs = self._accumulate_step_logs(
                    step_result, _infer_batch_size(batch), metric_sums, metric_counts
                )
                self.callbacks.on_batch_end(self, batch_idx, step_logs, "val", batch)
        self.callbacks.on_validation_end(self)

        avg = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
        # on_validation_epoch_end 同时负责 compute 与 reset：
        # 必须**只调用一次**，否则第二次会在已重置的累加器上算出全 0
        avg.update(self.task.on_validation_epoch_end())
        return avg

    def _gather_and_update_metrics(self, step_result: Dict[str, Any]) -> None:
        """分布式：all_gather 各 rank 结果，仅 rank0 用全局结果更新指标。"""
        merged = _gather_preds_targets(step_result, self.world_size, self.rank)
        if merged is not None:
            self.task.update_metrics(*merged)

    def _update_metrics_for_step(self, step_result: Dict[str, Any]) -> None:
        """验证步的指标更新入口 —— 训练中验证与 ``validate()`` 共用同一实现。

        单进程直接更新；分布式下走 ``_gather_and_update_metrics``（全局指标，
        而非只统计 rank0 分片）。
        """
        if self.is_distributed:
            self._gather_and_update_metrics(step_result)
            return
        preds = step_result.get("preds")
        targets = step_result.get("targets")
        if preds is not None and targets is not None:
            self.task.update_metrics(preds, targets)

    @staticmethod
    def _accumulate_step_logs(
        step_result: Dict[str, Any],
        batch_size: int,
        metric_sums: Dict[str, float],
        metric_counts: Dict[str, int],
    ) -> Dict[str, float]:
        """把 step_result 的标量日志按 batch size 加权累加，返回本步日志。

        加权是必要的：最后一个 batch 常常更小，等权平均会让它被过度放大。
        训练中验证与 ``validate()`` 共用本函数，避免两条路径给出不同的 val_loss
        （CodePlan 7.1.2 D8）。
        """
        step_logs: Dict[str, float] = {}
        for key, value in step_result.items():
            if key in ("preds", "targets"):
                continue
            step_logs[key] = (
                value.detach().cpu().item() if isinstance(value, torch.Tensor) else float(value)
            )
        for key, value in step_logs.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size
            metric_counts[key] = metric_counts.get(key, 0) + batch_size
        return step_logs

    # ------------------------------------------------------------------
    # 独立验证
    # ------------------------------------------------------------------
    def validate(
        self, model: nn.Module, val_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """独立验证：对给定模型与验证集计算指标并返回。

        走与训练中验证完全相同的 ``_evaluate`` 实现：
        分布式下给出**全局指标**（不再只统计 rank0 分片），
        并正常触发 ``on_validation_begin/end``（EMA 权重交换依赖它们）。
        """
        self.model = model.to(self.device)
        loader = val_loader or self.val_loader
        if loader is None:
            raise ValueError("No validation loader provided")
        return self._evaluate(loader)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def predict(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        """推理：对单个 batch 做模型前向。"""
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
        """保存完整检查点（model/optimizer/scheduler/scaler/rng/callbacks/task + 元信息）。"""
        if not self._is_rank0():
            return
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict() if self.model else None,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "rng_state": self._collect_rng_state(),
            **self._checkpoint_meta(),
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

    def _checkpoint_meta(self) -> Dict[str, Any]:
        """写成 checkpoint 的运行元信息，供 resume 时做兼容性校验。"""
        model = _unwrap_module(self.model)
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "task_type": type(self.task).__name__ if self.task is not None else None,
            "model_type": type(model).__name__ if model is not None else None,
            "config_hash": self.config_hash,
        }

    @staticmethod
    def _verify_checkpoint_meta(
        saved: Dict[str, Any],
        current: Dict[str, Any],
        path: str,
        allow_config_mismatch: bool,
    ) -> None:
        """校验 checkpoint 与当前运行是否兼容。

        v1.0.0 的 resume 完全不检查任何元信息：拿另一个任务的 checkpoint 续训同一个
        Trainer 会静默加载形状兼容但语义不同的权重，得到没有意义的结果
        （CodePlan 7.5 步骤 3-3）。
        """
        schema = saved.get("schema_version")
        if schema is None:
            warnings.warn(
                f"checkpoint {path} 缺少 schema_version（v1.0.0 之前保存的旧格式），"
                f"已按当前结构加载；建议用当前版本重新保存。",
                DeprecationWarning,
                stacklevel=3,
            )
        elif int(schema) > CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint {path} 的 schema_version={schema} 高于当前支持的 "
                f"{CHECKPOINT_SCHEMA_VERSION}；请升级代码后再加载。"
            )

        for key, label in (("task_type", "任务类型"), ("model_type", "模型类型")):
            saved_value, current_value = saved.get(key), current.get(key)
            if saved_value and current_value and saved_value != current_value:
                raise ValueError(
                    f"checkpoint {path} 的{label}为 {saved_value!r}，当前为 {current_value!r}；"
                    f"该 checkpoint 不属于当前任务，拒绝加载。"
                )

        saved_hash, current_hash = saved.get("config_hash"), current.get("config_hash")
        if saved_hash and current_hash and saved_hash != current_hash and not allow_config_mismatch:
            raise ValueError(
                f"checkpoint {path} 的配置哈希为 {saved_hash}，当前配置为 {current_hash}；"
                f"配置不一致的 resume 可能静默产生无意义的训练结果。"
                f"确认确实要续训请显式传 allow_config_mismatch=True。"
            )

    def load_checkpoint(
        self,
        path: str,
        model: nn.Module,
        strict: bool = True,
        allow_config_mismatch: bool = False,
    ) -> None:
        """加载检查点并恢复训练状态（断点续训）。

        Args:
            path: checkpoint 路径。
            model: 目标模型（``self.model`` 为空时使用）。
            strict: 传给 ``load_state_dict`` 的 strict 开关。
            allow_config_mismatch: 显式允许配置哈希不一致的续训（默认拒绝）。
        """
        # torch 2.6 起默认 weights_only=True，导致完整 checkpoint 无法反序列化
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:  # pragma: no cover
            checkpoint = torch.load(path, map_location=self.device)

        if not isinstance(checkpoint, dict) or not any(
            key in checkpoint
            for key in ("model_state_dict", "optimizer_state_dict", "epoch", "global_step")
        ):
            raise ValueError(
                f"{path} 不是完整 checkpoint（缺少 model_state_dict / epoch 等字段）。"
                f"纯 state_dict 请改用 dvisionix.training.load_backbone 加载骨干权重，"
                f"或显式包装为 {{'model_state_dict': state_dict}}。"
            )

        # 先挂上模型，_checkpoint_meta 才能取到真实的模型类名用于一致性校验
        if self.model is None:
            self.model = model.to(self.device)

        self._verify_checkpoint_meta(
            checkpoint, self._checkpoint_meta(), path, allow_config_mismatch
        )

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

    def _best_metrics(self) -> Optional[Dict[str, Any]]:
        """从 history 中按 ModelCheckpoint 的 monitor/mode 选出最优 epoch。

        Returns:
            (epoch_index, epoch_logs, best_value) 或 None（无可用监控指标）。
        """
        monitor, mode = None, "min"
        for cb in getattr(self.callbacks, "callbacks", None) or []:
            if isinstance(cb, ModelCheckpoint):
                monitor, mode = cb.monitor, cb.mode
                break
        if not monitor or not self.history:
            return None
        best = None  # (index, logs, value)
        for idx, epoch in enumerate(self.history):
            if monitor not in epoch:
                continue
            try:
                val = float(epoch[monitor])
            except (TypeError, ValueError):
                continue
            if best is None or (val < best[2] if mode == "min" else val > best[2]):
                best = (idx, epoch, val)
        return best

    def _write_best_metrics_csv(self) -> None:
        """导出最优 epoch 指标到 work_dir/best_metrics.csv（配合 ModelCheckpoint 的监控指标）。"""
        if not self.work_dir or not self.history:
            return
        best = self._best_metrics()
        if best is None:
            return
        idx, epoch_logs, _ = best
        try:
            import csv

            keys = ["best_epoch"] + sorted({k for epoch in self.history for k in epoch.keys()})
            path = os.path.join(self.work_dir, "best_metrics.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                row = {k: epoch_logs.get(k, "") for k in keys if k != "best_epoch"}
                row["best_epoch"] = idx
                writer.writerow(row)
            self.logger.info(f"Best metrics exported to: {path}")
        except Exception as exc:  # pragma: no cover
            self.logger.warning(f"Failed to export best_metrics.csv: {exc}")

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
