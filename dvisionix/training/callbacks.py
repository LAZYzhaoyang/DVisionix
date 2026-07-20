# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\training\\callbacks.py

"""
回调系统

支持在训练的各个阶段注入自定义逻辑，实现日志记录、检查点保存、早停等功能。
"""

import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Callback:
    """
    回调基类
    
    所有回调都继承此类，可以选择性地实现需要的钩子方法。
    
    每个方法接收 trainer 作为参数，可以访问 trainer 的所有属性：
    - trainer.model: 当前模型
    - trainer.optimizer: 优化器
    - trainer.current_epoch: 当前 epoch
    - trainer.global_step: 全局步数
    - trainer.train_logs: 训练日志
    """
    
    def on_train_begin(self, trainer: Any) -> None:
        """训练开始前调用"""
        pass
    
    def on_train_end(self, trainer: Any) -> None:
        """训练结束后调用"""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """
        epoch 开始前调用
        
        Args:
            epoch: 当前 epoch 编号（从 0 开始）
        """
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        """
        epoch 结束后调用
        
        Args:
            epoch: 当前 epoch 编号
            logs: 当前 epoch 的指标字典
        """
        pass
    
    def on_batch_begin(self, trainer: Any, batch_idx: int, mode: str) -> None:
        """
        batch 开始前调用
        
        Args:
            batch_idx: 当前 batch 编号
            mode: 'train' 或 'val'
        """
        pass
    
    def on_batch_end(self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str) -> None:
        """
        batch 结束后调用
        
        Args:
            batch_idx: 当前 batch 编号
            logs: 当前 batch 的指标字典
            mode: 'train' 或 'val'
        """
        pass


class CallbackList:
    """回调列表包装器，批量执行回调"""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def on_train_begin(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs)
    
    def on_batch_begin(self, trainer: Any, batch_idx: int, mode: str) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx, mode)
    
    def on_batch_end(self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, logs, mode)


# =============================================================================
# 内置回调实现
# =============================================================================

class ProgressBar(Callback):
    """简单的训练进度显示"""
    
    def __init__(self, log_interval: int = 50):
        """
        Args:
            log_interval: 每多少个 batch 打印一次日志
        """
        self.log_interval = log_interval
        self.epoch_start_time = 0.0
        self.batch_start_time = 0.0
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        self.epoch_start_time = time.time()
        print(f"\nEpoch [{epoch + 1}/{trainer.max_epochs}]")
        print("-" * 60)
    
    def on_batch_end(self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str) -> None:
        if batch_idx % self.log_interval == 0:
            prefix = "Train" if mode == "train" else "Val  "
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in logs.items())
            print(f"  [{prefix}] Batch {batch_idx}: {metrics_str}")
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        epoch_time = time.time() - self.epoch_start_time
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in logs.items())
        print(f"\nEpoch [{epoch + 1}] Summary: {metrics_str} | Time: {epoch_time:.1f}s")


class ModelCheckpoint(Callback):
    """模型检查点保存"""
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",  # 'min' or 'max'
        save_best_only: bool = True,
        save_last: bool = True,
        filename: Optional[str] = None,
    ):
        """
        Args:
            save_dir: 保存目录
            monitor: 监控的指标名称
            mode: 'min'（越小越好）或 'max'（越大越好）
            save_best_only: 是否只保存最佳模型
            save_last: 是否保存最后一个 epoch 的模型
            filename: 文件名模板，如 '{epoch:03d}-{val_loss:.4f}.pt'
        """
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.filename = filename or "checkpoint-{epoch:03d}.pt"
        
        if mode == "min":
            self.best_value = float("inf")
            self.is_better = lambda x, best: x < best
        else:
            self.best_value = float("-inf")
            self.is_better = lambda x, best: x > best
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        # 保存最后一个检查点
        if self.save_last:
            self._save(trainer, epoch, "last.pt")
        
        # 保存最佳检查点
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            if self.save_best_only:
                self._save(trainer, epoch, "best.pt")
                print(f"  [Checkpoint] New best {self.monitor}: {self.best_value:.4f}")
            else:
                filename = self.filename.format(epoch=epoch + 1, **logs)
                self._save(trainer, epoch, filename)
    
    def _save(self, trainer: Any, epoch: int, filename: str) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "best_value": self.best_value,
        }
        
        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)


class TensorBoardLogger(Callback):
    """TensorBoard 日志记录"""
    
    def __init__(self, log_dir: str = "./logs", log_every_n_steps: int = 10):
        """
        Args:
            log_dir: 日志目录
            log_every_n_steps: 每多少步记录一次 batch 日志
        """
        self.log_dir = Path(log_dir)
        self.log_every_n_steps = log_every_n_steps
        self.writer: Optional[SummaryWriter] = None
        
        if not TENSORBOARD_AVAILABLE:
            print("Warning: tensorboard not available. Install with 'pip install tensorboard'")
    
    def on_train_begin(self, trainer: Any) -> None:
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.log_dir)
    
    def on_batch_end(self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str) -> None:
        if self.writer is not None and batch_idx % self.log_every_n_steps == 0:
            global_step = trainer.global_step
            for k, v in logs.items():
                self.writer.add_scalar(f"{mode}/{k}", v, global_step)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        if self.writer is not None:
            for k, v in logs.items():
                self.writer.add_scalar(f"epoch/{k}", v, epoch)
    
    def on_train_end(self, trainer: Any) -> None:
        if self.writer is not None:
            self.writer.close()


class EarlyStopping(Callback):
    """早停机制"""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        """
        Args:
            monitor: 监控的指标名称
            mode: 'min' 或 'max'
            patience: 容忍多少个 epoch 没有提升
            min_delta: 最小提升幅度
            restore_best_weights: 是否恢复最佳权重
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        if mode == "min":
            self.best_value = float("inf")
            self.is_better = lambda x, best: x < best - min_delta
        else:
            self.best_value = float("-inf")
            self.is_better = lambda x, best: x > best + min_delta
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True  # 信号停止训练
                print(f"\nEarly stopping at epoch {epoch + 1}")
                
                if self.restore_best_weights and self.best_weights is not None:
                    trainer.model.load_state_dict(self.best_weights)
                    print("Restored best model weights")


class LearningRateScheduler(Callback):
    """学习率调度器"""
    
    def __init__(self, scheduler: Any, monitor: Optional[str] = None):
        """
        Args:
            scheduler: PyTorch 学习率调度器
            monitor: ReduceLROnPlateau 需要监控的指标
        """
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        if self.monitor is not None:
            # ReduceLROnPlateau 需要传入指标
            metric = logs.get(self.monitor)
            if metric is not None:
                self.scheduler.step(metric)
        else:
            # StepLR, CosineAnnealingLR 等不需要指标的调度器
            self.scheduler.step()
