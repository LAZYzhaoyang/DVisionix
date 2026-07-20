# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\training\\trainer.py

"""
通用训练引擎

纯执行引擎，只负责循环流程，不包含任何任务特定逻辑。
所有任务逻辑都通过 BaseTask 接口注入。
"""

import os
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import get_device
from .task import BaseTask
from .callbacks import Callback, CallbackList, ProgressBar, ModelCheckpoint


class Trainer:
    """
    通用训练引擎
    
    纯执行引擎，只负责循环流程。所有任务特定逻辑都在 Task 中实现。
    
    Examples:
        >>> # 1. 定义任务
        >>> task = ClassificationTask(num_classes=10)
        >>>
        >>> # 2. 创建数据加载器
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> val_loader = DataLoader(val_dataset, batch_size=32)
        >>>
        >>> # 3. 创建 Trainer
        >>> trainer = Trainer(
        ...     task=task,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     max_epochs=10,
        ...     device="auto",
        ... )
        >>>
        >>> # 4. 开始训练
        >>> trainer.fit(model)
    """
    
    def __init__(
        self,
        task: BaseTask,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callback]] = None,
        device: str = "auto",
        max_epochs: int = 10,
        gradient_clip_val: Optional[float] = None,
        log_interval: int = 50,
    ):
        """
        初始化通用训练引擎
        
        Args:
            task: 任务实例，实现 training_step, validation_step, configure_optimizers
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            callbacks: 回调列表
            device: 设备 ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            max_epochs: 最大训练轮数
            gradient_clip_val: 梯度裁剪阈值（None 表示不裁剪）
            log_interval: 日志打印间隔（batch 数）
        """
        self.task = task
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.log_interval = log_interval
        
        # 设备设置
        self.device = get_device(device)
        print(f"Using device: {self.device}")
        
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
        self.stop_training = False  # 早停信号
        
        # 日志缓存
        self.train_logs: Dict[str, List[float]] = {}
        self.val_logs: Dict[str, List[float]] = {}
    
    def fit(self, model: nn.Module) -> Dict[str, Any]:
        """
        开始训练
        
        Args:
            model: 神经网络模型
            
        Returns:
            包含训练历史的字典
        """
        self.model = model.to(self.device)
        
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
        
        # 回调: 训练开始
        self.callbacks.on_train_begin(self)
        
        print(f"\nStart training for {self.max_epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        
        # 主训练循环
        for epoch in range(self.current_epoch, self.max_epochs):
            if self.stop_training:
                break
            
            self.current_epoch = epoch
            
            # 回调: epoch 开始
            self.callbacks.on_epoch_begin(self, epoch)
            
            # 训练 epoch
            train_logs = self._run_epoch("train")
            
            # 验证 epoch
            val_logs = {}
            if self.val_loader:
                val_logs = self._run_epoch("val")
            
            # 合并日志
            epoch_logs = {**{f"train_{k}": v for k, v in train_logs.items()},
                         **{f"val_{k}": v for k, v in val_logs.items()}}
            
            # 学习率调度（epoch 级）
            if self.scheduler is not None:
                if self.scheduler_monitor is not None:
                    # ReduceLROnPlateau 需要监控指标
                    metric = epoch_logs.get(self.scheduler_monitor)
                    if metric is not None:
                        self.scheduler.step(metric)
                else:
                    # 其他调度器直接 step
                    self.scheduler.step()
            
            # 回调: epoch 结束
            self.callbacks.on_epoch_end(self, epoch, epoch_logs)
        
        # 回调: 训练结束
        self.callbacks.on_train_end(self)
        
        print("\nTraining finished!")
        
        return {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
        }
    
    def _run_epoch(self, mode: str) -> Dict[str, float]:
        """
        执行一个 epoch 的训练或验证
        
        Args:
            mode: 'train' 或 'val'
            
        Returns:
            当前 epoch 的平均指标字典
        """
        if mode == "train":
            self.model.train()
            loader = self.train_loader
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            loader = self.val_loader
            torch.set_grad_enabled(False)
        
        # 累积指标
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        
        for batch_idx, batch in enumerate(loader):
            # 回调: batch 开始
            self.callbacks.on_batch_begin(self, batch_idx, mode)
            
            if mode == "train":
                # 训练模式：前向 + 反向 + 优化
                self.optimizer.zero_grad()
                step_result = self.task.training_step(self.model, batch, self.device)
                
                # 反向传播
                loss = step_result["loss"]
                loss.backward()
                
                # 梯度裁剪
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val,
                    )
                
                self.optimizer.step()
                self.global_step += 1
            else:
                # 验证模式：仅前向
                step_result = self.task.validation_step(self.model, batch, self.device)
            
            # 分离张量，转换为 Python float
            step_logs = {}
            for k, v in step_result.items():
                if isinstance(v, torch.Tensor):
                    step_logs[k] = v.detach().cpu().item()
                else:
                    step_logs[k] = float(v)
            
            # 累积指标
            for k, v in step_logs.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
                metric_counts[k] = metric_counts.get(k, 0) + 1
            
            # 回调: batch 结束
            if batch_idx % self.log_interval == 0:
                self.callbacks.on_batch_end(self, batch_idx, step_logs, mode)
        
        # 计算平均指标
        avg_metrics = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
        
        torch.set_grad_enabled(True)
        
        return avg_metrics
    
    def validate(self, model: nn.Module, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        单独运行验证
        
        Args:
            model: 神经网络模型
            val_loader: 验证数据加载器（None 则使用初始化时的）
            
        Returns:
            验证指标字典
        """
        self.model = model.to(self.device)
        loader = val_loader or self.val_loader
        
        if loader is None:
            raise ValueError("No validation loader provided")
        
        original_mode = self.model.training
        
        self.model.eval()
        torch.set_grad_enabled(False)
        
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        
        for batch in loader:
            step_result = self.task.validation_step(self.model, batch, self.device)
            
            for k, v in step_result.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item()
                metric_sums[k] = metric_sums.get(k, 0.0) + v
                metric_counts[k] = metric_counts.get(k, 0) + 1
        
        avg_metrics = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
        
        torch.set_grad_enabled(True)
        self.model.train(original_mode)
        
        return avg_metrics
    
    def predict(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        """
        模型预测（推理模式）
        
        Args:
            model: 神经网络模型
            batch: 输入批次数据
            
        Returns:
            模型输出
        """
        model = model.to(self.device)
        model.eval()
        
        images = batch["image"].to(self.device)
        
        with torch.no_grad():
            outputs = model(images)
        
        return outputs
    
    def save_checkpoint(self, path: str) -> None:
        """
        手动保存检查点
        
        Args:
            path: 保存路径
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict() if self.model else None,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
        }
        
        if self.scheduler and hasattr(self.scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: str, model: nn.Module, strict: bool = True) -> None:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            model: 模型实例
            strict: 是否严格匹配参数
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is None:
            self.model = model.to(self.device)
        
        if checkpoint.get("model_state_dict"):
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        
        if checkpoint.get("optimizer_state_dict") and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        
        print(f"Checkpoint loaded from: {path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
