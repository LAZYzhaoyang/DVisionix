# D:\ZhaoyangProject\DVisionix\dvisionix\utils\visualization.py

"""
可视化工具

集成 TensorBoard，实时显示训练曲线、指标、图像样本等。
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """
    TensorBoard 可视化器
    
    封装 TensorBoard 常用功能，简化训练过程的可视化。
    
    Examples:
        >>> visualizer = Visualizer(log_dir="./logs")
        >>> visualizer.log_scalar("loss/train", 0.5, step=1)
        >>> visualizer.log_image("sample", image_tensor, step=1)
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        comment: str = "",
    ):
        """
        初始化可视化器
        
        Args:
            log_dir: 日志根目录
            experiment_name: 实验名称（默认使用时间戳）
            comment: 附加注释
        """
        # 自动生成实验名称
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f"exp_{timestamp}"
        
        if comment:
            experiment_name = f"{experiment_name}_{comment}"
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.step = 0
        
        print(f"TensorBoard 日志目录: {self.log_dir}")
        print(f"启动命令: tensorboard --logdir {log_dir}")
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        记录标量值
        
        Args:
            tag: 标签名（如 "loss/train", "accuracy/val"）
            value: 标量值
            step: 步数，None 时使用内部计数器
        """
        if step is None:
            step = self.step
            self.step += 1
        
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        批量记录多个相关标量（在同一个图中显示）
        
        Args:
            main_tag: 主标签
            tag_scalar_dict: {子标签: 值} 字典
            step: 步数
        """
        if step is None:
            step = self.step
        
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        """
        记录单张图像
        
        Args:
            tag: 标签名
            image: 图像张量 (C, H, W) 或 (H, W)
            step: 步数
            normalize: 是否归一化到 [0, 1]
        """
        if step is None:
            step = self.step
        
        # 反归一化（如果图像经过了 ImageNet 风格的归一化）
        if normalize and image.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
        
        self.writer.add_image(tag, image, step)
    
    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        """
        记录一批图像（网格形式）
        
        Args:
            tag: 标签名
            images: 图像张量 (N, C, H, W)
            step: 步数
            normalize: 是否归一化
        """
        if step is None:
            step = self.step
        
        if normalize and images.shape[1] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = images * std + mean
            images = torch.clamp(images, 0, 1)
        
        self.writer.add_images(tag, images, step)
    
    def log_graph(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
    ) -> None:
        """
        记录模型计算图
        
        Args:
            model: PyTorch 模型
            input_tensor: 示例输入
        """
        self.writer.add_graph(model, input_tensor)
    
    def log_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float],
    ) -> None:
        """
        记录超参数和对应的指标
        
        Args:
            hparam_dict: 超参数字典
            metric_dict: 指标字典（如 {"hparam/accuracy": 0.9}）
        """
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def flush(self) -> None:
        """将缓冲区内容写入磁盘"""
        self.writer.flush()
    
    def close(self) -> None:
        """关闭写入器"""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()