# D:\ZhaoyangProject\DVisionix\dvisionix\metrics\base.py

"""
指标基类定义
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseMetric(ABC):
    """
    指标基类
    
    所有指标都继承此类，提供统一的接口。
    """
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    @abstractmethod
    def update(self, outputs: Any, targets: Any) -> None:
        """
        更新指标
        
        Args:
            outputs: 模型输出
            targets: 真实标签
        """
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """计算并返回指标值"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置所有统计数据"""
        pass
    
    def __call__(self, outputs: Any, targets: Any) -> float:
        """更新并返回当前指标值"""
        self.update(outputs, targets)
        return self.compute()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"