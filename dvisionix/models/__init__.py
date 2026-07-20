# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\models\\__init__.py

"""
模型模块

提供模型基类和各种任务的示例模型。
"""

from .base import (
    BaseModel,
    SimpleCNN,
    SimpleSegmentationModel,
    SimpleDetectionModel,
)

__all__ = [
    "BaseModel",
    "SimpleCNN",
    "SimpleSegmentationModel",
    "SimpleDetectionModel",
]
