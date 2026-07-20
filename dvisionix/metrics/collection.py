# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\metrics\\collection.py

"""
指标收集器

自动根据任务类型创建和更新指标。
"""

from typing import Dict, Any, Optional
from .classification import ClassificationMetrics
from .segmentation import SegmentationMetrics
from .detection import DetectionMetrics


class MetricCollection:
    """
    指标收集器
    
    根据任务类型自动创建和更新指标。
    """
    
    def __init__(
        self,
        task_type: str,
        num_classes: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            task_type: 任务类型 ('classification', 'segmentation', 'detection')
            num_classes: 类别数量
            **kwargs: 其他指标特定的参数
        """
        self.task_type = task_type
        self.num_classes = num_classes
        
        if task_type == "classification":
            self.metrics = ClassificationMetrics(num_classes=num_classes, **kwargs)
        elif task_type == "segmentation":
            self.metrics = SegmentationMetrics(num_classes=num_classes, **kwargs)
        elif task_type == "detection":
            self.metrics = DetectionMetrics(num_classes=num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def reset(self) -> None:
        """重置所有指标"""
        self.metrics.reset()
    
    def update(self, outputs: Any, batch: Dict[str, Any]) -> None:
        """
        更新指标
        
        Args:
            outputs: 模型输出
            batch: 批次数据
        """
        if self.task_type == "classification":
            self.metrics.update(outputs, batch["label"])
        elif self.task_type == "segmentation":
            self.metrics.update(outputs, batch["mask"])
        elif self.task_type == "detection":
            # 检测任务需要特殊处理
            pass
    
    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        return self.metrics.compute()
