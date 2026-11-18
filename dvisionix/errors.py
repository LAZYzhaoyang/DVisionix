# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVisionix 统一异常族（P0-1；不依赖 torch，可被任意层导入）。
"""DVisionix 统一异常族。

所有领域错误都继承 :class:`DVisionixError`，便于调用方统一捕获；
每条消息应包含"发生了什么 + 应该怎么办"（hint），``__str__`` 会把 hint
以 ``提示：`` 前缀拼接在第二行输出。

本模块只依赖标准库，**不得** import torch：它属于分层清单中的
infrastructure 层（level 4），是 registry / data / models / training 的共同底座。
"""

__all__ = [
    "DVisionixError",
    "ConfigError",
    "RegistryError",
    "DatasetError",
    "DatasetSpecError",
    "AnnotationError",
    "SplitError",
    "TaskModelMismatchError",
    "TaskDataMismatchError",
    "ModelOutputMismatchError",
    "CheckpointError",
    "InferenceError",
    "ExportError",
]


class DVisionixError(Exception):
    """所有 DVisionix 领域异常的基类。

    Args:
        message: 发生了什么（必填，建议包含出错的实体名或路径）。
        hint: 应该怎么办（可选，人类可读的修复建议）。
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        """返回带提示的错误文本；无 hint 时只返回 message。"""
        if self.hint:
            return f"{self.message}\n提示：{self.hint}"
        return self.message


class ConfigError(DVisionixError):
    """配置字段缺失、类型不符或取值非法。"""


class RegistryError(DVisionixError):
    """注册表查询/构建失败（未注册、重名、构造参数错误）。"""


class DatasetError(DVisionixError):
    """数据集相关错误的基类。"""


class DatasetSpecError(DatasetError):
    """数据集规格（dvx_dataset.yaml）不合法或无法加载。"""


class AnnotationError(DatasetError):
    """标注文件缺失、解析失败或标签不在类别表内。"""


class SplitError(DatasetError):
    """数据切分配置非法、路径重叠或切分结果不自洽。"""


class TaskModelMismatchError(DVisionixError):
    """配置的任务类型与模型 task_type 不一致。"""


class TaskDataMismatchError(DVisionixError):
    """配置的任务类型与数据集返回的样本结构不一致。"""


class ModelOutputMismatchError(DVisionixError):
    """模型输出字段、形状或 dtype 不符合任务契约。"""


class CheckpointError(DVisionixError):
    """检查点缺失、损坏或与当前模型结构不匹配。"""


class InferenceError(DVisionixError):
    """推理链路失败：输入图像不合法、预处理产物异常或结果文件无法读取/解析。

    典型场景：传入不支持的图像类型、图像路径不存在、预处理流水线未产出张量、
    ``Results`` JSON 损坏。调用方捕获后可回退到自己的图像读取/结果容错逻辑。
    """


class ExportError(DVisionixError):
    """模型导出（ONNX 等）失败或导出产物校验不通过。"""
