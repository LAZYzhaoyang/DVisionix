# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVisionix: 深度学习算法库

"""
DVisionix: 深度学习算法库

一个模块化、可扩展的深度学习算法库，支持分类、检测、分割等多种任务。

核心特性：
- 统一的数据接口，支持所有任务
- 通用的训练引擎，支持自定义任务逻辑（Task 组件）
- 丰富的回调系统，支持灵活的训练控制
- Loss 作为模型层组件（models.losses），可继承、可自由组合
- 完整的指标计算，涵盖所有常见任务
- 多卡训练（DDP）、工作目录隔离、自动断点续训

快速开始（配置驱动）：
    python tools/train.py --config configs/classification/demo_synthetic.yaml

编程接口：
    from dvisionix.models import build_model
    from dvisionix.training import build_task, Trainer
"""

__version__ = "1.1.0"

# 子模块改为**按需导入**（PEP 562）。
#
# v1.0.0 在顶层无条件 `from . import config, data, export, metrics, models, training, utils`，
# 于是连 `from dvisionix.config import Config` 这种只想要配置的场景也会被迫加载
# torch / torchvision / cv2 等完整视觉栈（实测 import 时间相差一个数量级）。
#
# 现在：`import dvisionix` 不再拉起任何重依赖；访问 `dvisionix.models` 或
# `from dvisionix.models import build_model` 时才真正导入该子模块。
# 注意副作用：注册表（MODELS / HEADS / ...）在对应子模块被导入后才填充 ——
# 需要用到某个注册表时请显式导入对应子模块。
_SUBMODULES = ("config", "data", "export", "metrics", "models", "training", "utils")

#: 便捷导出 -> (提供该名字的子模块, 属性名)
_BUILDERS = {
    "build_model": ("models", "build_model"),
    "build_loss": ("models.losses", "build_loss"),
    "build_metric": ("metrics", "build_metric"),
    "build_dataset": ("data", "build_dataset"),
    "build_task": ("training", "build_task"),
}

#: 注册表对象：取用前必须先导入会向它们注册组件的子模块，
#: 否则会拿到一个空注册表（这是懒加载带来的语义变化，已在文档中说明）。
_REGISTRY_ATTRS = (
    "MODELS",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "LOSSES",
    "DATASETS",
    "TRANSFORMS",
    "METRICS",
    "TASKS",
    "Registry",
    "build_from_cfg",
)


def __getattr__(name: str):
    """按需导入子模块与便捷导出（PEP 562 模块级 __getattr__）。"""
    import importlib

    if name in _BUILDERS:
        module_name, attr = _BUILDERS[name]
        module = importlib.import_module(f".{module_name}", __name__)
        value = getattr(module, attr)
    elif name in _REGISTRY_ATTRS:
        # 组件的注册发生在各子模块被导入时，所以先导入它们再取注册表对象
        for submodule in ("data", "models", "metrics", "training"):
            importlib.import_module(f".{submodule}", __name__)
        from . import registry as _registry

        value = getattr(_registry, name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES) | set(_BUILDERS) | set(_REGISTRY_ATTRS))


__all__ = [
    "data",
    "models",
    "training",
    "metrics",
    "utils",
    "config",
    "export",
    "__version__",
    "Registry",
    "build_from_cfg",
    "MODELS",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "LOSSES",
    "DATASETS",
    "TRANSFORMS",
    "TASKS",
    "METRICS",
    "build_model",
    "build_task",
    "build_loss",
    "build_metric",
    "build_dataset",
]
