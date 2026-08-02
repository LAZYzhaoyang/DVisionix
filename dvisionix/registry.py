# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 组件注册表（Registry）
"""
组件注册表（Registry）

提供统一的组件注册 / 构建机制，使模型、数据集、变换、任务、
损失、指标等组件可以“注册即可用”，并支持从配置字典构建实例。

使用方式::

    from dvisionix.registry import MODELS

    @MODELS.register()
    class MyModel(...):
        ...

    model = MODELS.build({"type": "MyModel", "num_classes": 10})
"""

from typing import Any, Callable, Dict, Iterable, Optional


class Registry:
    """简单而通用的组件注册表。

    Args:
        name: 注册表名称（仅用于报错提示）。
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Callable[..., Any]] = {}

    @property
    def name(self) -> str:
        """注册表名称。"""
        return self._name

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def keys(self) -> Iterable[str]:
        """已注册的组件名列表（排序）。"""
        return self._registry.keys()

    def get(self, key: str) -> Callable[..., Any]:
        """按名称获取已注册组件；未注册时抛出 KeyError。"""
        if key not in self._registry:
            raise KeyError(
                f'"{key}" is not registered in registry "{self._name}". '
                f"Available: {sorted(self._registry.keys())}"
            )
        return self._registry[key]

    def register(
        self,
        obj: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        force: bool = False,
    ) -> Callable[..., Any]:
        """注册一个类/函数。可作为装饰器或直接调用。

        Examples:
            >>> @MODELS.register()
            ... class A: ...
            >>> @MODELS.register(name="B")
            ... class B: ...
            >>> MODELS.register(SomeClass)
        """

        def _do_register(target: Callable[..., Any]) -> Callable[..., Any]:
            key = name or getattr(target, "__name__", None)
            if key is None:
                raise ValueError("Cannot infer a name for the registered object.")
            if key in self._registry and not force:
                raise KeyError(f'"{key}" is already registered in registry "{self._name}".')
            self._registry[key] = target
            return target

        if obj is not None:
            return _do_register(obj)
        return _do_register

    def build(self, cfg: Dict[str, Any], **default_kwargs: Any) -> Any:
        """从配置字典构建实例。

                配置必须包含 `type` 字段（或
        ame`）指定注册名称，
                其余字段作为构造参数传入。`default_kwargs` 会被配置覆盖。
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, got {type(cfg)}")
        cfg = dict(cfg)
        obj_type = cfg.pop("type", None) or cfg.pop("name", None)
        if obj_type is None:
            raise KeyError('cfg must contain a "type" (or "name") field.')
        if isinstance(obj_type, str):
            builder = self.get(obj_type)
        elif callable(obj_type):
            builder = obj_type
        else:
            raise TypeError(f"cfg[type] must be str or callable, got {type(obj_type)}")

        kwargs = dict(default_kwargs)
        kwargs.update(cfg)
        return builder(**kwargs)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, items={sorted(self._registry.keys())})"


def build_from_cfg(cfg: Dict[str, Any], registry: Registry, **default_kwargs: Any) -> Any:
    """从配置字典和指定注册表构建实例的便捷函数。"""
    return registry.build(cfg, **default_kwargs)


# 全局注册表（各组件模块在导入时向其注册）
MODELS = Registry("models")
BACKBONES = Registry("backbones")
LAYERS = Registry("layers")
NECKS = Registry("necks")
HEADS = Registry("heads")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
TASKS = Registry("tasks")
LOSSES = Registry("losses")
METRICS = Registry("metrics")

__all__ = [
    "Registry",
    "build_from_cfg",
    "MODELS",
    "BACKBONES",
    "LAYERS",
    "NECKS",
    "HEADS",
    "DATASETS",
    "TRANSFORMS",
    "TASKS",
    "LOSSES",
    "METRICS",
]
