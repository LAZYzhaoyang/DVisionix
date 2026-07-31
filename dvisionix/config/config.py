# -*- coding: utf-8 -*-
"""
统一配置管理类

支持 YAML 加载、配置继承（_base_ 字段）、深度合并、点号访问、验证等功能。
"""

import os
import copy
from typing import Any, Dict, Optional, List, Union

import yaml


class Config:
    """
    统一配置管理类

    支持配置继承、点号访问、字典式访问、验证等功能。

    配置继承机制：
        在 YAML 文件中使用 `_base_` 字段指定父配置文件，
        当前配置会与父配置合并，当前配置的值优先。

    Examples:
        >>> cfg = Config.from_yaml("configs/classification/resnet50.yaml")
        >>> print(cfg.model.name)
        >>> print(cfg["training"]["batch_size"])
        >>> cfg = Config.from_default("classification")
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        object.__setattr__(self, "_config", config_dict or {})

    # ---------------- 创建配置 ----------------

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置，自动处理继承。"""
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        config_dict = cls._load_yaml_with_inheritance(path)
        return cls(config_dict)

    @classmethod
    def _load_yaml_with_inheritance(cls, path: str) -> Dict[str, Any]:
        """加载 YAML 并递归处理 `_base_` 继承。"""
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        base_files = config.pop("_base_", None)
        if base_files is None:
            return config

        if isinstance(base_files, str):
            base_files = [base_files]

        merged: Dict[str, Any] = {}
        config_dir = os.path.dirname(path)
        for base_file in base_files:
            base_path = base_file
            if not os.path.isabs(base_path):
                base_path = os.path.join(config_dir, base_file)
            base_config = cls._load_yaml_with_inheritance(base_path)
            merged = cls._deep_merge(merged, base_config)

        merged = cls._deep_merge(merged, config)
        return merged

    @classmethod
    def from_default(cls, task: str) -> "Config":
        """加载指定任务的默认配置（classification/detection/segmentation）。"""
        defaults_dir = os.path.join(os.path.dirname(__file__), "defaults")
        default_path = os.path.join(defaults_dir, f"{task}.yaml")
        if not os.path.exists(default_path):
            raise ValueError(
                f"No default config for task '{task}'. "
                f"Available: classification, detection, segmentation"
            )
        return cls.from_yaml(default_path)

    # ---------------- 合并逻辑 ----------------

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典，override 优先。"""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def merge(self, other: Union[Dict[str, Any], "Config"], override: bool = True) -> "Config":
        """合并其他配置，返回新的 Config。"""
        if isinstance(other, Config):
            other = other.to_dict()
        if override:
            merged = self._deep_merge(self._config, other)
        else:
            merged = self._deep_merge(other, self._config)
        return Config(merged)

    # ---------------- 访问接口 ----------------

    def __getattr__(self, name: str) -> Any:
        config = object.__getattribute__(self, "_config")
        if name in config:
            value = config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self._config[name] = value

    def __getitem__(self, key: str) -> Any:
        value = self._config[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        """安全获取配置值。"""
        return self._config.get(key, default)

    # ---------------- 序列化 ----------------

    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典（深拷贝）。"""
        return copy.deepcopy(self._config)

    def dump(self, path: str) -> None:
        """保存配置到 YAML 文件。"""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self._config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    # ---------------- 验证 ----------------

    def validate(self, required_keys: Optional[List[str]] = None) -> bool:
        """验证必填字段是否存在，支持点号路径（如 "model.name"）。"""
        if required_keys is None:
            required_keys = []
        for key in required_keys:
            if not self._has_nested_key(key):
                raise ValueError(f"Missing required config key: '{key}'")
        return True

    def _has_nested_key(self, key: str) -> bool:
        parts = key.split(".")
        current = self._config
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return True

    def validate_schema(self, task_type: Optional[str] = None) -> List[str]:
        """按任务 schema 校验配置，返回未知键/别名告警列表；类型/取值错误抛 ValueError。

        Args:
            task_type: 'classification' / 'detection' / 'segmentation'（可选）。

        Returns:
            告警字符串列表（未知键、便捷别名提示等）。
        """
        from .schema import validate_schema
        return validate_schema(self._config, task_type)



    # ---------------- CLI 覆盖 ----------------

    def update_from_cli(self, options: List[str]) -> "Config":
        """用命令行参数覆盖配置。

        支持 ``a.b.c=value`` 形式，value 会尝试解析为 int/float/bool/None，否则保留字符串。
        例如 ``["training.learning_rate=0.01", "data.image_size=128"]``。

        Args:
            options: ``key=value`` 字符串列表。

        Returns:
            self（便于链式调用）。
        """
        for item in options or []:
            if "=" not in item:
                raise ValueError(f"Invalid CLI override (missing '='): {item!r}")
            key, value = item.split("=", 1)
            key = key.strip()
            value = _parse_cli_value(value.strip())
            self._set_nested(key, value)
        return self

    def _set_nested(self, dotted_key: str, value: Any) -> None:
        """按点号路径写入嵌套字典。"""
        parts = dotted_key.split(".")
        node = self._config
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def __repr__(self) -> str:
        import json
        return f"Config({json.dumps(self._config, indent=2, ensure_ascii=False)})"



def _parse_cli_value(value: str) -> Any:
    """将 CLI 字符串值解析为 Python 类型。

    支持 bool / null / int / float，以及 YAML 子集（``[0, 1]``、``{"a": 1}``）。
    """
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    if value.startswith(("[", "{")):
        try:
            parsed = yaml.safe_load(value)
            if not isinstance(parsed, str):
                return parsed
        except yaml.YAMLError:
            pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_cli_options(options: List[str]) -> Dict[str, Any]:
    """将 ``a.b=v`` 列表解析为嵌套字典（供外部合并使用）。"""
    result: Dict[str, Any] = {}
    for item in options or []:
        if "=" not in item:
            raise ValueError(f"Invalid CLI override (missing '='): {item!r}")
        key, value = item.split("=", 1)
        parts = key.strip().split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_cli_value(value.strip())
    return result
