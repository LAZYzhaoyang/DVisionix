# -*- coding: utf-8 -*-
"""
配置系统模块

统一的 YAML 配置管理，支持配置继承（_base_ 字段）、深度合并、
点号访问、字典访问、验证与序列化。
"""

from .config import Config

__all__ = ["Config"]
