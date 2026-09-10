# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: pytest 全局配置：统一仓库路径常量，避免各测试文件重复计算。
"""pytest 全局配置。

本文件只做两件事：

1. 把仓库根目录放入 ``sys.path``，使 ``tools`` / ``tests`` 可作为包被导入
   （此前各测试文件各自用 ``os.path.dirname`` 层层回溯计算路径）。
2. 提供仓库路径 fixture。

marker 注册在 ``pyproject.toml`` 的 ``[tool.pytest.ini_options]`` 中，
并启用 ``--strict-markers``：使用未注册的 marker 会直接报错，避免拼写错误被静默忽略。
"""

import os
import sys

import pytest

#: 仓库根目录（tests/conftest.py 的上两级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def repo_root() -> str:
    """仓库根目录绝对路径。"""
    return ROOT


@pytest.fixture(scope="session")
def configs_root() -> str:
    """官方配置根目录（仓库根 ``configs/``）绝对路径。"""
    return os.path.join(ROOT, "configs")
