# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 数据集规格包（schema / classes / scan / split 的统一出口）。
"""DVX 数据集规格包：只做 re-export（契约 P1-1）。

四个子模块各自负责一件事，都不移动/改名磁盘上的任何文件：

- ``schema``：``DVXDatasetSpec`` 与 ``load_spec``（yaml 解析 + 校验）；
- ``classes``：``ClassTable``（类别名 <-> label id，顺序即 label id）；
- ``scan``：``SampleRef`` 与 ``scan_samples``（按 ``layout`` 扫描磁盘）；
- ``split``：``SplitResult`` 与 ``build_splits``（内存中的非破坏性切分）。

用法::

    from dvisionix.data.spec import DVXDatasetSpec, load_spec, scan_samples, build_splits
"""

from .classes import ClassTable
from .scan import SampleRef, scan_samples
from .schema import DVXDatasetSpec, ParseConfig, SplitConfig, TaskSpec, load_spec
from .split import SplitResult, build_splits

__all__ = [
    "DVXDatasetSpec",
    "load_spec",
    "TaskSpec",
    "SplitConfig",
    "ParseConfig",
    "ClassTable",
    "SampleRef",
    "scan_samples",
    "SplitResult",
    "build_splits",
]
