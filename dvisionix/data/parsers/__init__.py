# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 数据集解析器包（folder_per_class / labelme / 多边形栅格化）。
"""DVX 数据集解析器（契约 P1-6）。

本包只做三件事，彼此独立、都不移动磁盘上的任何文件：

- ``folder_per_class``（``parsers.folder``）：按"目录名即类别"扫描分类数据集；
- ``labelme``（``parsers.labelme``）：扫描/解析 labelme JSON，得到框与多边形；
- ``rasterize``（``parsers.rasterize``）：把多边形按顺序填成 ``(H, W)`` ``int64`` 掩码。

三者都返回 :class:`dvisionix.data.spec.scan.SampleRef`（rasterize 返回掩码），
统一由 :func:`dvisionix.data.spec.scan.scan_samples` 按 ``spec.layout`` 分派。
"""

from .folder import scan_folder_per_class
from .labelme import parse_labelme_file, scan_labelme
from .rasterize import polygons_to_mask

__all__ = [
    "parse_labelme_file",
    "polygons_to_mask",
    "scan_folder_per_class",
    "scan_labelme",
]
