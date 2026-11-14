# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 多边形栅格化为分割掩码（labelme polygon -> (H, W) int64 mask）。
"""多边形栅格化：``polygons_to_mask``（契约 P1-6）。

分割数据集的标注是一组独立多边形（labelme ``polygon``），训练需要的却是一张
``(H, W)`` 的整型掩码。本模块只负责这一层转换：

- 背景整张填 ``background``（默认 0），因此空多边形列表天然得到"全背景"掩码；
- 每个多边形按标注顺序用 ``cv2.fillPoly`` 填充，**后写覆盖先写**：这与标注顺序一致，
  让"后标注的实例盖住先标注的实例"成为可预期的语义；
- 多边形坐标允许是浮点（labelme 原始坐标就是浮点），内部四舍五入后裁剪到图像范围，
  越界坐标既不会让 ``cv2`` 抛错，也不会在掩码之外留下痕迹；
- 返回值固定为 ``(H, W)`` 的 ``int64`` 数组，与 ``Sample`` 契约中 ``mask`` 的 dtype 一致。

实现说明：本环境的 ``cv2.fillPoly`` 不接受 ``int64`` 输出数组（OpenCV 5 重载解析失败），
因此先在 ``int32`` 画布上填充、最后整体 ``astype(np.int64)``。类别 id 远小于 ``int32``
上限，转换不丢信息。

用法::

    from dvisionix.data.parsers.rasterize import polygons_to_mask

    mask = polygons_to_mask(ref.segments, ref.segment_labels, 512, 512, background=255)
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from dvisionix.errors import AnnotationError

__all__ = ["polygons_to_mask"]

#: cv2 画布支持且足够表达类别 id 的整型宽度（OpenCV 5 不支持 int64 输出）
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


def polygons_to_mask(
    polygons: Sequence[np.ndarray],
    labels: Sequence[int],
    height: int,
    width: int,
    background: int = 0,
) -> np.ndarray:
    """把一组多边形按顺序栅格化成 ``(H, W)`` 的 ``int64`` 掩码。

    Args:
        polygons: 多边形顶点序列，每个元素形状为 ``(P, 2)``（x, y；浮点/整型均可）。
            空多边形（``(0, 2)`` 或 ``[]``）会被跳过，不影响其他多边形。
        labels: 与 ``polygons`` 等长的类别 id 序列；第 i 个元素填充第 i 个多边形。
        height: 图像高度（像素），必须为正整数。
        width: 图像宽度（像素），必须为正整数。
        background: 背景填充值，默认 0（常见用法：语义分割传 ``ignore_index``）。

    Returns:
        形状为 ``(height, width)`` 的 ``int64`` 掩码；多边形覆盖之外的像素等于 ``background``。

    Raises:
        AnnotationError: ``height`` / ``width`` 非法、``background`` / 类别 id 不是整数或
            超出 ``int32`` 范围、``polygons`` 与 ``labels`` 长度不一致、某个多边形不是
            ``(P, 2)`` 形状，或坐标含 NaN / inf。
    """
    height_value = _checked_int(height, "height", minimum=1)
    width_value = _checked_int(width, "width", minimum=1)
    background_value = _checked_int(background, "background")

    polygon_list = list(polygons)
    label_list = list(labels)
    if len(polygon_list) != len(label_list):
        raise AnnotationError(
            f"多边形数量（{len(polygon_list)}）与类别数量（{len(label_list)}）不一致",
            hint="segments 与 segment_labels 必须一一对应；请检查扫描/解析的输出",
        )

    # int32 画布：cv2.fillPoly 不支持 int64，最后统一转 int64
    canvas = np.full((height_value, width_value), background_value, dtype=np.int32)
    for position, (polygon, label) in enumerate(zip(polygon_list, label_list)):
        points = _polygon_points(polygon, position)
        if points.size == 0:
            continue
        label_value = _checked_int(label, f"labels[{position}]")
        # 先在浮点域裁剪到图像范围（像素下标属于 [0, W-1] / [0, H-1]），
        # 再取整转 int32：避免极大/极小坐标在 astype 时溢出成反号像素
        points[:, 0] = np.clip(points[:, 0], 0, width_value - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height_value - 1)
        canvas_points = np.rint(points).astype(np.int32)
        cv2.fillPoly(canvas, [canvas_points], label_value)

    return canvas.astype(np.int64)


def _polygon_points(polygon: object, position: int) -> np.ndarray:
    """把单个多边形转成 ``(P, 2)`` float64 数组（空多边形返回 ``(0, 2)``）。"""
    try:
        points = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnnotationError(
            f"第 {position} 个多边形无法转换为浮点数组: {polygon!r}",
            hint="每个多边形应是 (P, 2) 的顶点序列，例如 [[x1, y1], [x2, y2], ...]",
        ) from exc
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise AnnotationError(
            f"第 {position} 个多边形的形状应为 (P, 2)，当前为 {points.shape}",
            hint="每行是 (x, y) 顶点；若传入的是 (2P,) 扁平数组，请先 reshape(-1, 2)",
        )
    if not np.all(np.isfinite(points)):
        raise AnnotationError(
            f"第 {position} 个多边形含 NaN/inf 坐标",
            hint="请在标注工具中修正该多边形（坐标必须是有限数值）",
        )
    return points


def _checked_int(value: object, where: str, minimum: int | None = None) -> int:
    """把整型标量转成 ``int`` 并检查 ``int32`` 范围（可选下界）。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AnnotationError(
            f"{where} 必须是整数（当前为 {value!r}）",
            hint=f"{where} 用于掩码的尺寸/取值，不能是布尔、浮点或字符串",
        )
    number = int(value)
    if minimum is not None and number < minimum:
        raise AnnotationError(
            f"{where} 必须 >= {minimum}（当前为 {number}）",
            hint="掩码尺寸来自图片的 (H, W)，必须是正整数",
        )
    if number < _INT32_MIN or number > _INT32_MAX:
        raise AnnotationError(
            f"{where} 超出 int32 范围: {number}",
            hint="类别 id / 背景值应远小于 int32 上限；请检查 ignore_index 与类别表",
        )
    return number
