# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: labelme 标注（.json）的扫描与解析（rectangle -> boxes，polygon -> segments）。
"""labelme 标注（``.json``）的扫描与解析（契约 P1-6）。

一个 labelme 样本 = 一张图片 + 与它**同目录同名**的 ``<stem>.json``::

    <root>/any/nested/dir/<stem>.jpg
    <root>/any/nested/dir/<stem>.json

解析规则（契约 P1-4）：

- ``rectangle`` → ``[x1, y1, x2, y2]``，取所有顶点的 min/max（兼容旋转矩形给出的 4 点）；
- ``polygon`` → ``(P, 2)`` ``float32`` 顶点数组，交由
  :func:`dvisionix.data.parsers.rasterize.polygons_to_mask` 栅格化；
  顶点少于 3 个的多边形无法栅格化，计入跳过的形状数；
- 其余形状类型（circle / line / point / linestrip / mask ...）跳过，并计入
  ``SampleRef.meta["skipped_shapes"]``；
- ``ignore_labels`` 中的 label 直接丢弃（不计入跳过数，属于"明确忽略"）；
- 未登记的 label → ``AnnotationError``，消息列出最多 5 个未知标签样例；
- ``imageWidth`` / ``imageHeight`` 写入 ``SampleRef.image_size``（``(H, W)``）；
  JSON 缺这两个字段时用 ``cv2`` 读取图片实际尺寸兜底并告警（不静默推断）；
- ``imageData`` 非空时按 ``spec.parse.labelme.image_data`` 处理：
  ``ignore`` 容忍并在 ``meta["image_data_present"]`` 记账，``error`` 直接报错；
- 缺失同名 ``.json`` 的图片**跳过**（不报错），扫描结束后汇总为**一次**
  ``UserWarning``，消息含跳过数量与最多 10 个示例相对路径；``"shapes": []`` 的 JSON
  仍作为**负样本**保留；
- JSON 语法错误、图片扩展名冲突、标注结构非法都抛 ``AnnotationError``
  （消息带 JSON/图片路径与原因）；
- 隐藏目录与隐藏文件（``.`` 开头）跳过；图片扩展名按 ``spec.parse.image_ext`` 小写比较。

``scan_labelme`` 按 ``spec.data_type`` 只保留目标形状：detection 只要 ``rectangle``，
segmentation 只要 ``polygon``（具体由 ``spec.parse.labelme.shape_types`` 的
``shape -> detection/segmentation`` 映射决定），另一种形状计入跳过数。

注意：跳过缺 JSON 只发生在**目录扫描**（``scan_labelme``）里；单文件 API
:func:`parse_labelme_file` 由调用方显式给出 JSON 路径，文件缺失仍抛 ``AnnotationError``。

用法::

    from dvisionix.data.parsers.labelme import scan_labelme
    from dvisionix.data.spec.classes import ClassTable
    from dvisionix.data.spec.schema import load_spec

    spec = load_spec("project/data/det/dvx_dataset.yaml")
    refs = scan_labelme(spec.root, spec, ClassTable(spec.class_names()))
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from dvisionix.data.parsers.folder import (
    check_stem_conflicts,
    iter_dirs,
    list_image_names,
    rel_path_posix,
)
from dvisionix.data.spec.classes import ClassTable
from dvisionix.data.spec.scan import SampleRef
from dvisionix.data.spec.schema import DVXDatasetSpec
from dvisionix.errors import AnnotationError, DatasetSpecError

__all__ = ["parse_labelme_file", "scan_labelme"]

#: labelme 形状类型 -> 由本模块产生的标注种类
_RECTANGLE = "rectangle"
_POLYGON = "polygon"


@dataclass
class _ParsedLabelme:
    """``_read_labelme`` 的解析结果（本模块私有的中间结构）。

    Args:
        polygons: 保留的 ``polygon`` 顶点数组，每个 ``(P, 2)`` ``float32``。
        polygon_labels: 与 ``polygons`` 等长的类别 id。
        boxes: 保留的 ``rectangle`` 框，每个 ``(4,)`` ``float32`` ``[x1, y1, x2, y2]``。
        box_labels: 与 ``boxes`` 等长的类别 id。
        skipped_shapes: 因形状类型不匹配或退化而跳过的形状数。
        image_size: ``(H, W)``，来自 ``imageHeight`` / ``imageWidth``；缺失为 None。
        image_data_present: ``imageData`` 是否非空（内嵌了图像数据）。
    """

    polygons: list[np.ndarray] = field(default_factory=list)
    polygon_labels: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)
    box_labels: list[int] = field(default_factory=list)
    skipped_shapes: int = 0
    image_size: tuple[int, int] | None = None
    image_data_present: bool = False


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def parse_labelme_file(
    json_path: str,
    class_table: ClassTable,
    ignore_labels: Sequence[str],
    allowed_shape_types: Sequence[str],
) -> tuple[list[np.ndarray], list[int], list[int], int]:
    """解析单个 labelme JSON（契约 P1-6 的签名）。

    Args:
        json_path: ``.json`` 文件路径。
        class_table: 类别表；``shape["label"]`` 必须能在其中查到。
        ignore_labels: 直接丢弃的 label（例如 ``__ignore__`` / ``__background__``）。
        allowed_shape_types: 参与解析的形状类型；其余类型计入跳过数。
            detection 传 ``("rectangle",)``，segmentation 传 ``("polygon",)``。

    Returns:
        四元组 ``(多边形列表, 多边形类别 id 列表, 矩形框类别 id 列表, 跳过的形状数)``：

        - 多边形列表：每个元素 ``(P, 2)`` ``float32``；
        - 多边形类别 id 列表：与多边形列表等长；
        - 矩形框类别 id 列表：与矩形框等长（**只有 id**；框坐标请用
          :func:`scan_labelme` 得到的 ``SampleRef.boxes``，详见 ``dvisionix.data.spec.scan``）；
        - 跳过的形状数：形状类型不匹配或退化（如顶点不足）的形状数。

    Raises:
        AnnotationError: 文件缺失/无法读取/JSON 语法错误/结构非法，或 label 未登记。
    """
    parsed = _read_labelme(json_path, class_table, ignore_labels, allowed_shape_types)
    return parsed.polygons, parsed.polygon_labels, parsed.box_labels, parsed.skipped_shapes


def scan_labelme(root: str, spec: DVXDatasetSpec, class_table: ClassTable) -> list[SampleRef]:
    """扫描 labelme 布局，返回按 ``rel_path`` 升序的样本列表（契约 P1-6）。

    只有 ``spec.data_type`` 对应的形状会被保留：detection 填 ``boxes`` / ``labels``，
    segmentation 填 ``segments`` / ``segment_labels``；另一种形状计入 ``skipped_shapes``。

    没有同名 ``<stem>.json`` 的图片会被**跳过**（不报错）：扫描结束后若存在跳过项，
    统一发**一次** ``UserWarning``，消息含数量与最多 10 个示例相对路径。
    ``"shapes": []`` 的 JSON 是合法负样本，仍会保留在返回值中。

    Args:
        root: 扫描根目录（通常是 ``spec.root``）。
        spec: 数据集规格，提供 ``data_type``、``parse`` 配置与 ``spec.root``。
        class_table: 类别表，``shape["label"]`` 必须能查到。

    Returns:
        样本列表（``rel_path`` 相对 ``spec.root``，posix 风格），按 ``rel_path`` 升序。
        每个样本的 ``meta`` 含 ``skipped_shapes``（跳过形状数）、``shape_count``
        （实际使用的形状数）与 ``image_data_present``（是否内嵌图像数据）。
        全部图片都缺 JSON 时返回空列表（同样告警），由 ``build_splits`` 报空数据集。

    Raises:
        AnnotationError: 根目录不存在、JSON 非法、未知标签、图片格式冲突等。
        DatasetSpecError: ``data_type`` 不是 detection/segmentation，或
            ``parse.labelme.shape_types`` 中没有属于该数据类型的形状。
    """
    root_path = os.path.abspath(os.fspath(root))
    if not os.path.isdir(root_path):
        raise AnnotationError(
            f"labelme 扫描根目录不存在或不是目录: {root_path}",
            hint="请检查 spec.root 是否正确",
        )
    allowed_shape_types = _allowed_shape_types(spec)
    ignore_labels = tuple(spec.parse.labelme_ignore_labels)
    image_data_mode = spec.parse.labelme_image_data
    extensions = {extension.lower() for extension in spec.parse.image_ext}

    refs: list[SampleRef] = []
    skipped_missing_json: list[str] = []
    image_data_count = 0
    # labelme 没有"类别目录"这一层：recursive=False 表示只看根目录下的图片，不进入任何子目录
    scan_dirs = iter_dirs(root_path, True) if spec.parse.recursive else [root_path]
    for dirpath in scan_dirs:
        names = list_image_names(dirpath, extensions)
        check_stem_conflicts(dirpath, names)
        for name in names:
            image_path = os.path.abspath(os.path.join(dirpath, name))
            json_path = os.path.splitext(image_path)[0] + ".json"
            if not os.path.isfile(json_path):
                # 缺失标注不是数据错误：跳过该图片，扫描结束后统一告警（见模块 docstring）
                skipped_missing_json.append(rel_path_posix(image_path, spec.root))
                continue
            parsed = _read_labelme(json_path, class_table, ignore_labels, allowed_shape_types)
            if parsed.image_data_present:
                if image_data_mode == "error":
                    raise AnnotationError(
                        f"labelme JSON 内嵌了 imageData（parse.labelme.image_data=error）: {json_path}",
                        hint="请在 labelme 中关闭'保存图像数据'后重新导出，"
                        "或把 parse.labelme.image_data 设为 ignore",
                    )
                image_data_count += 1
            image_size = _resolve_image_size(parsed.image_size, image_path, json_path)

            if spec.data_type == "detection":
                boxes: np.ndarray | None = np.asarray(parsed.boxes, dtype=np.float32).reshape(-1, 4)
                labels: np.ndarray | None = np.asarray(parsed.box_labels, dtype=np.int64)
                segments: list[np.ndarray] | None = None
                segment_labels: np.ndarray | None = None
                shape_count = len(parsed.box_labels)
            else:  # segmentation（_allowed_shape_types 已排除其他 data_type）
                boxes = None
                labels = None
                segments = list(parsed.polygons)
                segment_labels = np.asarray(parsed.polygon_labels, dtype=np.int64)
                shape_count = len(segments)

            refs.append(
                SampleRef(
                    image_path=image_path,
                    rel_path=rel_path_posix(image_path, spec.root),
                    boxes=boxes,
                    labels=labels,
                    segments=segments,
                    segment_labels=segment_labels,
                    ann_path=json_path,
                    image_size=image_size,
                    meta={
                        "skipped_shapes": parsed.skipped_shapes,
                        "shape_count": shape_count,
                        "image_data_present": parsed.image_data_present,
                    },
                )
            )

    if skipped_missing_json:
        warnings.warn(
            _missing_json_message(skipped_missing_json),
            UserWarning,
            stacklevel=2,
        )
    if image_data_count:
        warnings.warn(
            f"{image_data_count} 个 labelme JSON 内嵌了 imageData，"
            "已按 parse.labelme.image_data=ignore 忽略（计数见各样本 meta）",
            UserWarning,
            stacklevel=2,
        )
    refs.sort(key=lambda ref: ref.rel_path)
    return refs


def _missing_json_message(skipped: Sequence[str]) -> str:
    """构造"缺同名 JSON 已跳过"的告警消息（最多列 10 个示例相对路径）。"""
    preview = ", ".join(list(skipped)[:10])
    suffix = "；其余省略" if len(skipped) > 10 else ""
    return f"{len(skipped)} 张图片缺少同名 labelme JSON，已跳过（示例：{preview}{suffix}）"


# ---------------------------------------------------------------------------
# 解析实现
# ---------------------------------------------------------------------------


def _read_labelme(
    json_path: str,
    class_table: ClassTable,
    ignore_labels: Sequence[str],
    allowed_shape_types: Sequence[str],
) -> _ParsedLabelme:
    """读取并解析一个 labelme JSON，返回本模块私有的中间结构。"""
    data = _load_json(json_path)
    shapes = data.get("shapes", [])
    if shapes is None:
        shapes = []
    if not isinstance(shapes, list):
        raise AnnotationError(
            f"labelme JSON 的 shapes 必须是列表: {json_path}（当前为 {type(shapes).__name__}）",
            hint="请用 labelme 重新打开并保存该文件；shapes 缺失时应写成 []",
        )

    ignored = set(ignore_labels)
    allowed = set(allowed_shape_types)
    parsed = _ParsedLabelme(
        image_size=_image_size_from_json(data),
        image_data_present=bool(data.get("imageData")),
    )
    unknown: set[str] = set()
    for position, shape in enumerate(shapes):
        if not isinstance(shape, Mapping):
            raise AnnotationError(
                f"labelme JSON 的 shapes[{position}] 必须是映射: {json_path}"
                f"（当前为 {type(shape).__name__}）",
                hint="每个形状应形如 {label: 'cat', shape_type: 'polygon', points: [[x, y], ...]}",
            )
        label = shape.get("label")
        if not isinstance(label, str) or not label:
            raise AnnotationError(
                f"labelme JSON 的 shapes[{position}] 缺少合法的 label: {json_path}"
                f"（当前为 {label!r}）",
                hint="label 必须是非空字符串，且要能在类别表中查到",
            )
        if label in ignored:
            continue
        shape_type = shape.get("shape_type", _POLYGON)
        if not isinstance(shape_type, str) or not shape_type:
            raise AnnotationError(
                f"labelme JSON 的 shapes[{position}] 的 shape_type 非法: {json_path}"
                f"（当前为 {shape_type!r}）",
                hint="shape_type 应为 polygon / rectangle / circle / ... 之一",
            )
        if shape_type not in allowed:
            parsed.skipped_shapes += 1
            continue
        if not class_table.has(label):
            unknown.add(label)
            continue
        points = _shape_points(shape, position, shape_type, json_path)
        if shape_type == _RECTANGLE:
            if points.shape[0] < 2:
                raise AnnotationError(
                    f"labelme 矩形框顶点不足: {json_path} 的 shapes[{position}]"
                    f"（仅 {points.shape[0]} 个点，至少需要 2 个）",
                    hint="矩形框应给出 2 个角点（旋转矩形给 4 个）；" "请在 labelme 中修正该标注",
                )
            x1 = float(points[:, 0].min())
            y1 = float(points[:, 1].min())
            x2 = float(points[:, 0].max())
            y2 = float(points[:, 1].max())
            parsed.boxes.append(np.asarray([x1, y1, x2, y2], dtype=np.float32))
            parsed.box_labels.append(class_table.index(label))
        else:
            if points.shape[0] < 3:
                # 少于 3 个点无法栅格化，按"跳过"计（不是错误，可能是标注残留）
                parsed.skipped_shapes += 1
                continue
            parsed.polygons.append(points.astype(np.float32))
            parsed.polygon_labels.append(class_table.index(label))

    if unknown:
        samples = sorted(unknown)[:5]
        raise AnnotationError(
            f"labelme 标注含未登记类别（{len(unknown)} 个）: {samples}；文件: {json_path}",
            hint=f"已登记类别样例: {class_table.to_list()[:5]}（共 {len(class_table)} 类）；"
            "请检查标注 label 的拼写，或把类别加入 dvx_dataset.yaml 的 classes"
            "（顺序即 label id）",
        )
    return parsed


def _load_json(json_path: str) -> dict[str, Any]:
    """读取 labelme JSON 并保证顶层是映射。"""
    if not os.path.isfile(json_path):
        raise AnnotationError(
            f"labelme 标注文件不存在: {json_path}",
            hint="请确认 <stem>.json 与图片同目录同名",
        )
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise AnnotationError(
            f"无法读取 labelme 标注: {json_path}（{exc}）",
            hint="请检查文件权限与编码（要求 UTF-8）",
        ) from exc
    except json.JSONDecodeError as exc:
        raise AnnotationError(
            f"labelme JSON 解析失败: {json_path}（{exc.msg}，第 {exc.lineno} 行第 {exc.colno} 列）",
            hint="请用 labelme 重新打开并保存该文件，或用 jq 校验 JSON 语法",
        ) from exc
    if not isinstance(data, Mapping):
        raise AnnotationError(
            f"labelme JSON 顶层必须是映射: {json_path}（当前为 {type(data).__name__}）",
            hint="labelme 导出的 JSON 顶层是对象，应包含 shapes / imageWidth / imageHeight",
        )
    return dict(data)


def _shape_points(
    shape: Mapping[str, Any], position: int, shape_type: str, json_path: str
) -> np.ndarray:
    """把 ``shape["points"]`` 转成 ``(P, 2)`` float64 数组（校验结构）。"""
    raw = shape.get("points")
    if raw is None:
        raise AnnotationError(
            f"labelme JSON 的 shapes[{position}]（{shape_type}）缺少 points: {json_path}",
            hint="每个形状都必须给出点列表，例如 points: [[x1, y1], ...]",
        )
    try:
        points = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnnotationError(
            f"labelme JSON 的 shapes[{position}]（{shape_type}）的 points 无法解析: {json_path}",
            hint="points 必须是 [[x, y], ...] 形式的数值二维列表",
        ) from exc
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise AnnotationError(
            f"labelme JSON 的 shapes[{position}]（{shape_type}）的 points 形状应为 (P, 2): "
            f"{json_path}（当前为 {points.shape}）",
            hint="points 必须是 [[x, y], ...] 形式的数值二维列表",
        )
    if not np.all(np.isfinite(points)):
        raise AnnotationError(
            f"labelme JSON 的 shapes[{position}]（{shape_type}）含 NaN/inf 坐标: {json_path}",
            hint="请在 labelme 中修正该形状（坐标必须是有限数值）",
        )
    return points


def _image_size_from_json(data: Mapping[str, Any]) -> tuple[int, int] | None:
    """从 ``imageHeight`` / ``imageWidth`` 取 ``(H, W)``；缺失或非法返回 None。"""
    height = _positive_int(data.get("imageHeight"))
    width = _positive_int(data.get("imageWidth"))
    if height is None or width is None:
        return None
    return height, width


def _positive_int(value: Any) -> int | None:
    """把正整数（含 numpy 整型）转成 int；其他情况返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    number = int(value)
    return number if number > 0 else None


def _resolve_image_size(
    from_json: tuple[int, int] | None, image_path: str, json_path: str
) -> tuple[int, int] | None:
    """确定样本尺寸：优先用 JSON 的 imageWidth/imageHeight，缺失时用图片兜底并告警。"""
    if from_json is not None:
        return from_json
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        warnings.warn(
            f"labelme JSON 缺少 imageWidth/imageHeight，且无法读取图片尺寸: {json_path}",
            UserWarning,
            stacklevel=3,
        )
        return None
    size = (int(image.shape[0]), int(image.shape[1]))
    warnings.warn(
        f"labelme JSON 缺少 imageWidth/imageHeight，已用图片实际尺寸 {size} 兜底: {json_path}",
        UserWarning,
        stacklevel=3,
    )
    return size


def _allowed_shape_types(spec: DVXDatasetSpec) -> tuple[str, ...]:
    """按 ``data_type`` 过滤 ``parse.labelme.shape_types``，得到保留的形状类型。"""
    if spec.data_type not in ("detection", "segmentation"):
        raise DatasetSpecError(
            f"labelme 布局只服务 detection / segmentation（当前 data_type={spec.data_type!r}）",
            hint="分类请改用 layout: folder_per_class",
        )
    shape_types = spec.parse.labelme_shape_types
    allowed = tuple(shape for shape, target in shape_types.items() if target == spec.data_type)
    if not allowed:
        raise DatasetSpecError(
            f"parse.labelme.shape_types 中没有属于 {spec.data_type!r} 的形状",
            hint="请检查 shape_types，例如 {rectangle: detection, polygon: segmentation}",
        )
    return allowed
