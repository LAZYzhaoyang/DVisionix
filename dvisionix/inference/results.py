# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 统一推理结果（Results）：JSON 往返与 numpy/cv2 可视化。
"""统一推理结果（Results）：JSON 往返与 numpy/cv2 可视化。

分类 / 检测 / 分割共用同一结果对象，并且所有几何量都已完成**反变换**：

- 分类：``probs`` (C,) 与 ``topk``；
- 检测：``boxes`` (N, 4) xyxy **原图坐标**，配合 ``scores`` / ``labels``；
- 分割：``mask`` (H, W) int64 **原图尺寸**。

调用方拿到 :class:`Results` 后不需要再关心预处理缩放；``to_json`` / ``save`` 落盘，
``load`` 还原，``visualize`` 用 OpenCV 画成 BGR uint8 图像（不依赖 matplotlib）。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

from ..errors import InferenceError

#: 类别调色板（BGR）；按类别 id 取模，保证同一类别在框/掩码/不同图片之间颜色一致
_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (56, 56, 255),
    (151, 157, 255),
    (49, 210, 207),
    (31, 119, 180),
    (255, 128, 0),
    (0, 200, 0),
    (0, 0, 255),
    (255, 255, 0),
    (128, 0, 128),
    (0, 165, 255),
    (203, 192, 255),
    (60, 20, 220),
    (0, 128, 128),
    (255, 0, 255),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 128),
    (64, 64, 64),
)

#: 无原图时的深色画布底色（BGR）
_DARK_BACKGROUND = 32


def _to_jsonable(value: Any) -> Any:
    """递归把 numpy / Path / tuple 等对象转成 JSON 可序列化的纯 Python 对象。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):  # 兼容 torch.Tensor 等第三方数组（不引入 torch 依赖）
        return value.tolist()
    return str(value)


def _task_kind(task_type: str) -> Optional[str]:
    """把任务类型归一化为可视化分支：classification / detection / segmentation。"""
    key = str(task_type).strip().lower().replace("-", "_").replace(" ", "_")
    if "class" in key:
        return "classification"
    if "detect" in key:
        return "detection"
    if "segment" in key or "mask" in key:
        return "segmentation"
    return None


def _class_name(label: int, class_names: Any) -> str:
    """类别 id -> 可显示名称；``class_names`` 越界或缺失时退回 id 字符串。"""
    if class_names is not None:
        try:
            index = int(label)
            if 0 <= index < len(class_names):
                return str(class_names[index])
        except (TypeError, ValueError):
            pass
    return str(int(label))


def _class_color(label: int) -> Tuple[int, int, int]:
    """类别 id -> 稳定的 BGR 颜色。"""
    return _PALETTE[int(label) % len(_PALETTE)]


def _read_bgr(path: Any) -> np.ndarray:
    """用 OpenCV 读取 BGR 图像；失败时抛 :class:`InferenceError`。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise InferenceError(
            f"无法读取图像：{path}",
            hint="确认文件存在、可读，且格式受 OpenCV 支持（jpg/png/bmp 等）",
        )
    return image


def _as_bgr_image(image: Any) -> np.ndarray:
    """把可视化底图统一成 BGR uint8 数组。

    约定：``np.ndarray`` 直接按 **BGR** 解释（与 cv2 一致，也与 ``visualize`` 的
    返回值一致）；``PIL.Image`` 会先 RGB→BGR；路径用 cv2 读取。
    """
    if isinstance(image, (str, Path)):
        return _read_bgr(image)
    if isinstance(image, PILImage.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = cv2.cvtColor(array[..., 0], cv2.COLOR_GRAY2BGR)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = array[..., :3]
        elif array.ndim != 3 or array.shape[2] != 3:
            raise InferenceError(
                f"不支持的图像形状：{array.shape}",
                hint="可视化底图应为 (H, W, 3) 的 uint8 数组，或传路径 / PIL.Image",
            )
        return np.array(array, dtype=np.uint8, copy=True, order="C")
    raise InferenceError(
        f"不支持的可视化图像类型：{type(image).__name__}",
        hint="支持 None（深色底 / image_path）、str / pathlib.Path、uint8 HWC BGR 数组、PIL.Image",
    )


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    """在图像上画一条带底色文本框（白色文字），自动限制在图像范围内。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    height, width = image.shape[:2]
    x = max(0, min(int(origin[0]), max(0, width - text_width - 4)))
    top = max(0, int(origin[1]) - text_height - 4)
    bottom = min(height - 1, top + text_height + baseline + 4)
    right = min(width - 1, x + text_width + 4)
    if bottom <= top:  # 极端小图：至少保证矩形非空
        bottom = min(height - 1, top + 1)
    cv2.rectangle(image, (x, top), (right, bottom), color, -1)
    text_y = max(text_height, min(height - 1, top + text_height + 1))
    cv2.putText(
        image,
        text,
        (x + 2, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


@dataclasses.dataclass
class Results:
    """统一推理结果。

    Attributes:
        task_type: 任务类型。
        orig_size: 原图 ``(H, W)``。
        image_path: 输入图像路径（可选）。
        probs: 分类概率 ``(C,)``。
        topk: 分类 top-k ``[(class_id, prob), ...]``，按概率降序。
        boxes: 检测框 ``(N, 4)`` xyxy，**原图坐标**。
        scores: 检测置信度 ``(N,)``。
        labels: 检测类别 ``(N,)``。
        mask: 分割掩码 ``(H, W)`` int64，**原图尺寸**。
        meta: 附加信息（如分割概率、耗时等）。
    """

    task_type: str
    orig_size: Tuple[int, int]
    image_path: Optional[str] = None
    # 分类
    probs: Optional[np.ndarray] = None
    topk: Optional[list] = None
    # 检测
    boxes: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    # 分割
    mask: Optional[np.ndarray] = None
    meta: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """规范化基础字段，保证序列化/可视化时拿到确定的类型。"""
        try:
            height, width = int(self.orig_size[0]), int(self.orig_size[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise InferenceError(
                f"Results.orig_size 必须是 (H, W)，实际为 {self.orig_size!r}",
                hint="orig_size 取原图高宽，例如 (480, 640)",
            ) from exc
        if height <= 0 or width <= 0:
            raise InferenceError(
                f"Results.orig_size 必须为正，实际为 {(height, width)}",
                hint="orig_size 取原图高宽，例如 (480, 640)",
            )
        self.orig_size = (height, width)
        self.task_type = str(self.task_type)
        if self.image_path is not None:
            self.image_path = str(self.image_path)
        if self.meta is None:
            self.meta = {}

    def to_dict(self) -> dict:
        """序列化为 JSON 友好字典：numpy -> list，且不包含原始图像数据。"""
        return {
            "task_type": self.task_type,
            "orig_size": [int(self.orig_size[0]), int(self.orig_size[1])],
            "image_path": self.image_path,
            "probs": _to_jsonable(self.probs),
            "topk": _to_jsonable(self.topk),
            "boxes": _to_jsonable(self.boxes),
            "scores": _to_jsonable(self.scores),
            "labels": _to_jsonable(self.labels),
            "mask": _to_jsonable(self.mask),
            "meta": _to_jsonable(dict(self.meta)),
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """序列化为 JSON 文本；给出 ``path`` 时同时写入该文件。

        Args:
            path: 可选输出路径；父目录不存在会自动创建。

        Returns:
            ``json.dumps`` 文本（indent=2, ensure_ascii=False）。
        """
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path is not None:
            target = Path(path)
            if target.parent and not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return text

    def save(self, path: str) -> None:
        """把结果写入 JSON 文件（等价于 ``to_json(path)``）。"""
        self.to_json(path)

    @classmethod
    def load(cls, path: str) -> "Results":
        """从 :meth:`save` / :meth:`to_json` 产物还原结果。

        Args:
            path: JSON 文件路径。

        Returns:
            还原后的 :class:`Results`。

        Raises:
            InferenceError: 文件不可读、JSON 损坏或字段结构非法。
        """
        target = Path(path)
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise InferenceError(
                f"无法读取结果 JSON：{target}",
                hint="确认文件存在且当前用户可读",
            ) from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InferenceError(
                f"结果 JSON 解析失败：{target}（第 {exc.lineno} 行第 {exc.colno} 列）",
                hint="确认文件由 Results.to_json / Results.save 生成且未被手工改坏",
            ) from exc
        if not isinstance(raw, dict):
            raise InferenceError(
                f"结果 JSON 根节点必须是对象，实际为 {type(raw).__name__}",
                hint="Results JSON 顶层应包含 task_type / orig_size 等字段",
            )
        try:
            orig = raw.get("orig_size") or (0, 0)
            orig_size = (int(orig[0]), int(orig[1]))
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            raise InferenceError(
                f"结果 JSON 的 orig_size 非法：{raw.get('orig_size')!r}",
                hint="orig_size 应为 [H, W] 两个正整数",
            ) from exc
        return cls(
            task_type=str(raw.get("task_type") or ""),
            orig_size=orig_size,
            image_path=raw.get("image_path"),
            probs=_load_array(raw.get("probs"), np.float64, "probs"),
            topk=_load_topk(raw.get("topk")),
            boxes=_load_array(raw.get("boxes"), np.float64, "boxes", columns=4),
            scores=_load_array(raw.get("scores"), np.float64, "scores"),
            labels=_load_array(raw.get("labels"), np.int64, "labels"),
            mask=_load_array(raw.get("mask"), np.int64, "mask"),
            meta=dict(raw.get("meta") or {}),
        )

    def visualize(self, image: Any = None, class_names: Any = None) -> np.ndarray:
        """把结果画到图像上，返回 BGR uint8 图像。

        Args:
            image: 可视化底图。``None`` 且有 ``image_path`` 时读原图；否则在深色底上画
                （尺寸取 ``orig_size``）。支持 ``str`` / ``Path`` / BGR ndarray / PIL.Image。
            class_names: 可选类别名列表；越界或缺失时显示类别 id。

        Returns:
            ``(H, W, 3) uint8`` BGR 图像。

        Raises:
            InferenceError: 底图或结果字段形状非法、图像无法读取。
        """
        base = self._base_image(image)
        task_kind = _task_kind(self.task_type)
        if task_kind == "classification":
            return self._draw_classification(base, class_names)
        if task_kind == "detection":
            return self._draw_detection(base, class_names)
        if task_kind == "segmentation":
            return self._draw_segmentation(base)
        # 未知任务类型：至少返回原图副本，保证"非空 BGR 图像"契约。
        return base

    # ------------------------------------------------------------------
    # 可视化内部实现
    # ------------------------------------------------------------------
    def _base_image(self, image: Any) -> np.ndarray:
        """准备与 ``orig_size`` 对齐的 BGR uint8 底图。"""
        height, width = self.orig_size
        if image is not None:
            base = _as_bgr_image(image)
        elif self.image_path:
            base = _read_bgr(self.image_path)
        else:
            base = np.full((height, width, 3), _DARK_BACKGROUND, dtype=np.uint8)
        if base.shape[0] != height or base.shape[1] != width:
            base = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(base, dtype=np.uint8)

    def _topk_entries(self) -> list:
        """取 top-k 列表；缺失时从 probs 推前 5 个。"""
        entries = []
        if self.topk:
            for item in self.topk:
                try:
                    entries.append((int(item[0]), float(item[1])))
                except (TypeError, ValueError, IndexError):
                    continue
            return entries
        if self.probs is not None:
            probs = np.asarray(self.probs, dtype=np.float64).reshape(-1)
            if probs.size:
                order = np.argsort(-probs)[:5]
                entries = [(int(index), float(probs[index])) for index in order]
        return entries

    def _draw_classification(self, base: np.ndarray, class_names: Any) -> np.ndarray:
        """左上角逐行写 top-k ``名称 概率`` 文本。"""
        entries = self._topk_entries()
        if not entries:
            return base
        for rank, (class_id, prob) in enumerate(entries):
            name = _class_name(class_id, class_names)
            text = f"{name}: {prob:.3f}"
            _draw_text(base, text, (4, 18 + rank * 20), _class_color(class_id))
        return base

    def _draw_detection(self, base: np.ndarray, class_names: Any) -> np.ndarray:
        """画检测框与 ``label score`` 文本（框坐标已是原图坐标）。"""
        if self.boxes is None:
            return base
        try:
            boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        except (TypeError, ValueError) as exc:
            raise InferenceError(
                f"Results.boxes 形状非法：{np.shape(self.boxes)}",
                hint="boxes 应为 (N, 4) 的 xyxy 原图坐标",
            ) from exc
        if boxes.size == 0:
            return base
        scores = None
        if self.scores is not None:
            scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        labels = None
        if self.labels is not None:
            labels = np.asarray(self.labels).reshape(-1)
        height, width = base.shape[:2]
        for index, box in enumerate(boxes):
            label = int(labels[index]) if labels is not None and index < labels.size else 0
            score = float(scores[index]) if scores is not None and index < scores.size else None
            color = _class_color(label)
            x1, y1, x2, y2 = (float(value) for value in box)
            x1 = int(round(min(max(x1, 0.0), width - 1.0)))
            y1 = int(round(min(max(y1, 0.0), height - 1.0)))
            x2 = int(round(min(max(x2, 0.0), width - 1.0)))
            y2 = int(round(min(max(y2, 0.0), height - 1.0)))
            if x2 <= x1:
                x2 = min(width - 1, x1 + 1)
            if y2 <= y1:
                y2 = min(height - 1, y1 + 1)
            cv2.rectangle(base, (x1, y1), (x2, y2), color, 2)
            text = _class_name(label, class_names)
            if score is not None:
                text = f"{text} {score:.2f}"
            _draw_text(base, text, (x1, max(y1, 18)), color)
        return base

    def _draw_segmentation(self, base: np.ndarray) -> np.ndarray:
        """按类别调色板给 mask 叠色（透明度 0.5）。"""
        if self.mask is None:
            return base
        mask = np.asarray(self.mask)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise InferenceError(
                f"Results.mask 形状非法：{mask.shape}",
                hint="mask 应为 (H, W) 的类别图，且已是原图尺寸",
            )
        height, width = base.shape[:2]
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST
            )
        color_layer = np.zeros_like(base)
        for value in np.unique(mask):
            color_layer[mask == value] = _class_color(int(value))
        return cv2.addWeighted(base, 0.5, color_layer, 0.5, 0)


def _load_array(
    value: Any,
    dtype: Any,
    field: str,
    columns: Optional[int] = None,
) -> Optional[np.ndarray]:
    """把 JSON 列表还原为数组；``columns`` 用于恢复 ``(N, C)`` 的空数组形状。"""
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise InferenceError(
            f"结果 JSON 字段 {field} 无法转为数组：{value!r}",
            hint=f"确认 {field} 是数值列表或 null",
        ) from exc
    if columns is not None:
        if array.size == 0:
            array = array.reshape(0, columns)
        elif array.ndim == 1 and array.size % columns == 0:
            array = array.reshape(-1, columns)
        elif array.ndim != 2 or array.shape[1] != columns:
            raise InferenceError(
                f"结果 JSON 字段 {field} 形状非法：{array.shape}",
                hint=f"{field} 应为 (N, {columns}) 的列表",
            )
    return array


def _load_topk(value: Any) -> Optional[list]:
    """把 JSON 中的 top-k 还原为 ``[(class_id, prob), ...]``。"""
    if value is None:
        return None
    if not isinstance(value, list):
        raise InferenceError(
            f"结果 JSON 字段 topk 应为列表，实际为 {type(value).__name__}",
            hint="topk 形如 [[class_id, prob], ...]",
        )
    entries = []
    for item in value:
        try:
            entries.append((int(item[0]), float(item[1])))
        except (TypeError, ValueError, IndexError) as exc:
            raise InferenceError(
                f"结果 JSON 字段 topk 条目非法：{item!r}",
                hint="topk 形如 [[class_id, prob], ...]",
            ) from exc
    return entries


__all__ = ["Results"]
