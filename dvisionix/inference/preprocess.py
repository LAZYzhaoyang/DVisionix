# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 推理预处理：规格（PreprocessSpec）、元信息（PreprocessMeta）与确定性流水线。
"""推理预处理：规格、元信息与确定性流水线。

P4 的推理预处理与训练/验证**必须完全一致**。为了保证单一事实来源，本模块不实现
任何 resize / normalize，而是通过 :mod:`dvisionix.training.tasks` 中具体 Task 类的
``build_transforms(train=False, image_size=...)`` 取预设流水线——训练用哪套变换，
推理就用哪套。改预设只需改一处，不会出现"训练/推理两套实现悄悄漂移"。

典型用法::

    from dvisionix.inference import PreprocessPipeline

    prep = PreprocessPipeline.from_task("detection", 640)
    tensor, meta = prep("a.jpg")  # tensor: (1, 3, H, W) float32 CPU
    print(meta.original_size, meta.scale)

反变换约定：``meta.scale = (sh, sw)``，其中
``sh = 处理后高 / 原图高``、``sw = 处理后宽 / 原图宽``；
检测框 / 掩码按 ``x / sw``、``y / sh`` 即可映射回原图坐标。缩放比优先取流水线
（``BoxSyncResize``）写回 ``sample["meta"]`` 的值，缺失时才按尺寸比计算。
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from ..data.sample import Sample
from ..data.transforms import ImageNormalize
from ..errors import InferenceError

#: ImageNet 归一化均值（RGB 顺序，与训练预设一致）
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
#: ImageNet 归一化标准差（RGB 顺序）
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

#: 任务类型 -> (模块相对路径, Task 类名)；预设变换的唯一来源
_TASK_SOURCES = {
    "classification": ("..training.tasks.classification", "ClassificationTask"),
    "detection": ("..training.tasks.detection", "DetectionTask"),
    "segmentation": ("..training.tasks.segmentation", "SegmentationTask"),
    "simclr": ("..training.tasks.simclr", "SimCLRTask"),
}

#: task_type 别名 -> 规范任务名；MaskFormer 的 build_transforms 与分割一致
_TASK_ALIASES = {
    "classification": "classification",
    "multi_task_classification": "classification",
    "multi_label": "classification",
    "linear_eval": "classification",
    "detection": "detection",
    "segmentation": "segmentation",
    "maskformer": "segmentation",
    "mask_former": "segmentation",
    "instance_segmentation": "segmentation",
    "panoptic_segmentation": "segmentation",
    "simclr": "simclr",
}


def _canonical_task_type(task_type: str) -> str:
    """把任务类型（含别名）归一化为 ``classification/detection/segmentation/simclr``。

    Args:
        task_type: 调用方传入的任务类型，大小写与空格/连字符不敏感。

    Returns:
        规范任务名。

    Raises:
        InferenceError: 不支持的任务类型。
    """
    key = str(task_type).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = _TASK_ALIASES.get(key)
    if canonical is None:
        raise InferenceError(
            f"不支持的推理任务类型：{task_type!r}",
            hint=(
                "支持 classification / multi_task_classification / detection / "
                "segmentation / MaskFormer / simclr"
            ),
        )
    return canonical


def _build_task_pipeline(task_type: str, image_size: int):
    """按 Task 的 ``build_transforms(train=False)`` 构建确定性预设流水线。

    Args:
        task_type: 任务类型（支持别名）。
        image_size: 短边/目标尺寸，含义由具体任务预设决定。

    Returns:
        任务的验证态变换流水线（``TransformPipeline`` 或单个组合变换）。
    """
    canonical = _canonical_task_type(task_type)
    module_name, class_name = _TASK_SOURCES[canonical]
    module = importlib.import_module(module_name, package=__package__)
    task_cls = getattr(module, class_name)
    pipeline = task_cls.build_transforms(train=False, image_size=int(image_size))
    if pipeline is None:  # pragma: no cover - 预设实现异常时才会触发
        raise InferenceError(
            f"{class_name}.build_transforms(train=False) 返回了 None",
            hint="检查该 Task 的 build_transforms 实现",
        )
    return pipeline


def _iter_transforms(pipeline: Any) -> list:
    """把流水线展开为变换列表（单个 ``BaseTransform`` 视为只有一项）。"""
    transforms = getattr(pipeline, "transforms", None)
    if transforms is None:
        return [pipeline]
    return list(transforms)


def _float_tuple(value: Any, length: int, field: str) -> Tuple[float, ...]:
    """把 numpy 数组 / 列表统一转为定长 float 元组。

    预设 ``ImageNormalize`` 内部用 float32 存 mean/std，直接 ``float()`` 会带出
    ``0.48500001430511475`` 这类存储噪声；这里保留 6 位小数，既远高于实际精度需求，
    又保证 ``spec.mean == IMAGENET_MEAN`` 与配置往返稳定。
    """
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise InferenceError(
            f"无法解析预处理规格字段 {field}：{value!r}",
            hint=f"{field} 必须是长度为 {length} 的数值序列",
        ) from exc
    if array.size != length:
        raise InferenceError(
            f"预处理规格字段 {field} 长度应为 {length}，实际为 {array.size}",
            hint=f"请传入形如 (0.485, 0.456, 0.406) 的 {field}",
        )
    return tuple(round(float(item), 6) for item in array)


def _as_hw(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    """把 ``(H, W)`` 元信息转为 int 元组；缺失或非法时回退到 ``default``。"""
    if value is None:
        return (int(default[0]), int(default[1]))
    try:
        items = tuple(value)
    except TypeError:
        return (int(default[0]), int(default[1]))
    if len(items) != 2:
        return (int(default[0]), int(default[1]))
    try:
        height, width = int(items[0]), int(items[1])
    except (TypeError, ValueError):
        return (int(default[0]), int(default[1]))
    if height <= 0 or width <= 0:
        return (int(default[0]), int(default[1]))
    return height, width


def _as_scale(
    value: Any, original_size: Tuple[int, int], processed_size: Tuple[int, int]
) -> Tuple[float, float]:
    """解析缩放比；优先使用流水线写回的值，缺失时按尺寸比计算。"""
    if value is not None:
        try:
            items = tuple(value)
        except TypeError:
            items = ()
        if len(items) == 2:
            try:
                scale_h, scale_w = float(items[0]), float(items[1])
            except (TypeError, ValueError):
                scale_h = scale_w = 0.0
            if np.isfinite(scale_h) and np.isfinite(scale_w) and scale_h > 0 and scale_w > 0:
                return scale_h, scale_w
    orig_h, orig_w = original_size
    proc_h, proc_w = processed_size
    return proc_h / max(orig_h, 1), proc_w / max(orig_w, 1)


def _load_image(image: Any) -> Tuple[np.ndarray, Optional[str]]:
    """把支持的三类输入统一为 ``uint8 HWC RGB`` numpy 数组（返回副本）。

    Args:
        image: ``str`` / ``Path``（cv2 读取后 BGR→RGB）、``np.ndarray``（uint8 HWC，
            按 RGB 解释）、``PIL.Image.Image``。

    Returns:
        ``(rgb_array, image_path)``；非路径输入时 ``image_path`` 为 None。

    Raises:
        InferenceError: 类型不支持、路径不可读或数组形状/dtype 不合法。
    """
    if isinstance(image, (str, Path)):
        path = str(image)
        array = cv2.imread(path, cv2.IMREAD_COLOR)
        if array is None:
            raise InferenceError(
                f"无法读取图像：{path}",
                hint="确认文件存在、可读，且格式受 OpenCV 支持（jpg/png/bmp 等）",
            )
        return np.ascontiguousarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)), path

    if isinstance(image, PILImage.Image):
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
        return np.array(array, dtype=np.uint8, copy=True, order="C"), None

    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            raise InferenceError(
                f"不支持的 ndarray dtype：{array.dtype}",
                hint="推理输入必须是 uint8 HWC 图像（RGB 顺序）",
            )
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3:
            channels = array.shape[2]
            if channels == 1:
                array = np.repeat(array, 3, axis=2)
            elif channels == 4:
                array = array[..., :3]
            elif channels != 3:
                raise InferenceError(
                    f"不支持的 ndarray 通道数：{channels}",
                    hint="输入图像应为 HWC 且 C=1/3/4，推荐直接传 RGB 三通道 uint8",
                )
        else:
            raise InferenceError(
                f"不支持的 ndarray 形状：{array.shape}",
                hint="输入图像应为 uint8 (H, W, 3) 的 RGB 数组",
            )
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise InferenceError(
                f"图像尺寸非法：{array.shape[:2]}",
                hint="图像高宽必须为正数",
            )
        # 防御性拷贝：流水线里若出现就地修改也不会污染调用方数组。
        return np.array(array, dtype=np.uint8, copy=True, order="C"), None

    raise InferenceError(
        f"不支持的图像输入类型：{type(image).__name__}",
        hint="支持 str / pathlib.Path（磁盘路径）、uint8 HWC RGB numpy 数组、PIL.Image",
    )


def _extract_tensor(sample: Sample) -> torch.Tensor:
    """从变换后的 Sample 中取出图像张量。

    常规预设把结果写回 ``image``；SimCLR 双视角预设写 ``image1`` / ``image2``，
    因此按 ``image -> image1 -> image2`` 的顺序取第一个张量（推理只用一个视角）。
    """
    if isinstance(sample, dict):
        for key in ("image", "image1", "image2"):
            value = sample.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise InferenceError(
        "预处理流水线未产出图像张量",
        hint="确认 pipeline 末尾包含 ToTensor，且输入图像是 numpy uint8 数组",
    )


@dataclasses.dataclass(frozen=True)
class PreprocessSpec:
    """推理预处理规格——描述"输入如何变成网络张量"。

    Attributes:
        task_type: 任务类型（原样保留调用方传入的值，便于结果/日志回溯）。
        image_size: 目标尺寸；由预设变换的 ``size[0]``（或 ``image_size``）提取。
        mean: 归一化均值（RGB），默认 ImageNet。
        std: 归一化标准差（RGB），默认 ImageNet。
        scale: 像素缩放系数（uint8 -> [0, 1] 用 ``1/255``）。
    """

    task_type: str
    image_size: int
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD
    scale: float = 1.0 / 255.0

    def __post_init__(self) -> None:
        """规范化字段类型并做最小合法性校验（frozen dataclass 用 object.__setattr__）。"""
        try:
            image_size = int(self.image_size)
        except (TypeError, ValueError) as exc:
            raise InferenceError(
                f"PreprocessSpec.image_size 必须是正整数，实际为 {self.image_size!r}",
                hint="image_size 通常取配置里的 data.image_size",
            ) from exc
        if image_size <= 0:
            raise InferenceError(
                f"PreprocessSpec.image_size 必须为正，实际为 {image_size}",
                hint="image_size 应大于 0",
            )
        scale = float(self.scale)
        if not np.isfinite(scale) or scale <= 0:
            raise InferenceError(
                f"PreprocessSpec.scale 必须是正的有限数，实际为 {self.scale!r}",
                hint="默认 1/255；若自定义请确认归一化实现与训练一致",
            )
        object.__setattr__(self, "task_type", str(self.task_type))
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "mean", _float_tuple(self.mean, 3, "mean"))
        object.__setattr__(self, "std", _float_tuple(self.std, 3, "std"))
        object.__setattr__(self, "scale", scale)

    @classmethod
    def from_pipeline(cls, pipeline: Any, task_type: str = "") -> "PreprocessSpec":
        """从已有变换流水线提取规格。

        提取规则（契约 §1）：

        1. ``image_size``：取第一个带 ``.size`` 的变换的 ``size[0]``；若没有，则退回
           找带 ``.image_size`` 的组合变换（如 ``SimCLRTransforms``）。
        2. ``mean/std/scale``：取 ``ImageNormalize`` 的 numpy 参数；缺失时用 ImageNet
           默认值（``scale=1/255``）。若流水线没有 ``ImageNormalize`` 但存在其他带
           ``mean/std`` 的组合变换，也从它提取（等价兜底）。

        Args:
            pipeline: ``TransformPipeline`` / 单个 ``BaseTransform``。
            task_type: 任务类型；为空时尝试读 ``pipeline.task_type``。

        Returns:
            提取出的不可变规格。

        Raises:
            InferenceError: 无法提取 ``image_size`` 或归一化参数格式非法。
        """
        transforms = _iter_transforms(pipeline)

        image_size: Optional[int] = None
        for transform in transforms:
            size = getattr(transform, "size", None)
            if size is None:
                continue
            try:
                candidate = int(tuple(size)[0])
            except (TypeError, ValueError, IndexError):
                continue
            if candidate > 0:
                image_size = candidate
                break
        if image_size is None:
            # 组合变换（如 SimCLRTransforms）不暴露 .size，但自带 image_size。
            for transform in transforms:
                candidate = getattr(transform, "image_size", None)
                if candidate is None:
                    continue
                try:
                    image_size = int(candidate)
                except (TypeError, ValueError):
                    continue
                if image_size > 0:
                    break
                image_size = None
        if image_size is None:
            raise InferenceError(
                "无法从预处理流水线提取 image_size",
                hint="请确认 pipeline 含带 size 的 resize 变换，或显式构造 PreprocessSpec",
            )

        mean: Tuple[float, ...] = IMAGENET_MEAN
        std: Tuple[float, ...] = IMAGENET_STD
        scale = 1.0 / 255.0
        normalizer = next(
            (transform for transform in transforms if isinstance(transform, ImageNormalize)),
            None,
        )
        if normalizer is None:
            normalizer = next(
                (
                    transform
                    for transform in transforms
                    if hasattr(transform, "mean") and hasattr(transform, "std")
                ),
                None,
            )
        if normalizer is not None:
            mean = _float_tuple(normalizer.mean, 3, "mean")
            std = _float_tuple(normalizer.std, 3, "std")
            normalizer_scale = getattr(normalizer, "scale", None)
            if normalizer_scale is not None:
                try:
                    parsed_scale = float(normalizer_scale)
                except (TypeError, ValueError) as exc:
                    raise InferenceError(
                        f"无法解析 ImageNormalize.scale：{normalizer_scale!r}",
                        hint="scale 必须是正数（默认 1/255）",
                    ) from exc
                if parsed_scale > 0 and np.isfinite(parsed_scale):
                    scale = parsed_scale

        resolved_task_type = task_type or str(getattr(pipeline, "task_type", "") or "")
        return cls(
            task_type=resolved_task_type,
            image_size=image_size,
            mean=mean,
            std=std,
            scale=scale,
        )

    def to_dict(self) -> dict:
        """序列化为 JSON 友好的字典（tuple -> list）。"""
        return {
            "task_type": self.task_type,
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessSpec":
        """从 :meth:`to_dict` 产物（或等价字典）还原规格。

        Args:
            data: 至少包含 ``image_size``；其余字段缺失时使用默认值。

        Returns:
            还原后的规格。

        Raises:
            InferenceError: ``data`` 不是字典或缺少合法的 ``image_size``。
        """
        if not isinstance(data, dict):
            raise InferenceError(
                f"PreprocessSpec.from_dict 需要 dict，实际为 {type(data).__name__}",
                hint="传入 PreprocessSpec.to_dict() 的结果，或包含 image_size 的配置字典",
            )
        if "image_size" not in data:
            raise InferenceError(
                "PreprocessSpec 字典缺少 image_size",
                hint="至少提供 {'task_type': ..., 'image_size': ...}",
            )
        return cls(
            task_type=str(data.get("task_type") or ""),
            image_size=data["image_size"],
            mean=tuple(data.get("mean") or IMAGENET_MEAN),
            std=tuple(data.get("std") or IMAGENET_STD),
            scale=data.get("scale", 1.0 / 255.0),
        )


@dataclasses.dataclass
class PreprocessMeta:
    """预处理元信息——反变换与结果溯源所需的尺寸/缩放信息。

    Attributes:
        original_size: 原图 ``(H, W)``。
        processed_size: 网络输入 ``(H, W)``。
        scale: ``(sh, sw)``，处理后尺寸 / 原图尺寸。
        image_path: 输入来自磁盘时的路径，否则为 None。
    """

    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    scale: Tuple[float, float]
    image_path: Optional[str] = None

    def to_dict(self) -> dict:
        """序列化为 JSON 友好的字典。"""
        return {
            "original_size": [int(self.original_size[0]), int(self.original_size[1])],
            "processed_size": [int(self.processed_size[0]), int(self.processed_size[1])],
            "scale": [float(self.scale[0]), float(self.scale[1])],
            "image_path": self.image_path,
        }


class PreprocessPipeline:
    """确定性推理预处理流水线。

    与训练/验证共用 Task 预设变换；``__call__`` 返回可直接喂给模型的 CPU 张量与
    反变换元信息。

    Args:
        spec: 预处理规格（尺寸/归一化参数）。
        pipeline: 可选的既有变换流水线；为 None 时按 ``spec.task_type`` 从 Task
            预设（``train=False``）构建。
    """

    def __init__(self, spec: PreprocessSpec, pipeline: Any = None) -> None:
        if not isinstance(spec, PreprocessSpec):
            raise InferenceError(
                f"PreprocessPipeline 需要 PreprocessSpec，实际为 {type(spec).__name__}",
                hint="用 PreprocessSpec(...) 或 PreprocessSpec.from_pipeline(...) 构造规格",
            )
        self._spec = spec
        self._pipeline = (
            pipeline
            if pipeline is not None
            else _build_task_pipeline(spec.task_type, spec.image_size)
        )
        if not callable(self._pipeline):
            raise InferenceError(
                f"预处理 pipeline 不可调用：{type(self._pipeline).__name__}",
                hint="pipeline 必须是 TransformPipeline 或 BaseTransform 实例",
            )

    @classmethod
    def from_task(cls, task_type: str, image_size: int) -> "PreprocessPipeline":
        """按任务类型构建推理预处理流水线（复用 Task 预设，``train=False``）。

        Args:
            task_type: 任务类型，支持 classification / multi_task_classification /
                detection / segmentation / MaskFormer / simclr 及其别名。
            image_size: 目标尺寸，传给 ``build_transforms``。

        Returns:
            可直接调用的预处理流水线；``spec`` 从实际流水线提取，保证与预设一致。
        """
        pipeline = _build_task_pipeline(task_type, image_size)
        spec = PreprocessSpec.from_pipeline(pipeline, task_type=str(task_type))
        return cls(spec, pipeline)

    @property
    def spec(self) -> PreprocessSpec:
        """当前预处理规格。"""
        return self._spec

    def __call__(self, image: Any) -> Tuple[torch.Tensor, PreprocessMeta]:
        """执行确定性预处理。

        Args:
            image: ``str`` / ``Path``（磁盘图像，cv2 读取并 BGR→RGB）、``np.ndarray``
                （``uint8 HWC``，**按 RGB 解释**）或 ``PIL.Image.Image``。

        Returns:
            ``(tensor, meta)``：tensor 形状 ``(1, 3, H, W)``、dtype float32、在 CPU；
            meta 含原图/处理后尺寸、缩放比与可选路径。

        Raises:
            InferenceError: 输入类型/内容不合法，或流水线未产出图像张量。
        """
        array, image_path = _load_image(image)
        original_size = (int(array.shape[0]), int(array.shape[1]))

        # 每次都用全新 Sample / meta，避免上一次调用的残留影响确定性。
        sample = Sample({"image": array, "meta": {}})
        sample = self._pipeline(sample)

        tensor = _extract_tensor(sample)
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tensor.ndim != 3:
            raise InferenceError(
                f"预处理输出张量应为 (C, H, W)，实际为 {tuple(tensor.shape)}",
                hint="确认指定了正确的 task_type，且流水线末尾包含 ToTensor",
            )
        processed_size = (int(tensor.shape[1]), int(tensor.shape[2]))
        if processed_size[0] <= 0 or processed_size[1] <= 0:
            raise InferenceError(
                f"预处理输出尺寸非法：{processed_size}",
                hint="检查 resize/crop 尺寸配置",
            )

        meta = sample.get("meta") if isinstance(sample, dict) else None
        meta = meta if isinstance(meta, dict) else {}
        # 优先级：流水线写回的 original_size / scale > 按尺寸比计算。
        resolved_original = _as_hw(meta.get("original_size"), original_size)
        resolved_scale = _as_scale(meta.get("scale"), resolved_original, processed_size)

        return tensor.unsqueeze(0), PreprocessMeta(
            original_size=resolved_original,
            processed_size=processed_size,
            scale=resolved_scale,
            image_path=image_path,
        )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PreprocessSpec",
    "PreprocessMeta",
    "PreprocessPipeline",
]
