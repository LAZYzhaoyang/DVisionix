# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 推理预测器（P4-A2）：加载 checkpoint、前向、后处理与反变换。
"""推理预测器：把训练好的模型包装成「图片 -> :class:`~dvisionix.inference.results.Results`」。

本模块是 P4 推理链路的 A2 部分，契约见 ``docs/history/design/v2_p4_contract.md`` §3：

- :class:`BasePredictor`：设备解析、预处理、``eval`` + ``no_grad`` 前向、状态恢复；
- :class:`ClassificationPredictor`：softmax + top-k；
- :class:`DetectionPredictor`：复用模型自己的 ``decode``，再把框反变换回原图坐标；
- :class:`SegmentationPredictor`：argmax 后最近邻插值回原图尺寸；
- :func:`build_predictor`：按 ``task_type`` 构造对应子类。

设计约束：

- 预处理复用 :class:`~dvisionix.inference.preprocess.PreprocessPipeline`
  （即训练/验证用的 ``*Transforms(train=False)`` 确定性流水线），本模块**不重写**
  任何 resize / normalize；
- 后处理只做「任务语义」部分：分类概率、检测坐标反变换、分割上采样。检测框的
  阈值筛选与 NMS 全部交给模型自己的 ``decode``（与 ``DetectionTask.validation_step``
  同一契约），因此自定义检测器不需要改预测器；
- ``predict`` 结束后必须恢复模型原来的 ``training`` 状态，否则连续调用推理/训练
  会让 BatchNorm / Dropout 静默改变后续训练行为。

坐标约定：``PreprocessMeta.scale = (sh, sw) = 处理后尺寸 / 原图尺寸``，因此反变换
（processed -> original）是**除以**该比例，再 clip 到 ``[0, W-1] / [0, H-1]``，
保证 ``Results.boxes`` 永远落在原图像素范围内。
"""

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..errors import CheckpointError, ConfigError
from ..training.checkpoint import strip_wrapper_prefixes
from ..utils import get_device
from .preprocess import PreprocessMeta, PreprocessPipeline
from .results import Results

__all__ = [
    "BasePredictor",
    "ClassificationPredictor",
    "DetectionPredictor",
    "SegmentationPredictor",
    "build_predictor",
]


# ---------------------------------------------------------------------------
# 模块级工具
# ---------------------------------------------------------------------------
def _load_state_dict(checkpoint: str, *, trusted: bool = False) -> dict:
    """读取 checkpoint 文件并返回「已剥离包装前缀」的 state_dict。

    兼容三种形态：

    1. 纯 ``state_dict``（如 ``ema_last.pt``）；
    2. Trainer 完整 checkpoint（含 ``model_state_dict`` 键，其余键忽略）；
    3. DDP（``module.``）/ ``torch.compile``（``_orig_mod.``）包装过的键。

    Args:
        checkpoint: checkpoint 路径。
        trusted: 安全开关，``False``（默认）时用 ``torch.load(weights_only=True)``，
            只允许反序列化张量与基本容器。反序列化 pickle 等价于执行任意代码，
            因此来自第三方的权重默认按不可信处理；只有确认来源可信、且它确实
            携带非张量状态（优化器 / RNG / numpy 随机状态）时才传 ``True``。

    Returns:
        去掉包装前缀后的 ``{参数名: 张量}`` 字典。

    Raises:
        CheckpointError: 文件不存在、反序列化失败或内容不是 state_dict。
    """
    try:
        # 不以安全模式加载完整 Trainer checkpoint 时，会同时带回优化器 / RNG 状态，
        # 这里只要权重，所以统一先落到 CPU，避免在 GPU 上多占一份显存。
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=not trusted)
    except FileNotFoundError as exc:
        raise CheckpointError(
            f"checkpoint 不存在：{checkpoint}",
            hint="请检查路径是否拼写正确；相对路径以当前工作目录为基准。",
        ) from exc
    except Exception as exc:  # 反序列化失败 / 文件损坏 / 安全模式拒绝非张量状态
        raise CheckpointError(
            f"加载 checkpoint 失败：{checkpoint}（{type(exc).__name__}: {exc}）",
            hint="完整 Trainer checkpoint 含非张量状态（优化器 / RNG / numpy 等），"
            "安全模式（weights_only=True）无法加载，需要 trusted=True；"
            "如果确认文件来源可信仍失败，说明文件已损坏或不完整，请重新训练/导出。",
        ) from exc

    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise CheckpointError(
            f"checkpoint 的内容不是 state_dict：{checkpoint}（实际为 {type(state).__name__}）",
            hint="请传入模型保存的 state_dict，或含 model_state_dict 键的 Trainer checkpoint。",
        )
    return strip_wrapper_prefixes(state)


def _to_numpy(value: Any) -> np.ndarray:
    """把张量转成 CPU numpy 数组（非张量按 ``np.asarray`` 处理）。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resolve_scale(
    meta: PreprocessMeta,
    processed_hw: tuple[int, int],
    orig_h: int,
    orig_w: int,
) -> tuple[float, float]:
    """解析 ``(sh, sw)`` = 处理后 / 原图；缺失或非法时按尺寸比回退。

    ``PreprocessPipeline`` 的检测/分割预设会把 ``scale`` 写进 ``meta``；分类预设
    不写，由预处理层按「原图尺寸 vs 处理后尺寸」计算。这里再做一次兜底，
    避免上游字段缺失时反变换静默失效（直接除以 0 会得到 inf/nan 框）。
    """
    scale = getattr(meta, "scale", None)
    try:
        sh, sw = float(scale[0]), float(scale[1])
    except (TypeError, ValueError, IndexError):
        sh = sw = 0.0
    if sh <= 0.0 or sw <= 0.0:
        sh = processed_hw[0] / orig_h
        sw = processed_hw[1] / orig_w
    return sh, sw


# ---------------------------------------------------------------------------
# 预测器
# ---------------------------------------------------------------------------
class BasePredictor:
    """预测器基类：预处理 -> 前向 -> 任务后处理，返回统一 ``Results``。

    Args:
        model: 已构建的模型（``torch.nn.Module``）；构造时会搬到 ``device``。
        task_type: 任务类型，决定默认预处理流水线与后处理语义。
        device: 设备，``"auto"`` 时按 CUDA -> MPS -> CPU 自动选择。
        image_size: 预处理的正方形边长，必须与训练/验证时一致（默认 224）。
        class_names: 类别名列表（可选），会写入 ``Results.meta["class_names"]``，
            便于可视化与 JSON 落盘时保留语义。
        preprocess: 自定义预处理流水线；``None`` 时用
            ``PreprocessPipeline.from_task(task_type, image_size)`` 构建。
        **kwargs: 子类参数（顶层基类忽略），便于 ``from_checkpoint`` 透传。

    Attributes:
        model: 被包装的模型（已搬到 ``device``）。
        task_type: 任务类型字符串。
        device: 推理设备（``torch.device``）。
        image_size: 预处理边长。
        class_names: 类别名列表或 ``None``。
        preprocess: 预处理流水线实例。
    """

    def __init__(
        self,
        model,
        task_type: str,
        device="auto",
        image_size: int = 224,
        class_names: Sequence[str] | None = None,
        preprocess: PreprocessPipeline | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.task_type = str(task_type)
        self.device = get_device(device)
        self.image_size = int(image_size)
        self.class_names = None if class_names is None else [str(name) for name in class_names]
        self.preprocess = (
            preprocess
            if preprocess is not None
            else PreprocessPipeline.from_task(self.task_type, self.image_size)
        )
        # 模型与输入必须同设备：这里统一搬运，调用方无需再手动 .to(device)。
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        model,
        checkpoint: str,
        task_type: str,
        *,
        trusted: bool = False,
        **kwargs: Any,
    ) -> "BasePredictor":
        """加载 checkpoint 权重并构造预测器。

        支持的 checkpoint 形态：纯 ``state_dict``、含 ``model_state_dict`` 的
        Trainer 完整存档、以及 DDP / ``torch.compile`` 包装过的键
        （``module.`` / ``_orig_mod.`` 前缀会被剥离）。加载使用 ``strict=True``：
        键缺失或多出都视为「模型结构与权重不匹配」并报错，避免静默加载部分权重。

        Args:
            model: 待加载权重的模型实例（结构必须与 checkpoint 完全一致）。
            checkpoint: checkpoint 路径。
            task_type: 任务类型。
            trusted: 安全开关，默认 ``False``（``weights_only=True``）。完整
                Trainer checkpoint 含非张量状态，需要显式传 ``True``；只有确认
                文件来源可信时才这么做。
            **kwargs: 透传给 ``cls`` 的构造参数（``device`` / ``image_size`` /
                子类的 ``topk`` 等）。

        Returns:
            加载好权重的预测器实例（权重已搬到预测器的 ``device``）。

        Raises:
            CheckpointError: 文件不存在/损坏、反序列化失败、键与模型结构不匹配。
        """
        state = _load_state_dict(checkpoint, trusted=trusted)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise CheckpointError(
                f"checkpoint 与模型 {type(model).__name__} 的结构不匹配：{checkpoint}"
                f"（{type(exc).__name__}: {exc}）",
                hint="请确认 checkpoint 与模型配置来自同一次训练；完整 Trainer checkpoint "
                "会自动取 model_state_dict，若仍不匹配说明模型定义已变化"
                "（层名 / 类别数 / 是否带 DDP 包装等）。",
            ) from exc
        return cls(model, task_type, **kwargs)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def predict(self, image) -> Results:
        """对单张图片推理，返回 ``Results``。

        Args:
            image: ``str`` / ``Path``（路径）、``np.ndarray``（uint8 HWC，按 RGB 解释）
                或 ``PIL.Image``；类型校验由 ``PreprocessPipeline`` 负责。

        Returns:
            统一结果对象；``orig_size`` 与原图一致，框/掩码已回到原图坐标/尺寸。

        推理期间临时切到 ``eval`` 并禁用梯度，结束后**恢复**模型原来的
        ``training`` 状态（前向抛异常时同样恢复）。
        """
        tensor, meta = self.preprocess(image)
        tensor = tensor.to(self.device)
        processed_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                preds = self._forward(tensor)
                return self._postprocess(preds, meta, processed_hw)
        finally:
            self.model.train(was_training)

    def predict_many(self, inputs: Sequence) -> list[Results]:
        """逐张推理一批输入，返回 ``Results`` 列表。

        v1 不做跨图 batch 堆叠（不同原图尺寸的样本无法直接拼成一个张量），
        因此就是 ``[self.predict(item) for item in inputs]``。

        Args:
            inputs: 与 :meth:`predict` 单张输入同类型的序列。

        Returns:
            与 ``inputs`` 等长的 ``Results`` 列表。
        """
        return [self.predict(item) for item in inputs]

    # ------------------------------------------------------------------
    # 内部：前向与后处理
    # ------------------------------------------------------------------
    def _forward(self, images: torch.Tensor):
        """前向；``needs_batch`` 模型按 ``DetectionTask`` 约定附 ``batch={"image": ...}``。"""
        if getattr(self.model, "needs_batch", False):
            return self.model(images, batch={"image": images})
        return self.model(images)

    def _postprocess(
        self,
        preds: Any,
        meta: PreprocessMeta,
        processed_hw: tuple[int, int],
    ) -> Results:
        """任务后处理入口（子类实现）。"""
        raise NotImplementedError

    def _make_results(self, pre_meta: PreprocessMeta, **fields: Any) -> Results:
        """组装 ``Results``：统一填 ``task_type`` / ``orig_size`` / ``image_path`` / ``meta``。

        子类通过关键字参数传入任务字段（``probs`` / ``boxes`` / ``mask`` 等）；
        ``fields["meta"]`` 为额外元数据（如分割概率），与类别名合并后写入 ``Results.meta``。
        """
        results_meta = dict(fields.pop("meta", None) or {})
        if self.class_names is not None:
            results_meta.setdefault("class_names", list(self.class_names))
        return Results(
            task_type=self.task_type,
            orig_size=(int(pre_meta.original_size[0]), int(pre_meta.original_size[1])),
            image_path=getattr(pre_meta, "image_path", None),
            meta=results_meta,
            **fields,
        )


class ClassificationPredictor(BasePredictor):
    """分类预测器：``softmax(logits[0])`` -> top-k。

    Args:
        topk: ``Results.topk`` 的条目数，超过类别数时自动截断为 ``num_classes``。
        其余参数同 :class:`BasePredictor`。

    Raises:
        ConfigError: ``topk`` 小于 1。
    """

    def __init__(
        self,
        model,
        task_type: str = "classification",
        device="auto",
        image_size: int = 224,
        class_names: Sequence[str] | None = None,
        preprocess: PreprocessPipeline | None = None,
        topk: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            task_type,
            device=device,
            image_size=image_size,
            class_names=class_names,
            preprocess=preprocess,
            **kwargs,
        )
        if int(topk) < 1:
            raise ConfigError(
                f"topk 必须是正整数（当前为 {topk!r}）",
                hint="topk 表示返回的 top-k 类别数，至少为 1；超过类别数时自动截断。",
            )
        self.topk = int(topk)

    def _postprocess(
        self,
        preds: Any,
        meta: PreprocessMeta,
        processed_hw: tuple[int, int],
    ) -> Results:
        """softmax 后取 top-k；概率为 float32 ``(C,)``，topk 按概率降序。"""
        probs = torch.softmax(preds[0].float(), dim=0)
        k = min(self.topk, int(probs.numel()))
        values, indices = torch.topk(probs, k=k)
        topk = [
            (int(index), float(value)) for value, index in zip(values.tolist(), indices.tolist())
        ]
        return self._make_results(
            meta,
            probs=probs.detach().cpu().numpy().astype(np.float32),
            topk=topk,
        )


class DetectionPredictor(BasePredictor):
    """检测预测器：模型 ``decode`` -> 反变换回原图坐标 -> clip 到像素范围。

    阈值筛选 / NMS 全部由模型自己的 ``decode`` 完成（与训练侧验证步一致），
    本类只负责坐标反变换：``x / sw``、``y / sh``，再 clip 到
    ``[0, W-1] / [0, H-1]``。

    Args:
        score_threshold: 置信度阈值，透传给 ``model.decode``。
        iou_threshold: NMS 的 IoU 阈值，透传给 ``model.decode``。
        max_detections: 每张图最多保留的框数，透传给 ``model.decode``。
        其余参数同 :class:`BasePredictor`。
    """

    def __init__(
        self,
        model,
        task_type: str = "detection",
        device="auto",
        image_size: int = 224,
        class_names: Sequence[str] | None = None,
        preprocess: PreprocessPipeline | None = None,
        score_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            task_type,
            device=device,
            image_size=image_size,
            class_names=class_names,
            preprocess=preprocess,
            **kwargs,
        )
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)

    def _postprocess(
        self,
        preds: Any,
        meta: PreprocessMeta,
        processed_hw: tuple[int, int],
    ) -> Results:
        """decode -> 取 batch0 -> 坐标反变换 + clip，返回 ``Results``。"""
        boxes_list, scores_list, labels_list = self.model.decode(
            preds,
            processed_hw,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )
        boxes = _to_numpy(boxes_list[0]).astype(np.float32).reshape(-1, 4)
        scores = _to_numpy(scores_list[0]).astype(np.float32).reshape(-1)
        labels = _to_numpy(labels_list[0]).astype(np.int64).reshape(-1)

        orig_h, orig_w = int(meta.original_size[0]), int(meta.original_size[1])
        sh, sw = _resolve_scale(meta, processed_hw, orig_h, orig_w)
        # xyxy：x 除以水平比例，y 除以垂直比例；反变换回原图像素坐标。
        boxes = boxes / np.array([sw, sh, sw, sh], dtype=np.float32)
        # clip 到 [0, W-1] / [0, H-1]：decode 在「处理后坐标」里允许取到 img_w/img_h，
        # 反变换后是原图边界外（等于 W/H），而像素坐标最大值是 W-1/H-1。
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(max(orig_w - 1, 0)))
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(max(orig_h - 1, 0)))
        return self._make_results(meta, boxes=boxes, scores=scores, labels=labels)


class SegmentationPredictor(BasePredictor):
    """分割预测器：argmax -> 最近邻插值回原图尺寸。

    Args:
        return_probs: 为 ``True`` 时把 softmax 概率（原图尺寸 ``(C, H, W)``
            float32）写入 ``Results.meta["probs"]``（插值为双线性，各类概率和仍为 1）。
        其余参数同 :class:`BasePredictor`。
    """

    def __init__(
        self,
        model,
        task_type: str = "segmentation",
        device="auto",
        image_size: int = 224,
        class_names: Sequence[str] | None = None,
        preprocess: PreprocessPipeline | None = None,
        return_probs: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            task_type,
            device=device,
            image_size=image_size,
            class_names=class_names,
            preprocess=preprocess,
            **kwargs,
        )
        self.return_probs = bool(return_probs)

    def _postprocess(
        self,
        preds: Any,
        meta: PreprocessMeta,
        processed_hw: tuple[int, int],
    ) -> Results:
        """``argmax(1)`` 后在类别维取最近邻，回到原图 ``(H, W)`` int64 掩码。"""
        orig_h, orig_w = int(meta.original_size[0]), int(meta.original_size[1])
        extra: dict[str, Any] = {}
        if self.return_probs:
            probs = torch.softmax(preds.float(), dim=1)
            probs = F.interpolate(
                probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
            extra["probs"] = probs[0].detach().cpu().numpy().astype(np.float32)

        # 先 argmax 再上采样：nearest 插值只是复制类别索引，不会产生类别间的混合值。
        # 转 float 是因为 F.interpolate 不支持整型输入的多数后端组合。
        mask = preds.argmax(dim=1, keepdim=True).float()
        mask = F.interpolate(mask, size=(orig_h, orig_w), mode="nearest")
        mask = mask[0, 0].detach().cpu().to(torch.int64).numpy()
        return self._make_results(meta, mask=mask, meta=extra)


#: ``task_type`` -> 预测器子类；``build_predictor`` 的唯一事实来源。
_PREDICTOR_TYPES: dict[str, type[BasePredictor]] = {
    "classification": ClassificationPredictor,
    "detection": DetectionPredictor,
    "segmentation": SegmentationPredictor,
}


def build_predictor(model, task_type: str, **kwargs: Any) -> BasePredictor:
    """按任务类型构造预测器。

    Args:
        model: 已构建的模型。
        task_type: ``classification`` / ``detection`` / ``segmentation``
            （大小写与首尾空白不敏感）。
        **kwargs: 透传给预测器构造函数（``device`` / ``image_size`` /
            ``score_threshold`` / ``topk`` 等）。

    Returns:
        对应的预测器实例。

    Raises:
        ConfigError: 任务类型不在支持列表内（如 ``simclr`` / 多任务分类）。
    """
    key = str(task_type).strip().lower()
    predictor_cls = _PREDICTOR_TYPES.get(key)
    if predictor_cls is None:
        supported = " / ".join(sorted(_PREDICTOR_TYPES))
        raise ConfigError(
            f"不支持的推理任务类型：{task_type!r}（模型 {type(model).__name__}）",
            hint=f"当前支持的任务类型：{supported}；simclr 等自监督任务没有推理语义，"
            "多任务分类（MultiTaskClassifier）目前没有推理封装，"
            "如需推理请直接调用模型并用 task=... 取对应任务的 logits。",
        )
    return predictor_cls(model, task_type=key, **kwargs)
