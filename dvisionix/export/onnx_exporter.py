# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: ONNX 模型导出器
"""
ONNX 模型导出器

支持：
- 单输入 / 多输入（input_shapes 列表或自定义 dummy_inputs）
- 输出自适应：Tensor / tuple / list / dict（dict 按键命名输出）
- 多输出 verify（onnxruntime 逐输出对比）
- trace（默认，零额外依赖）与 dynamo（可选，需 onnxscript）两种导出后端
- 任务感知：写入 transforms 归一化元数据（mean/std/scale）到 ONNX metadata_props
- P5：继承 :class:`~dvisionix.export.base.BaseExporter`，``export`` 返回
  :class:`~dvisionix.export.base.ExportResult`，``verify`` 返回
  :class:`~dvisionix.export.base.VerifyReport`（二者均兼容旧的路径/布尔用法）。
"""

import json
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils import get_logger
from .base import EXPORTERS, BaseExporter, ExportResult, VerifyReport

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型检查
    from ..inference.preprocess import PreprocessSpec

_logger = get_logger("dvisionix.export")


def _flatten_outputs(outputs: Any) -> Tuple[List[torch.Tensor], List[str], Dict[str, str]]:
    """递归把模型输出展平为 ``(有序 Tensor 列表, 输出名列表, 输出名 -> 路径映射)``。

    v1.0.0 只处理**一层**的 Tensor / dict / list / tuple：检测模型的典型输出
    ``{"boxes": [t0, t1], "scores": [...], "labels": [...]}`` 或
    ``(boxes_list, scores_list, labels_list)`` 里含有 list 时，会把 list 本身当成
    Tensor 传给 ``.numpy()`` 而直接崩；嵌套 dict 同理（CodePlan 7.1.2 D12）。

    名称由访问路径生成（``boxes_0`` / ``aux_cls_1``），并返回
    ``{输出名: "boxes[0]"}`` 形式的人类可读路径映射，便于落盘后回溯对应关系。
    """
    tensors: List[torch.Tensor] = []
    names: List[str] = []
    paths: Dict[str, str] = {}

    def visit(value: Any, name: str, path: str) -> None:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
            names.append(name or "output")
            paths[name or "output"] = path or "output"
            return
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{name}_{key}" if name else str(key)
                visit(item, child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                child = f"{name}_{index}" if name else f"output_{index}"
                visit(item, child, f"{path}[{index}]" if path else f"[{index}]")
            return
        raise TypeError(
            f"不支持的模型输出叶子类型: {type(value)}"
            f"（路径 {path or '<root>'}；期望 Tensor / dict / list / tuple）"
        )

    visit(outputs, "", "")
    if not tensors:
        raise ValueError("模型输出不包含任何 Tensor")
    return tensors, names, paths


@EXPORTERS.register(name="onnx")
class ONNXExporter(BaseExporter):
    """ONNX 模型导出器。

    Args:
        model: 待导出的 PyTorch 模型（forward 需可被 trace）。
        input_shape: 单输入模型输入形状（兼容旧 API），如 ``(3, 32, 32)``。
        input_shapes: 多输入模型输入形状列表，如 ``[(3, 32, 32), (8,)]``。
        dummy_inputs: 自定义 dummy 输入（Tensor 或 Tensor 元组/列表），非 image 输入用这个。
        device: 导出设备（'cpu' / 'cuda'）。
        task_type: 任务类型（'classification' / 'detection' / 'segmentation'，可选）。
        normalize: 归一化元数据，如 ``{"mean": [...], "std": [...], "scale": 1/255}``。
        preprocess: 显式推理预处理规格；缺省时按 ``task_type`` + ``normalize`` 推断。

    Examples:
        >>> exporter = ONNXExporter(model, input_shape=(3, 32, 32))
        >>> result = exporter.export("model.onnx", dynamic_batch=True)
        >>> assert exporter.verify(result)
    """

    format = "onnx"

    def __init__(
        self,
        model: nn.Module,
        input_shape: Optional[Tuple[int, ...]] = None,
        input_shapes: Optional[List[Tuple[int, ...]]] = None,
        dummy_inputs: Any = None,
        device: str = "cpu",
        task_type: Optional[str] = None,
        normalize: Optional[Dict[str, Any]] = None,
        preprocess: Optional["PreprocessSpec"] = None,
    ) -> None:
        # 不在构造期修改调用者的模型：v1.0.0 的 `model.to(device).eval()` 会让导出
        # 永久改变原模型的设备与 train/eval 状态。改为在 export/verify 内用
        # _temporary_eval() 临时切换并在 finally 中恢复。
        super().__init__(model)
        self.device = torch.device(device)
        self.task_type = task_type
        self.normalize = normalize
        self.preprocess = preprocess

        if dummy_inputs is not None:
            self.dummy_inputs = (
                dummy_inputs if isinstance(dummy_inputs, (list, tuple)) else (dummy_inputs,)
            )
        elif input_shapes is not None:
            self.dummy_inputs = tuple(torch.randn(1, *s, device=self.device) for s in input_shapes)
        elif input_shape is not None:
            self.dummy_inputs = (torch.randn(1, *input_shape, device=self.device),)
        else:
            raise ValueError("必须提供 input_shape / input_shapes / dummy_inputs 之一")

        self.single_input = len(self.dummy_inputs) == 1
        # 兼容旧 API：self.input_shape
        self.input_shape = tuple(self.dummy_inputs[0].shape[1:]) if self.single_input else None
        self.input_shapes = [tuple(t.shape[1:]) for t in self.dummy_inputs]

    # ------------------------------------------------------------------
    # 模型状态保护
    # ------------------------------------------------------------------
    def _model_device(self) -> torch.device:
        """推断模型当前所在设备（无参数/无 buffer 时退回 CPU）。"""
        for param in self.model.parameters():
            return param.device
        for buffer in self.model.buffers():
            return buffer.device
        return torch.device("cpu")

    @contextmanager
    def _temporary_eval(self):
        """临时把模型切到目标设备的 eval 模式，退出时**恢复**原设备与 train/eval 状态。

        v1.0.0 在 ``__init__`` 里直接 ``model.to(device).eval()``：导出一次就会永久
        改变调用者的模型（搬到 CPU、关掉训练模式），调用方毫无察觉
        （CodePlan 7.1.2 D12）。
        """
        device_before = self._model_device()
        training_before = self.model.training
        try:
            self.model.to(self.device)
            self.model.eval()
            yield
        finally:
            self.model.to(device_before)
            self.model.train(training_before)

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def export(
        self,
        output_path: str,
        dynamic_batch: Optional[bool] = None,
        dynamic_size: Optional[bool] = None,
        opset_version: int = 17,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        simplify: bool = False,
        backend: str = "trace",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """导出模型为 ONNX 格式，返回导出结果。

        Args:
            dynamic_batch: 导出动态 batch。``None``（默认）时按后端取默认值：
                trace → ``True``（历史行为），dynamo → ``False``。
            dynamic_size: 导出动态高宽，默认 ``False``。
            backend: 'trace'（默认）或 'dynamo'（需安装 onnxscript）。
            metadata: 附加元数据，与 normalize 合并写入 ONNX metadata_props。

        Returns:
            :class:`~dvisionix.export.base.ExportResult`；``str(result)`` 与
            ``os.fspath(result)`` 均为导出文件路径，兼容旧的字符串用法。
        """
        if backend not in ("trace", "dynamo"):
            raise ValueError(f"backend 必须是 'trace' 或 'dynamo'，当前: {backend!r}")

        # dynamo 用 dynamic_shapes 而非 dynamic_axes，因此 `export(path, backend="dynamo")`
        # 这个最自然的默认调用此前必然抛 ValueError（dynamic_batch 默认 True）。
        # 这里用 None 哨兵区分"用户没写"与"用户显式要求"：默认按后端取合理值，
        # 只有显式要求动态维度时才报错，避免静默丢掉动态 batch 的承诺。
        if dynamic_batch is None:
            dynamic_batch = backend != "dynamo"
        if dynamic_size is None:
            dynamic_size = False

        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)

        # 探测输出结构（dict / 多输出 / 嵌套）
        with self._temporary_eval(), torch.no_grad():
            outputs = self.model(*self.dummy_inputs)
        out_tensors, default_output_names, output_paths = _flatten_outputs(outputs)

        n_inputs = len(self.dummy_inputs)
        input_names = input_names or (
            [f"input_{i}" for i in range(n_inputs)] if n_inputs > 1 else ["input"]
        )
        output_names = output_names or default_output_names
        if len(output_names) != len(out_tensors):
            raise ValueError(
                f"output_names 数量 ({len(output_names)}) 与模型输出数 ({len(out_tensors)}) 不一致"
            )

        dynamic_axes: Dict[str, Dict[int, str]] = {}
        if dynamic_batch:
            for name in input_names:
                dynamic_axes[name] = {0: "batch_size"}
            for name in output_names:
                dynamic_axes.setdefault(name, {})[0] = "batch_size"
        if dynamic_size:
            for i, name in enumerate(input_names):
                shape = self.input_shapes[i]
                if len(shape) >= 3:  # (C, H, W, ...)
                    axes = dynamic_axes.setdefault(name, {})
                    axes[2] = "height"
                    axes[3] = "width"

        args = self.dummy_inputs[0] if self.single_input else self.dummy_inputs

        if backend == "dynamo":
            # 参数兼容性先判：dynamo 导出改用 dynamic_shapes 描述动态维度，
            # 静默忽略 dynamic_axes 会让「动态 batch」的承诺失效（CodePlan 7.1.2 D12）。
            if dynamic_axes:
                raise ValueError(
                    "backend='dynamo' 不支持 dynamic_axes —— dynamo 导出需用 dynamic_shapes "
                    "描述动态维度。请显式传 dynamic_batch=False, dynamic_size=False，"
                    "或改用 backend='trace'。（另：dynamo 后端还需额外安装 onnxscript）"
                )
            try:
                import onnxscript  # noqa: F401
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "backend='dynamo' 需要安装 onnxscript：pip install onnxscript"
                ) from exc
            torch.onnx.export(
                self.model,
                args,
                output_path,
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamo=True,
            )
        else:
            with self._temporary_eval():
                torch.onnx.export(
                    self.model,
                    args,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes if dynamic_axes else None,
                    dynamo=False,
                )

        dummy_input_shapes = [list(t.shape) for t in self.dummy_inputs]
        result = ExportResult(
            path=output_path,
            format="onnx",
            input_names=list(input_names),
            input_shapes=dummy_input_shapes,
            output_names=list(output_names),
            opset=opset_version,
            dynamic_axes=dict(dynamic_axes) if dynamic_axes else None,
            preprocess=self._resolve_preprocess(dummy_input_shapes),
            extra={
                "backend": backend,
                "simplify": simplify,
                "output_path_map": output_paths,
            },
        )

        # 契约 §2：把完整的 ExportResult 写入 ONNX metadata_props 的
        # `dvisionix_export` 键；同时保留原有 normalize_* 与 output_path_map 键。
        self._write_metadata(
            output_path,
            metadata,
            extra={
                "output_path_map": output_paths,
                "dvisionix_export": result.to_dict(),
            },
        )
        _logger.info(f"[OK] ONNX model exported: {output_path}")

        if simplify:
            self._simplify(output_path)

        return result

    def _resolve_preprocess(self, input_shapes: List[List[int]]) -> Optional["PreprocessSpec"]:
        """解析本次导出应记录的预处理规格。

        优先级（契约 §2）：显式传入的 ``preprocess`` > 由 ``task_type`` +
        ``normalize`` 推断 > ``None``。

        Args:
            input_shapes: dummy 输入的完整形状（含 batch 维）。

        Returns:
            预处理规格；没有足够信息时为 None。

        Raises:
            ValueError: 设置了 ``task_type`` 但第一个输入不是图像（不足 2 维）。
        """
        if self.preprocess is not None:
            return self.preprocess
        if not self.task_type:
            return None

        from ..inference.preprocess import IMAGENET_MEAN, IMAGENET_STD, PreprocessSpec

        shape = input_shapes[0]
        if len(shape) < 2:
            raise ValueError(
                f"task_type={self.task_type!r} 需要图像输入（至少 2 维），"
                f"但第一个输入形状为 {shape}"
            )
        normalize = self.normalize or {}
        return PreprocessSpec(
            task_type=str(self.task_type),
            image_size=int(shape[-2]),
            mean=tuple(normalize.get("mean") or IMAGENET_MEAN),
            std=tuple(normalize.get("std") or IMAGENET_STD),
            scale=normalize.get("scale", 1.0 / 255.0),
        )

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------
    def _write_metadata(
        self,
        onnx_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        props: Dict[str, Any] = {}
        if self.normalize:
            props.update({f"normalize_{k}": json.dumps(v) for k, v in self.normalize.items()})
        if metadata:
            props.update(
                {
                    str(k): json.dumps(v) if not isinstance(v, str) else v
                    for k, v in metadata.items()
                }
            )
        if extra:
            props.update({str(k): json.dumps(v) for k, v in extra.items()})
        if not props:
            return
        try:
            import onnx

            model = onnx.load(onnx_path)
            for key, value in props.items():
                prop = model.metadata_props.add()
                prop.key = key
                prop.value = str(value)
            onnx.save(model, onnx_path)
        except ImportError:  # pragma: no cover
            _logger.warning("onnx 未安装，跳过 metadata 写入")

    # ------------------------------------------------------------------
    # 简化（可选）
    # ------------------------------------------------------------------
    def _simplify(self, onnx_path: str) -> None:
        try:
            import onnx
            from onnxsim import simplify

            model_onnx = onnx.load(onnx_path)
            model_simplified, check = simplify(model_onnx)
            if check:
                onnx.save(model_simplified, onnx_path)
                _logger.info(f"[OK] ONNX model simplified: {onnx_path}")
            else:
                _logger.warning("[WARN] ONNX simplify check failed, keep original model")
        except ImportError:  # pragma: no cover
            _logger.warning("[WARN] onnxsim not installed, skip simplify. pip install onnxsim")

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    def verify(
        self,
        result_or_path: Any,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        num_samples: int = 3,
    ) -> VerifyReport:
        """验证 ONNX 输出与 PyTorch 输出的一致性（多输入/多输出）。

        Args:
            result_or_path: :class:`~dvisionix.export.base.ExportResult` 或 ONNX 路径；
                传入 ExportResult 时会把全局最大误差回填到
                ``result.parity_max_abs_diff``。
            rtol: 相对容差。
            atol: 绝对容差。
            num_samples: 重复校验的样本轮数。

        Returns:
            :class:`~dvisionix.export.base.VerifyReport`；``bool(report)`` 等价于
            ``report.passed``，既有 ``assert exporter.verify(path)`` 语义不变。
            ``outputs`` 为逐输出明细（跨样本汇总），``max_abs_diff`` 为全局最大误差。
        """
        result = result_or_path if isinstance(result_or_path, ExportResult) else None
        onnx_path = result.path if result is not None else os.fspath(result_or_path)

        try:
            import onnxruntime as ort
        except ImportError:  # pragma: no cover
            _logger.warning(
                "[WARN] onnxruntime not installed, skip verify. pip install onnxruntime"
            )
            return VerifyReport(passed=False)

        import numpy as np

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_input_names = [inp.name for inp in session.get_inputs()]
        onnx_output_names = [out.name for out in session.get_outputs()]

        all_passed = True
        global_max_diff: Optional[float] = None
        details: Dict[str, Dict[str, Any]] = {}

        with self._temporary_eval():
            for sample_idx in range(num_samples):
                dummy = tuple(t.clone() for t in self.dummy_inputs)
                with torch.no_grad():
                    torch_outs = _flatten_outputs(self.model(*dummy))[0]

                # CUDA / 需要梯度的张量必须 detach 且搬到 CPU 才能转 numpy
                feed = {name: t.detach().cpu().numpy() for name, t in zip(onnx_input_names, dummy)}
                onnx_outs = session.run(None, feed)

                if len(torch_outs) != len(onnx_outs):
                    raise ValueError(
                        f"输出数量不一致：PyTorch {len(torch_outs)} 个，ONNX {len(onnx_outs)} 个。"
                        f"（zip 会静默截断，因此这里直接报错）"
                    )

                for out_idx, (torch_out, onnx_out) in enumerate(zip(torch_outs, onnx_outs)):
                    onnx_name = (
                        onnx_output_names[out_idx]
                        if out_idx < len(onnx_output_names)
                        else f"#{out_idx}"
                    )
                    torch_array = torch_out.detach().cpu().numpy()
                    entry = details.setdefault(
                        onnx_name,
                        {
                            "name": onnx_name,
                            "shape": list(onnx_out.shape),
                            "max_abs_diff": None,
                            "passed": True,
                        },
                    )
                    if torch_array.shape != onnx_out.shape:
                        _logger.error(
                            f"  [FAIL] Sample {sample_idx + 1} output '{onnx_name}': "
                            f"形状不一致 torch={torch_array.shape} onnx={onnx_out.shape}"
                        )
                        all_passed = False
                        entry["passed"] = False
                        continue
                    is_close = bool(np.allclose(torch_array, onnx_out, rtol=rtol, atol=atol))
                    max_diff = float(np.abs(torch_array - onnx_out).max())
                    previous = entry["max_abs_diff"]
                    entry["max_abs_diff"] = (
                        max_diff if previous is None else max(float(previous), max_diff)
                    )
                    global_max_diff = (
                        max_diff if global_max_diff is None else max(global_max_diff, max_diff)
                    )
                    status = "[OK]" if is_close else "[FAIL]"
                    _logger.info(
                        f"  {status} Sample {sample_idx + 1} output '{onnx_name}': "
                        f"max_diff={max_diff:.2e}"
                    )
                    if not is_close:
                        all_passed = False
                        entry["passed"] = False

        if all_passed:
            _logger.info("[OK] ONNX accuracy verification passed")
        else:
            _logger.error("[FAIL] ONNX accuracy verification failed")

        report = VerifyReport(
            passed=all_passed,
            max_abs_diff=global_max_diff,
            outputs=list(details.values()),
        )
        if result is not None:
            result.parity_max_abs_diff = global_max_diff
        return report


__all__ = ["ONNXExporter"]
