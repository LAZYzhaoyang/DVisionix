# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 模型契约门禁：所有已注册检测器的 decode() 必须接受统一关键字参数。
"""模型契约门禁（v1.1 阶段 0 新增）。

背景（见 CodePlan 7.1.2 的 D1）：
R4 规定「每个模型保留 ``decode()`` 实例方法做薄桥接」，v0.7.1 也声称已统一 decode 契约。
但实际存在违约：

- ``models/detectors/centernet.py`` 的 ``decode()`` 缺少 ``iou_threshold``
- ``models/detectors/nmsfree_yolo.py`` 的 ``decode()`` 缺少 ``iou_threshold``

而 ``training/tasks/detection.py`` 的 ``validation_step`` **无条件**传入
``score_threshold`` / ``iou_threshold`` / ``max_detections`` 三个关键字参数，
因此上述两个检测器一旦进入验证阶段必然抛 ``TypeError``。

v1.0.0 的 286 条测试全绿也没能发现它，因为「配置加载」类测试只调用 ``build_model()``，
从不构建 loss、也从不执行 ``validation_step``。

本文件把该契约变成可执行断言：任何新增检测器只要 decode 签名不兼容，CI 立即失败。
"""

import inspect
import os

import pytest

import dvisionix.models  # noqa: F401  仅为触发各检测器向 MODELS 注册
from dvisionix.registry import MODELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: DetectionTask.validation_step 无条件传入的关键字参数，检测器 decode 必须全部接受。
#: 见 dvisionix/training/tasks/detection.py 中 model.decode(...) 的调用点。
REQUIRED_DECODE_KWARGS = ("score_threshold", "iou_threshold", "max_detections")

_DETECTOR_MODULE_PREFIX = "dvisionix.models.detectors"


def _registered_modules(prefix: str) -> dict:
    """返回 {类: {注册名...}}，只保留定义在 ``prefix`` 模块下的已注册类。"""
    found: dict = {}
    for key in MODELS.keys():
        cls = MODELS.get(key)
        module = getattr(cls, "__module__", "")
        if module.startswith(prefix):
            found.setdefault(cls, set()).add(key)
    return found


DETECTORS = _registered_modules(_DETECTOR_MODULE_PREFIX)


def _accepts_kwarg(func, name: str) -> bool:
    """判断可调用对象是否能接受名为 ``name`` 的关键字参数（含 **kwargs 兜底）。"""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - 内置对象无签名
        return False
    for param in params.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return name in params


def test_detector_registry_is_not_empty():
    """自检：确保注册表被正确导入，否则下面的参数化会静默变成空集合。"""
    assert DETECTORS, "未发现任何已注册检测器，说明注册表导入失败或模块路径已变更"


@pytest.mark.unit
@pytest.mark.parametrize(
    "cls",
    sorted(DETECTORS, key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
def test_detector_decode_accepts_unified_kwargs(cls):
    """每个检测器的 decode() 必须接受 DetectionTask 传入的全部关键字参数。"""
    assert hasattr(cls, "decode"), f"{cls.__name__} 缺少 decode() 实例方法（违反 R4）"
    missing = [name for name in REQUIRED_DECODE_KWARGS if not _accepts_kwarg(cls.decode, name)]
    assert not missing, (
        f"{cls.__name__}.decode() 不接受 {missing}；"
        f"DetectionTask.validation_step 会无条件传入这些参数，验证阶段将抛 TypeError。"
        f"当前签名: {inspect.signature(cls.decode)}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "cls",
    sorted(DETECTORS, key=lambda c: c.__name__),
    ids=lambda c: c.__name__,
)
def test_detector_decode_has_canonical_parameter_order(cls):
    """decode() 的前两个位置参数必须是 (preds, image_hw)，保证可位置传参。"""
    params = list(inspect.signature(cls.decode).parameters)
    assert params[:3] == ["self", "preds", "image_hw"], (
        f"{cls.__name__}.decode() 前两个位置参数应为 (preds, image_hw)，"
        f"当前为 {params[:3]}；decode 契约见 dvisionix/models/detectors/base.py"
    )
