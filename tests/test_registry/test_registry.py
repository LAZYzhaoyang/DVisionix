# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 注册表与配置驱动构建的单元测试。
"""注册表与配置驱动构建的单元测试。

注意：``dvisionix`` 的顶层导入是**惰性**的（见 dvisionix/__init__.py 的 PEP 562
说明），组件的注册发生在各子模块被导入时。因此本文件必须显式导入会填充注册表的
子模块 —— 不能依赖「导入 dvisionix 就自动注册好一切」，那会让本文件的结果取决于
测试执行顺序（此前正是因为顺序凑巧才一直通过）。
"""

import pytest

# 显式导入以完成注册：models -> MODELS/BACKBONES/HEADS/NECKS/LAYERS，
# training -> TASKS，metrics -> METRICS，models.losses -> LOSSES
import dvisionix.metrics  # noqa: F401
import dvisionix.models  # noqa: F401
import dvisionix.training  # noqa: F401
from dvisionix.registry import LOSSES, METRICS, MODELS, TASKS, Registry, build_from_cfg


def test_register_and_build():
    reg = Registry("t")

    @reg.register()
    class Foo:
        def __init__(self, a, b=2):
            self.a, self.b = a, b

    obj = reg.build({"type": "Foo", "a": 1, "b": 5})
    assert obj.a == 1 and obj.b == 5
    assert "Foo" in reg and len(reg) == 1


def test_register_with_alias():
    reg = Registry("t")

    class Bar:
        pass

    reg.register(Bar, name="bar_alias")
    assert reg.build({"type": "bar_alias"}).__class__ is Bar


def test_missing_key_raises():
    reg = Registry("t")
    with pytest.raises(KeyError):
        reg.get("nope")
    with pytest.raises(KeyError):
        reg.build({"type": "nope"})


def test_build_requires_type():
    reg = Registry("t")
    with pytest.raises(KeyError):
        reg.build({"a": 1})


def test_duplicate_register_raises():
    reg = Registry("t")

    class Baz:
        pass

    reg.register(Baz)
    with pytest.raises(KeyError):
        reg.register(Baz)


def test_default_kwargs_overridden_by_cfg():
    reg = Registry("t")

    class Q:
        def __init__(self, x=0):
            self.x = x

    reg.register(Q)
    assert reg.build({"type": "Q"}, x=9).x == 9
    # cfg 覆盖 default_kwargs
    assert reg.build({"type": "Q", "x": 1}, x=9).x == 1


def test_global_registries_populated():
    assert "SimpleCNN" in MODELS
    assert "simple_cnn" in MODELS
    assert "ClassificationTask" in TASKS
    assert "DiceLoss" in LOSSES
    assert "ClassificationMetrics" in METRICS


def test_build_model_simple_cnn():
    m = MODELS.build({"type": "simple_cnn", "num_classes": 4})
    assert m.__class__.__name__ == "SimpleCNN"


def test_build_from_cfg_helper():
    m = build_from_cfg({"type": "simple_cnn", "num_classes": 3}, MODELS)
    assert m.__class__.__name__ == "SimpleCNN"


# ---------------------------------------------------------------------------
# 注册表选择器策略（CodePlan 7.3 步骤 1-1）
#
# 背景：v1 版计划（commit e20d5e9）曾把「构造参数 name」误判为缺陷，并打算改成只认
# `_name_`。实际 `cfg.pop("type", None) or cfg.pop("name", None)` 因 `or` 短路，
# `type` 存在时 `name` 从未被消耗；而 `name` 作为选择器是
# `dvisionix/config/defaults/*.yaml` 一直依赖的既有约定。以下测试把最终策略锁死。
# ---------------------------------------------------------------------------


def test_type_takes_priority_and_name_is_forwarded_to_constructor():
    """`type` 是标准选择器；同名构造参数 `name` 必须原样透传（P0-3 的验收标准）。"""
    reg = Registry("t")

    class Named:
        def __init__(self, name):
            self.name = name

    reg.register(Named, name="named")
    assert reg.build({"type": "named", "name": "resnet18"}).name == "resnet18"


def test_explicit_underscore_name_is_an_alias_for_type():
    """`_name_` 是 `type` 的显式别名，便于与构造参数 `name` 并存。"""
    reg = Registry("t")

    class Named:
        def __init__(self, name):
            self.name = name

    reg.register(Named, name="named")
    assert reg.build({"_name_": "named", "name": "mobilenet"}).name == "mobilenet"


def test_legacy_name_selector_still_works_and_warns():
    """旧配置只用 `name` 选择组件时必须仍然可用，但发出 DeprecationWarning。

    v1.0.0 的 3 个内置默认配置与 3 个官方 demo 配置都写作 `model: {name: ...}`；
    直接移除该兼容路径会一次性打断这 6 个配置。
    """
    reg = Registry("t")

    class Legacy:
        def __init__(self, value=0):
            self.value = value

    reg.register(Legacy, name="legacy_alias")
    with pytest.warns(DeprecationWarning, match="name"):
        obj = reg.build({"name": "legacy_alias", "value": 7})
    assert isinstance(obj, Legacy) and obj.value == 7


def test_unregistered_name_is_not_consumed_as_selector():
    """`name` 值不是已注册键时，不应被当作选择器消耗，而应给出明确的缺 type 错误。"""
    reg = Registry("t")

    class NeedsName:
        def __init__(self, name):
            self.name = name

    reg.register(NeedsName, name="needs_name")
    # 只有构造参数 name、没有 type：必须报「缺少 type」，而不是猜注册名
    with pytest.raises(KeyError, match="type"):
        reg.build({"name": "resnet18"})
    # 显式给出 type 后，name 正常透传
    assert reg.build({"type": "needs_name", "name": "resnet18"}).name == "resnet18"


def test_official_backbone_config_still_builds_timm_model():
    """回归：`{"type": "timm_backbone", "name": "resnet18"}` 必须真的构造 resnet18。

    这是 v1 版 P0-3 给出的验收标准 —— 它在 v1.0.0 上本来就成立，
    本测试的作用是防止将来有人再把 `name` 从构造函数参数里剥掉。
    """
    pytest.importorskip("timm")
    from dvisionix.registry import BACKBONES

    backbone = BACKBONES.build(
        {
            "type": "timm_backbone",
            "name": "resnet18",
            "pretrained": False,
            "features_only": True,
            "out_indices": [1, 2, 3, 4],
        }
    )
    assert getattr(backbone, "name", None) == "resnet18"
