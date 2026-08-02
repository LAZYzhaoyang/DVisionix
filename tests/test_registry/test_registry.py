# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 注册表与配置驱动构建的单元测试。
"""注册表与配置驱动构建的单元测试。"""

import pytest

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
