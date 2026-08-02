# -*- coding: utf-8 -*-
"""Config._delete_ 替换语义测试。"""

from dvisionix.config import Config


def test_delete_marker_replaces_dict():
    base = {"model": {"name": "grid_detection", "width": 64, "num_classes": 3}}
    override = {"model": {"_delete_": True, "type": "fcos", "num_classes": 5}}
    merged = Config._deep_merge(base, override)
    assert merged["model"] == {"type": "fcos", "num_classes": 5}
    assert "name" not in merged["model"] and "width" not in merged["model"]


def test_delete_marker_removed_from_result():
    base = {"loss": {"type": "grid_detection", "obj_weight": 1.0}}
    override = {"loss": {"_delete_": True, "type": "fcos_detection"}}
    merged = Config._deep_merge(base, override)
    assert merged["loss"] == {"type": "fcos_detection"}
    assert "_delete_" not in merged["loss"]


def test_normal_merge_still_deep():
    base = {"training": {"num_epochs": 50, "batch_size": 32}}
    override = {"training": {"num_epochs": 5}}
    merged = Config._deep_merge(base, override)
    assert merged["training"] == {"num_epochs": 5, "batch_size": 32}
