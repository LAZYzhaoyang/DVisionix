# -*- coding: utf-8 -*-
"""配置系统单元测试。"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dvisionix.config import Config


def test_config_from_dict():
    cfg = Config({"model": {"name": "resnet50", "num_classes": 10}})
    assert cfg.model.name == "resnet50"
    assert cfg["model"]["num_classes"] == 10


def test_config_dot_access():
    cfg = Config({"training": {"batch_size": 32, "lr": 0.001}})
    assert cfg.training.batch_size == 32
    assert cfg.training.lr == 0.001


def test_config_set():
    cfg = Config({})
    cfg.batch_size = 64
    assert cfg.batch_size == 64
    cfg["epochs"] = 10
    assert cfg["epochs"] == 10


def test_config_merge():
    cfg1 = Config({"model": {"name": "resnet", "layers": 50}})
    cfg2 = {"model": {"layers": 101}, "training": {"lr": 0.01}}
    merged = cfg1.merge(cfg2, override=True)
    assert merged.model.name == "resnet"
    assert merged.model.layers == 101
    assert merged.training.lr == 0.01


def test_config_yaml_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config({"model": {"name": "resnet50"}, "training": {"batch_size": 32}})
        path = os.path.join(tmpdir, "test.yaml")
        cfg.dump(path)
        loaded = Config.from_yaml(path)
        assert loaded.model.name == "resnet50"
        assert loaded.training.batch_size == 32


def test_config_inheritance():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "base.yaml")
        with open(base_path, "w", encoding="utf-8") as f:
            f.write("training:\n  batch_size: 32\n  lr: 0.001\nmodel:\n  name: base_model\n")
        child_path = os.path.join(tmpdir, "child.yaml")
        with open(child_path, "w", encoding="utf-8") as f:
            f.write("_base_: base.yaml\ntraining:\n  lr: 0.01\nmodel:\n  name: child_model\n")
        cfg = Config.from_yaml(child_path)
        assert cfg.training.batch_size == 32
        assert cfg.training.lr == 0.01
        assert cfg.model.name == "child_model"


def test_config_default():
    cfg = Config.from_default("classification")
    assert cfg.task_type == "classification"
    assert cfg.training.batch_size == 32
    assert cfg.model.num_classes == 10


def test_config_validate():
    cfg = Config({"model": {"name": "resnet"}, "training": {"lr": 0.01}})
    assert cfg.validate(["model.name", "training.lr"])
    try:
        cfg.validate(["model.missing_key"])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    print("Running config tests...")
    test_config_from_dict(); print("ok test_config_from_dict")
    test_config_dot_access(); print("ok test_config_dot_access")
    test_config_set(); print("ok test_config_set")
    test_config_merge(); print("ok test_config_merge")
    test_config_yaml_roundtrip(); print("ok test_config_yaml_roundtrip")
    test_config_inheritance(); print("ok test_config_inheritance")
    test_config_default(); print("ok test_config_default")
    test_config_validate(); print("ok test_config_validate")
    print("All config tests passed!")
