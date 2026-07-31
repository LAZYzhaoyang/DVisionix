# -*- coding: utf-8 -*-
"""Config schema 校验与 CLI 解析增强测试。"""

import pytest

from dvisionix.config import Config, parse_cli_options


def _base_config():
    return {
        "task_type": "classification",
        "experiment_name": "exp",
        "model": {"name": "simple_cnn", "num_classes": 4},
        "data": {"image_size": 32},
        "training": {"num_epochs": 2, "batch_size": 16},
    }


def test_valid_config_no_warnings():
    cfg = Config(_base_config())
    warnings = cfg.validate_schema(cfg.task_type)
    assert warnings == []


def test_unknown_key_warns():
    cfg = Config({**_base_config(), "training": {"num_epochs": 2, "num_epcohs": 3}})
    warnings = cfg.validate_schema(cfg.task_type)
    assert any("num_epcohs" in w for w in warnings)


def test_type_error_raises():
    cfg = Config({**_base_config(), "training": {"num_epochs": "two"}})
    with pytest.raises(ValueError, match="num_epochs"):
        cfg.validate_schema(cfg.task_type)


def test_checkpoint_mode_validation():
    cfg = Config({**_base_config(), "checkpoint": {"mode": "invalid"}})
    with pytest.raises(ValueError, match="checkpoint.mode"):
        cfg.validate_schema(cfg.task_type)


def test_alias_warning_when_both_set():
    cfg = Config({**_base_config(), "training": {
        "num_epochs": 2, "learning_rate": 0.01, "optimizer": {"type": "adam", "lr": 0.001},
    }})
    warnings = cfg.validate_schema(cfg.task_type)
    assert any("learning_rate" in w for w in warnings)


def test_unknown_task_type_raises():
    cfg = Config({**_base_config(), "task_type": "foo"})
    with pytest.raises(ValueError, match="task_type"):
        cfg.validate_schema("foo")


def test_cli_parse_list_and_dict():
    assert parse_cli_options(["training.devices=[0,1]"])["training"]["devices"] == [0, 1]
    assert parse_cli_options(["training.optimizer={type: adam, lr: 0.01}"])["training"]["optimizer"] == {
        "type": "adam", "lr": 0.01
    }


def test_cli_parse_scalars():
    cfg = Config(_base_config())
    cfg.update_from_cli([
        "training.num_epochs=5",
        "training.amp=true",
        "training.gradient_clip_val=null",
        "experiment_name=x",
    ])
    assert cfg.training.num_epochs == 5
    assert cfg.training.amp is True
    assert cfg.training.gradient_clip_val is None
    assert cfg.experiment_name == "x"


def test_validate_schema_returns_list():
    cfg = Config(_base_config())
    assert isinstance(cfg.validate_schema(), list)