# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: tools/infer.py 命令行入口测试（P4-B）。
"""``tools/infer.py`` 命令行入口测试。

用 toy ``SimpleCNN`` + 纯 state_dict checkpoint + 最小配置，在 tmp 目录中
端到端跑一遍 ``main()``（通过 monkeypatch ``sys.argv`` 注入参数），
验证 JSON / 可视化产物的数量与基本内容。
"""

import json
import os
import sys

import cv2
import numpy as np
import torch

from dvisionix.models import SimpleCNN
from tools.infer import main

#: 测试用输入图片边长
_IMAGE_SIZE = 32
#: 测试用类别数
_NUM_CLASSES = 3

#: 最小推理配置：只保留模型结构与预处理尺寸，不依赖仓库内任何数据
_CONFIG_TEXT = """\
task_type: classification
model:
  type: simple_cnn
  num_classes: 3
data:
  image_size: 32
"""


def _prepare_cli_inputs(tmp_path):
    """在 tmp 目录造 2 张图片、1 个纯 state_dict checkpoint、1 个最小配置。

    Returns:
        (config_path, checkpoint_path, input_dir, output_dir) 四个字符串路径。
    """
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    for index in range(2):
        image = np.random.default_rng(index).integers(
            0, 255, (_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8
        )
        cv2.imwrite(str(input_dir / f"sample_{index}.png"), image)

    checkpoint = tmp_path / "model.pt"
    torch.save(SimpleCNN(num_classes=_NUM_CLASSES).state_dict(), checkpoint)

    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG_TEXT, encoding="utf-8")
    return str(config), str(checkpoint), str(input_dir), str(tmp_path / "outputs")


def _run_cli(monkeypatch, config, checkpoint, input_path, output_dir, *extra_args):
    """以给定参数调用 CLI 主函数（main 读 sys.argv）。"""
    argv = [
        "infer.py",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--input",
        input_path,
        "--output",
        output_dir,
        "--device",
        "cpu",
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()


def test_infer_cli_writes_one_json_per_image(tmp_path, monkeypatch):
    """目录输入：每张图片产出 1 个同名 JSON，且内容为分类结果。"""
    config, checkpoint, input_dir, output_dir = _prepare_cli_inputs(tmp_path)

    _run_cli(monkeypatch, config, checkpoint, input_dir, output_dir)

    names = sorted(os.listdir(output_dir))
    assert names == ["sample_0.json", "sample_1.json"]

    with open(os.path.join(output_dir, "sample_0.json"), encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["task_type"] == "classification"
    assert len(payload["probs"]) == _NUM_CLASSES
    assert len(payload["topk"]) == _NUM_CLASSES
    # 分类概率之和为 1，top-k 按概率降序
    assert abs(sum(payload["probs"]) - 1.0) < 1e-5
    assert [score for _idx, score in payload["topk"]] == sorted(
        (score for _idx, score in payload["topk"]), reverse=True
    )


def test_infer_cli_save_vis_writes_jpg(tmp_path, monkeypatch):
    """--save-vis：在 JSON 之外额外为每张图片产出 1 张可视化 JPG。"""
    config, checkpoint, input_dir, output_dir = _prepare_cli_inputs(tmp_path)

    _run_cli(
        monkeypatch,
        config,
        checkpoint,
        input_dir,
        output_dir,
        "--save-vis",
        "--cfg-options",
        "data.image_size=32",
    )

    names = sorted(os.listdir(output_dir))
    assert names == [
        "sample_0.json",
        "sample_0_vis.jpg",
        "sample_1.json",
        "sample_1_vis.jpg",
    ]
    for name in names:
        if name.endswith(".jpg"):
            # cv2.imwrite 成功时文件非空
            assert os.path.getsize(os.path.join(output_dir, name)) > 0
