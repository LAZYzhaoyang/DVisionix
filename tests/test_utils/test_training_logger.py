# -*- coding: utf-8 -*-
"""TrainingLogger（utils.logging）测试：JSONL 事件流与指标记录。"""

import json
import os
import tempfile

from dvisionix.utils.logging import TrainingLogger


def test_training_logger_jsonl_and_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(
            "test_tl", log_dir=tmpdir, tb_dir=os.path.join(tmpdir, "tb"), console=False
        )
        logger.log_metrics(step=1, mode="train", metrics={"loss": 0.5, "acc": 0.9})
        logger.log_event("train_end", epochs=2, global_step=10)
        logger.close()

        events_path = os.path.join(tmpdir, "events.jsonl")
        assert os.path.exists(events_path)
        with open(events_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert lines[0]["event"] == "metrics"
        assert lines[0]["mode"] == "train" and lines[0]["step"] == 1
        assert lines[0]["metrics"]["loss"] == 0.5
        assert lines[1]["event"] == "train_end"
        assert lines[1]["global_step"] == 10


def test_training_logger_no_dir():
    logger = TrainingLogger("test_tl2", console=False)
    logger.log_metrics(step=0, mode="val", metrics={"loss": 1.0})
    logger.close()  # 不应抛异常
