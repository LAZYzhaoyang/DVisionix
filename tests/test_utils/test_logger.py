# -*- coding: utf-8 -*-
"""日志系统单元测试。"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dvisionix.utils import format_metrics, get_logger, log_metrics


def test_console_logger():
    logger = get_logger("test_console", level="debug", console=True)
    assert logger.level == 10
    assert len(logger.handlers) == 1


def test_file_logger_writes():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger("test_file", log_dir=tmpdir, console=False)
        logger.info("hello world")
        log_file = getattr(logger, "log_file")
        # 关闭 handler 以刷新写入
        for h in list(logger.handlers):
            h.flush()
            h.close()
            logger.removeHandler(h)
        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "hello world" in content


def test_format_metrics():
    s = format_metrics({"loss": 0.12345, "acc": 95.2}, precision=3)
    assert "loss: 0.123" in s
    assert "acc: 95.200" in s


def test_log_metrics_no_error():
    logger = get_logger("test_metrics", console=True)
    log_metrics(logger, {"loss": 0.5, "acc": 90.0}, step=1, stage="val")


if __name__ == "__main__":
    print("Running logger tests...")
    test_console_logger()
    print("ok test_console_logger")
    test_file_logger_writes()
    print("ok test_file_logger_writes")
    test_format_metrics()
    print("ok test_format_metrics")
    test_log_metrics_no_error()
    print("ok test_log_metrics_no_error")
    print("All logger tests passed!")
