# 日志系统

`dvisionix.utils` 提供结构化日志器，支持控制台 + 文件双输出、分级与按阶段记录指标。

## 基本用法

```python
from dvisionix.utils import get_logger, log_metrics

logger = get_logger("dvisionix.demo", level="info", log_dir="./logs")
logger.info("start training")

# 按阶段记录一组指标
log_metrics(logger, {"loss": 0.51, "acc": 90.2}, step=1, stage="val")
# 输出: [val][step 1] loss: 0.5100 | acc: 90.2000
```

## 参数说明（get_logger）
- `name`: 日志器名称。
- `level`: `debug/info/warning/error/critical`。
- `log_dir`: 提供后自动生成带时间戳的日志文件；也可用 `log_file` 指定路径。
- `console`: 是否输出到控制台。

## 与训练结合
在 Config 驱动的 demo（`demos/train_from_config.py`）中，日志目录取自
`cfg.logging.log_dir`，训练开始/结束、样本数、模型参数量等均会记录到文件，便于排查问题。
