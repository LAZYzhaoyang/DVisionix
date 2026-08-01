# 日志系统

日志/可视化统一在 `dvisionix.utils.logging`：

- `get_logger`：结构化日志器（console + 文件）。
- `TrainingLogger`：训练级封装（console + 文件 + JSONL 事件流 + TensorBoard），
  由 Trainer 自动创建并挂在 `trainer.logger` 上；所有回调输出都走它（全库不使用 print）。

## 基本用法

```python
from dvisionix.utils import get_logger, log_metrics

logger = get_logger("dvisionix.demo", level="info", log_dir="./logs")
logger.info("start training")

# 按阶段记录一组指标
log_metrics(logger, {"loss": 0.51, "acc": 90.2}, step=1, stage="val")
# 输出: [val][step 1] loss: 0.5100 | acc: 90.2000
```

## 训练日志（TrainingLogger）

```python
from dvisionix.utils.logging import TrainingLogger

tl = TrainingLogger("dvisionix.train", log_dir="./logs", tb_dir="./logs/tb")
tl.log_metrics(step=1, mode="train", metrics={"loss": 0.5, "acc": 0.9})  # console + JSONL + TensorBoard
tl.log_event("train_end", epochs=2, global_step=10)                      # 自定义 JSONL 事件
tl.close()
```

Config 驱动训练（`tools/train.py`）会自动在 work_dir 下产出：

```
work_dir/
├── logs/dvisionix.trainer_*.log     # 结构化日志
├── logs/events.jsonl                # 机器可读事件流
└── tb/                              # TensorBoard（tensorboard --logdir <work_dir>/tb）
```

## 参数说明（get_logger）
- `name`: 日志器名称。
- `level`: `debug/info/warning/error/critical`。
- `log_dir`: 提供后自动生成带时间戳的日志文件；也可用 `log_file` 指定路径。
- `console`: 是否输出到控制台。