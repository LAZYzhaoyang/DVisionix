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
---

## v1.1 变更要点

> 完整清单见 [v1.1 变更与迁移指南](v1.1_changes.md)。

**TensorBoard 标量现在真正会写入。** v1.0.0 会创建 `work_dir/tb` 目录，
但 Trainer 从不调用 `logger.log_metrics(...)`，因此 TensorBoard 里始终是空的。
现在每个 epoch 结束都会写入 `epoch/<metric>` 标量，同时落一条 JSONL 指标事件：

```bash
tensorboard --logdir <work_dir>/tb
```

`log_metrics` 新增 `console` 参数：Trainer 传 `console=False`，
因为 epoch 摘要已经单独打印过，避免重复刷屏。

**库代码不再使用 `print()`。** `load_backbone` 的缺失/多余键告警改为走 logger
（`dvisionix.checkpoint`）。

**分布式评估通信可度量。** 分布式验证会打印一次通信统计，
也可通过 API 读取：

```python
trainer.gather_report()
# {'calls': 2, 'ms_total': 1.83, 'ms_per_call': 0.92,
#  'local_samples': 8, 'ms_per_sample': 0.23}
```
