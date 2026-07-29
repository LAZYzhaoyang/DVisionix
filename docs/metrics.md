# 指标（Metrics）

DVisionix 的指标模块采用「原子指标 + 组合容器 + 预设」的设计：每个指标是独立、可累积的对象，可自由组合，并注册到全局 `METRICS` 注册表以支持配置驱动构建。

## 一、设计与接口

所有指标继承 `dvisionix.metrics.BaseMetric`，遵循累积式接口：

- `reset()`：清空内部累积状态（每个 epoch 开始时调用）。
- `update(*args, **kwargs)`：喂入一个 batch，仅累加内部状态，不返回最终值。
- `compute()`：基于当前累积的全部状态计算并返回结果。

返回值约定：
- 原子指标 `compute()` 返回标量 `float`（per-class 模式返回 `list`）。
- `MetricCollection.compute()` 返回 `dict`：成员返回 dict 则合并其键，返回标量则以成员 `name` 为键。

> 为什么用累积式而非"逐 batch 求平均"：mIoU / mAP / macro-F1 等指标 `mean(每个 batch 的值) != 全局值`，必须先累积混淆矩阵 / TP-FP 再计算。

## 二、原子指标清单

| 任务 | 指标类 | 注册名 | 说明 |
| --- | --- | --- | --- |
| 分类 | `Accuracy` | `accuracy` | Top-1 准确率 |
| 分类 | `TopKAccuracy` | `top_k_accuracy` | Top-K 准确率（需 (B, C) logits） |
| 分类 | `Precision` | `precision` | 精确率（`average`: macro/micro/weighted/none） |
| 分类 | `Recall` | `recall` | 召回率 |
| 分类 | `F1Score` | `f1_score` | F1 分数 |
| 分割 | `MeanIoU` | `mean_iou` | 平均交并比 mIoU |
| 分割 | `PixelAccuracy` | `pixel_accuracy` | 像素准确率 |
| 分割 | `DiceScore` | `dice_score` | Dice 系数 |
| 检测 | `MeanAveragePrecision` | `map` / `mean_average_precision` | COCO mAP / mAP_50 / mAP_75 |

```python
import torch
from dvisionix.metrics import Accuracy, MeanIoU, MeanAveragePrecision

acc = Accuracy()
acc.update(logits, targets)   # 可多次调用累积
print(acc.compute())          # 标量
```

## 三、组合：MetricCollection

`MetricCollection` 把多个指标组合在一起，把 `reset`/`update` 广播给成员，`compute` 汇总为一个 dict。成员支持三种写法（可混用）：

- 指标实例：`Accuracy()`
- 配置字典：`{"type": "f1_score", "average": "macro", "num_classes": 10}`
- 字符串：`"accuracy"`（等价于无参 `{"type": "accuracy"}`）

```python
from dvisionix.metrics import MetricCollection, TopKAccuracy

metrics = MetricCollection([
    "accuracy",
    {"type": "f1_score", "average": "macro", "num_classes": 10},
    TopKAccuracy(k=5),
])

metrics.reset()
for logits, targets in loader:
    metrics.update(logits, targets)      # 广播给每个成员
print(metrics.compute())                 # {"accuracy": ..., "f1": ..., "top5_acc": ...}

metrics.add("recall")                    # 链式追加成员
```

## 四、预设组合

两种入口，任选其一：

```python
from dvisionix.metrics import get_preset_metrics, ClassificationMetrics

# 1) 函数式快捷入口
metrics = get_preset_metrics("classification", num_classes=10)  # accuracy/precision/recall/f1
metrics = get_preset_metrics("segmentation", num_classes=19)    # mIoU/pixel_accuracy
metrics = get_preset_metrics("detection", num_classes=80)       # mAP/mAP_50/mAP_75

# 2) 预设类（也是自定义组合指标的范例）
metrics = ClassificationMetrics(num_classes=10)
```

预设类 `ClassificationMetrics` / `SegmentationMetrics` / `DetectionMetrics` 内部就是「`MetricCollection` + 原子指标」的封装，既开箱即用，也可作为自定义组合指标的模板参考。

## 五、配置驱动构建

所有指标注册到 `METRICS`，可用 `build_metric` 构建单个指标，或在配置里用列表组合：

```python
from dvisionix.metrics import build_metric
m = build_metric({"type": "accuracy"})
```

```yaml
metrics:
  - {type: accuracy}
  - {type: top_k_accuracy, k: 5}
  - {type: f1_score, average: macro, num_classes: 10}
```

## 六、自定义指标

继承 `BaseMetric`，实现 `update` / `compute` / `reset` 三个方法即可，注册后可参与组合与配置构建：

```python
import torch
from dvisionix.metrics import BaseMetric, MetricCollection, Accuracy
from dvisionix.registry import METRICS

@METRICS.register()                 # 以类名 "ErrorRate" 注册
@METRICS.register(name="error_rate")
class ErrorRate(BaseMetric):
    def __init__(self, name="error_rate"):
        super().__init__(name)      # BaseMetric.__init__ 会调用 self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, logits, targets):
        preds = logits.argmax(dim=1)
        self.correct += int((preds == targets).sum())
        self.total += int(targets.numel())

    def compute(self):
        return 1.0 - (self.correct / self.total if self.total else 0.0)

# 直接组合使用
metrics = MetricCollection([Accuracy(), ErrorRate()])
# 或配置构建
metrics = MetricCollection(["accuracy", {"type": "error_rate"}])
```

要让自定义指标在包导入时自动注册，把定义放进被导入的模块（或训练脚本导入前完成注册）。

## 七、说明

- **训练循环接入**：当前 `Trainer`/`Task` 仍在 step 内内联计算简单指标（如分类 acc），本模块作为独立可组合工具提供；把可组合 metrics 正式接入验证循环将在后续 Trainer 改造中进行。
- **检测后端**：`MeanAveragePrecision(use_torchmetrics=True)` 可切换到 `torchmetrics` 后端（需 `pip install torchmetrics[detection]`），内置实现仅用于快速验证。