# DVisionix 重构计划（Refactor Plan v0.2）

> 本文档取代旧的开发进度清单，作为算法库重构的唯一执行依据。
> 目标：把当前“教学脚手架”升级为 **配置驱动 + 组件注册** 的标准视觉算法库。

---

## 一、重构目标

1. **配置驱动**：所有可替换组件（model / data / task / loss / metric）均由 YAML 配置构建。
2. **组件注册**：统一 Registry 机制，新增组件“注册即可用”，不改 `__init__` 硬编码。
3. **模型组件化**：`backbone → neck → head` 三段式，后处理与模型解耦。
4. **训练引擎现代化**：AMP、梯度累积、完整 resume、确定性、按样本加权指标。
5. **正确性优先**：检测 mAP / NMS 对接成熟实现，去掉“简化版”自研逻辑。
6. **代码卫生**：修复全部乱码、依赖冲突、打包 bug，兑现或删除文档承诺。

---

## 二、目标目录结构

```
dvisionix/
  registry.py          # 新增：统一 Registry + build_from_cfg
  config/              # 保留 Config，补 schema 校验 + CLI override
  data/
    backends/          # torchvision/COCO/VOC 真实数据源封装
    adapters/ datasets/ transforms/ collate.py
  models/
    base.py            # 强化 BaseModel 契约 + task_type 枚举
    backbones/         # timm features_only 多尺度 + out_channels
    necks/             # FPN 等（新增）
    heads/             # ClsHead/SegHead/DetHead（新增）
    detectors/         # 组装器，含 decode 包装（后处理独立）
    toy/               # SimpleCNN 等玩具网络，仅测试/教学
    postprocess.py     # NMS/decode 集中于此
  training/            # Trainer/Task/Callback/losses（去调度器双轨）
  metrics/             # 对接 torchmetrics/pycocotools
  utils/  export/
tools/
  train.py             # 统一 config 驱动训练入口
```

---

## 三、已识别问题清单（重构依据）

### 代码卫生（P0）
- 多文件中文注释乱码（GBK↔UTF-8 混淆）：models/detection.py、models/postprocess.py、
  config/config.py、data/factory.py、data/collate.py、data/transforms/third_party.py、
  metrics/base.py、utils/device.py、utils/logger.py、export/*、setup.py、requirements.txt 等。
- setup.py 使用 os.path.exists 但未 import os，安装会报错。
- 依赖冲突：setup.py 写 torch>=2.12.0（不存在），requirements.txt 写 torch>=2.0.0。
- README 宣称“覆盖率>80% / black / flake8 / mypy”未兑现，需补齐或删除。

### 架构（P1）
- 无注册表：models/tasks/transforms 硬编码 import，DatasetFactory 靠字符串反射，脆弱。
- Config 半配线：train_from_config.py 只认分类，检测/分割配置无法真正驱动。
- 调度器双轨：Trainer.fit 内部 scheduler.step() 与 LearningRateScheduler 回调重复，易 double-step。
- Task 把优化器/调度器写死（lr、T_max 不可配）。

### Model 层（P1，重点）
- 三套互不兼容体系并存：SimpleCNN/SimpleSegmentationModel/SimpleDetectionModel、TimmBackbone/TimmClassifier、GridDetectionModel。
- 两个检测模型互相矛盾：SimpleDetectionModel 输出 dict 但无 loss/decode/task（死代码）；只有 GridDetectionModel 接入训练链路。
- 输出契约不统一：有的返回 Tensor，有的返回 dict，decode/NMS 塞进模型内部。
- BaseModel 契约过弱，task_type 只是随手字符串。
- TimmBackbone 用 num_classes=0 只取全局向量，丢弃多尺度特征，无法喂检测/分割。
- 无 backbone/neck/head 分层，无 from_config，无预训练权重管理。

### Trainer 能力缺口（P1）
- 无 AMP、无梯度累积、无 DDP。
- resume 不完整：load_checkpoint 未在 fit 调用；EarlyStopping/scheduler/scaler 状态未存。
- 指标聚合按 batch 均值再平均，最后不满 batch 失真，应按样本数加权。
- on_batch_end 被 log_interval 截断，导致 TensorBoardLogger 丢数据。
- 到处 print，未使用现成的 utils.logger；无 seed 设置。

### 数据管线（P2）
- BaseDataset._standardize_format 在 __getitem__ 硬编码 ImageNet 归一化，与 transforms 的 Normalize 重复。
- DetectionResize 不裁剪越界框；AlbumentationsWrapper 不处理 bbox；MetricCollection 检测分支为 pass。

### Metrics / 后处理（P2）
- DetectionMetrics 自研简化实现，compute 与 _compute_map_at_iou 重复计算，正确性/效率存疑。
- GIoULoss 假设框已配对，缺正负样本匹配。

---

## 四、分阶段执行计划

### 阶段 0 — 卫生清理（P0）
- [ ] 修复全部乱码文件（统一 UTF-8 重写注释/docstring）。
- [ ] 修 setup.py（import os），统一 torch 版本约束（torch>=2.0,<3）。
- [ ] 对齐 README/文档承诺（补工具链配置或删除未兑现说法）。
- [ ] 引入 pyproject.toml + black/ruff/mypy 配置。

### 阶段 1 — Registry 与 Config 配线（P1）
- [ ] 新增 dvisionix/registry.py：Registry + build_from_cfg。
- [ ] 定义 MODELS/BACKBONES/NECKS/HEADS/DATASETS/TRANSFORMS/TASKS/LOSSES/METRICS 注册表。
- [ ] 组件用装饰器注册，去掉 __init__ 硬编码依赖。
- [ ] Config 增加 schema 校验 + CLI 覆盖（--cfg-options a.b=c）。
- [ ] 新增 tools/train.py 统一入口，config 真正驱动 data+model+task+trainer。

### 阶段 2 — Model 层重构（P1）
- [ ] 强化 BaseModel 契约：task_type 枚举、forward 只出 raw 预测、from_config、init_weights。
- [ ] TimmBackbone 改 features_only=True + out_indices，输出多尺度 + out_channels。
- [ ] 新增 necks/（FPN）与 heads/（ClsHead/SegHead/DetHead）。
- [ ] 新增 detectors/ 组装器；decode/NMS 移至 postprocess.py。
- [ ] 玩具网络移入 models/toy/，标注“仅测试”；删除死代码 SimpleDetectionModel。
- [ ] GridDetectionModel 标注 demo 或重写为 backbone+DetHead 可用版本。

### 阶段 3 — Trainer 能力补强（P1）
- [x] 去调度器双轨（保留回调，移除 fit 内部 step）。
- [x] 新增 AMP（autocast+GradScaler）、梯度累积、seed/确定性。
- [x] 指标聚合改为按样本数加权。
- [x] on_batch_end 脱钩 log_interval，打印频率交回调自身。
- [x] 完整 resume：fit 调用 load_checkpoint；保存 optimizer/scheduler/scaler/callback 状态。
- [x] 统一日志走 utils.logger，去掉散落 print。
- [ ] （可选）DDP/多卡（暂缓，后续按需求引入）。

### 阶段 4 — 数据管线澄清（P2）
- [x] 归一化职责唯一化：BaseDataset 感知 transforms.provides_normalization；transforms 内 Normalize 默认启用，双重归一化路径打 DeprecationWarning。
- [x] DetectionResize 补越界裁剪 + 退化框过滤；AlbumentationsWrapper 自动补 bbox_params 并回写 boxes/labels。
- [ ] （后续）真实数据源迁入 data/backends/，DatasetFactory 走全局 DATASETS 注册表。

### 阶段 5 — Metrics / 后处理正确性（P2）
- [x] 检测 mAP 支持 torchmetrics 可选后端（use_torchmetrics=True），无依赖时回退内置实现。
- [x] MetricCollection 检测分支从空 pass 补为标准派发（outputs.pred_* + batch.boxes/labels）。
- [x] GIoULoss 文档明确用途：正负样本匹配由 task 侧负责，Loss 仅接收匹配后的 (N,4) 对。
- [ ] （后续）分类/分割指标全面迁移到 torchmetrics（当前内置实现在小规模数据下可用）。

### 阶段 6 — 文档 / 测试 / 示例收尾（P2）
- [x] 补齐单测：registry 构建、model 前向 shape、trainer 一步训练、resume、AMP、指标数值（68 passed）。
- [x] demos 顶部统一加迁移提示指向 tools/train.py；verify_*.py 加提示指向 pytest。
- [x] 更新 README：架构概览、config 驱动用法、0.1.x→0.2.0 迁移说明。

---

## 五、执行顺序与依赖

- 必须先做：阶段 0 → 阶段 1（地基）。
- 可并行：阶段 2（Model）与阶段 3（Trainer）。
- 依赖注册表：阶段 2/4/5。
- 建议节奏：0 → 1 → (2 ∥ 3) → 4 → 5 → 6。

## 六、兼容与风险

- 破坏性变更：model 命名空间、Trainer 调度行为、数据归一化位置将 break 现有 demo。
- 计划在 0.2.0 打破，保留迁移说明；玩具网络保留在 toy/ 以免破坏教学示例。
- 检测第三方依赖通过 extras_require 可选安装。

---

## 七、进度记录

- [x] 阶段 0：卫生清理
  - 丢弃“乱码修复”任务：经验证，所谦“乱码”文件在磁盘上均为合法 UTF-8，仅为 PowerShell GBK 代码页显示假象。
  - 修复 setup.py（import os）、统一 torch 版本约束（>=2.0,<3）、新增 pyproject.toml（black/ruff/mypy/pytest）、版本号 0.2.0、README 软化未兑现承诺。
  - 基线测试 38 passed。
- [x] 阶段 1：Registry + Config 配线
  - 新增 dvisionix/registry.py（Registry + build_from_cfg + 9 个全局注册表）。
  - 注册 models/tasks/losses/metrics/datasets 组件 + 小写别名，提供 build_model/build_task/build_loss/build_metric/build_dataset 并在顶层导出。
  - Config 新增 update_from_cli / parse_cli_options（--cfg-options a.b=v）。
  - 新增 tools/train.py 配置驱动入口：分类/分割 demo 端到端跑通，检测构建验证通过。
  - 新增 tests/test_registry，全量 47 passed。
- [x] 阶段 2：Model 层组件化重构
  - 强化 BaseModel 契约（TASK_TYPES 校验、init_weights、from_config）。
  - TimmBackbone 支持 features_only 多尺度输出 + out_channels。
  - 新增 necks/FPN、heads/（ClsHead/SegHead/DetHead）、detectors/GeneralizedModel，均注册到注册表。
  - GeneralizedModel 验证了分类/分割/检测三任务的 backbone+neck+head 组合。
  - SimpleDetectionModel 标注为废弃死代码。
  - 新增 tests/test_models/test_components.py（6 个测试），全量 53 passed。
- [x] 阶段 3：Trainer 能力补强
  - Trainer 增加 amp / accumulate_grad_batches / seed / resume_from 参数。
  - _run_epoch 采用 AMP autocast + 梯度累积 + 按样本数加权聚合，on_batch_end 每步触发。
  - _has_lr_scheduler_callback 消除调度器双轨；fit 起始自动 resume。
  - save/load_checkpoint 纳入 scaler_state_dict 与 callbacks_state_dict；EarlyStopping / ModelCheckpoint 实现 state_dict。
  - Trainer 内 print 全部改走 utils.logger（dvisionix.trainer 命名空间）。
  - tools/train.py 增加 --resume 命令行开关并透传 amp/accumulate/seed。
  - 新增 tests/test_training/test_stage3.py（AMP smoke / 累积 / seed 可复现 / resume / 加权聚合），全量 58 passed。
- [x] 阶段 4：数据管线澄清（P2 部分完成）
  - BaseTransform 引入 provides_normalization 标记；Normalize/DetectionNormalize/SegmentationNormalize 声明为 True，Compose 及各 pipeline 自动汇总。
  - ClassificationTransforms/DetectionTransforms/SegmentationTransforms 默认自带 ImageNet 归一化，transforms 成为归一化的唯一权威。
  - BaseDataset._standardize_format 检测 transforms 状态；未提供归一化时保留旧行为并发 DeprecationWarning。
  - DetectionResize 在缩放后 clip 到目标尺寸内并过滤退化框，与 labels 同步。
  - AlbumentationsWrapper 自动补 bbox_params（pascal_voc），并双向转换 boxes/labels。
  - 新增 tests/test_data/test_stage4.py（5 项），全量 63 passed（2 条 DeprecationWarning 为预期）。
  - 磁盘全仓扫描 79 个文件，未发现 U+FFFD/mojibake，仓库内所谓"乱码"确为 PowerShell 显示假象。
- [x] 阶段 5：Metrics / 后处理正确性
  - MetricCollection 检测分支实现真正的派发（pred_boxes/pred_scores/pred_labels + batch.boxes/labels），不再是 pass。
  - DetectionMetrics 支持 use_torchmetrics=True 后端，缺依赖时打印警告回退到内置实现；文档说明内置实现的性能取舍。
  - GIoULoss docstring 明确职责边界：正负样本匹配（assigner）由 task 侧承担，Loss 仅计算已匹配对。
  - 新增 tests/test_metrics/test_stage5.py（5 项：分类准确率 / 分割 mIoU 完美 / 检测 mAP 完美与空 / MetricCollection 检测分支），全量 68 passed。
- [x] 阶段 6：文档 / 测试 / 示例收尾
  - README 增补 v0.2.0 架构概览、config 驱动 CLI/编程用法、0.1.x→0.2.0 迁移章节。
  - demos/ 六个脚本与 verify_*.py 顶部加入迁移提示，引导到 tools/train.py 与 pytest。
  - 端到端验证：tools/train.py 分类 demo 训练 + --resume 从 checkpoint 续训均通过。
  - 最终全量测试 68 passed（2 条 DeprecationWarning 为归一化过渡路径的预期提示）。

---

# 训练子系统重构（v0.3.0）

> 目标：把"data/model/metrics 已重构、training 未达预期"的现状，升级为
> **统一 Trainer + Task 组件 + 模型层 Loss + utils 日志 + 多卡/工作目录隔离/自动续训**。
> 决策（用户确认）：Loss 放在 `dvisionix/models/losses/`（模型层，独立可组合组件，模型不内嵌 loss）；
> 直接移除旧 API 不保留垫片；work_dir 默认在代码库外；文档只同步本文件，docs 仅保留使用说明。

## 一、目标架构

```
dvisionix/
├── models/losses/          # Loss 作为模型层组件（独立、可继承、可自由组合）
│   ├── base.py             # BaseLoss + LossComposer + build_loss/build_losses + compute_loss
│   ├── classification.py   # CrossEntropy(含 label_smoothing) / Focal
│   ├── segmentation.py     # Dice / CombinedSegmentation
│   └── detection/          # GridAssigner / GridDetectionLoss / Objectness / GIoU/CIoU/L1Box
├── training/
│   ├── trainer.py          # 统一 Trainer：Task 驱动、metrics 接入、DDP、梯度累积 flush、history
│   ├── task.py             # BaseTask + 三大任务（optimizer/scheduler/loss/metrics 全配置化）
│   ├── optimizers.py       # OPTIMIZERS 注册表 + build_optimizer（adam/adamw/sgd/rmsprop）
│   ├── schedulers.py       # SCHEDULERS 注册表 + build_scheduler（cosine/plateau/step/multi_step/one_cycle）
│   ├── callbacks.py        # ProgressBar / ModelCheckpoint / EarlyStopping（统一走 utils.logging）
│   ├── workdir.py          # work_dir 解析、resume 三态、config dump
│   ├── builder.py          # build_callbacks / build_trainer
│   └── evaluation.py       # 检测评估（复用 metrics）
├── utils/logging/          # 日志/可视化唯一权威（console+file+JSONL+TensorBoard）
└── tools/train.py          # 配置驱动薄入口（--config/--cfg-options/--resume/--work-dir/--devices）
```

## 二、关键设计

1. **统一 Trainer / Task 组件**：Trainer 是纯执行引擎；任务差异全部由 Task 承载
   （分类/检测/分割/自定义），通过 `build_task(cfg)` 配置驱动。
2. **Loss 与 training 解耦**：`training/losses.py` 已删除；Loss 是 `models/losses` 组件，
   继承 `BaseLoss` 实现 `forward(preds, targets, **kwargs)`，`LossComposer` 加权组合，
   模型保持纯前向不内嵌 loss；`training/losses` 旧路径不再存在（直接移除）。
3. **配置闭环**：`training.optimizer/scheduler`、`loss`、`metrics`、`work_dir`、`resume`、
   `strategy/devices` 全部生效（此前 optimizer/scheduler/loss 为死配置）。
4. **验证指标接入训练循环**：`validation_step` 返回 `{"loss","preds","targets"}`，
   Trainer 喂给 Task 持有的 MetricCollection，epoch 末 `on_validation_epoch_end()` 输出
   accuracy/mAP/mIoU 等真实指标。
5. **日志在 utils**：`utils/logging/`（TrainingLogger：console + file + JSONL + TensorBoard）；
   全库去 print；旧 `TensorBoardLogger` 回调与 `utils/visualization.Visualizer` 已移除。
6. **多卡**：`strategy=ddp` + `devices`，DDP 包装、DistributedSampler（drop_last=True 防死锁）、
   rank0 专属保存/日志、验证指标 rank0 聚合（all_gather_object）；`test_ddp_smoke.py` 无卡自动跳过。
7. **工作目录隔离**：默认 `~/dvisionix_runs/<experiment>/<ts>`（代码库外），
   可经 CLI/配置/`DVISIONIX_WORK_DIR` 覆盖；合成数据缓存移入 work_dir；`runs/` 入 .gitignore。
8. **自动续训**：`resume: false|auto|latest|<path>`；checkpoint 含
   model/optimizer/scheduler/scaler/callbacks/rng/epoch/step；`config.resolved.yaml` dump 保证可复现。

## 三、执行记录

- [x] 阶段 A：Loss 模块独立
  - 新建 `dvisionix/models/losses/`（BaseLoss/LossComposer/build_losses/compute_loss +
    分类/分割/检测损失 + GridAssigner），全部注册 LOSSES。
  - 删除 `dvisionix/training/losses.py`；`dvisionix/__init__` 改为从 `models.losses` 导出 build_loss。
- [x] 阶段 B：Task 组件化与配置化
  - 新增 optimizers.py / schedulers.py 注册表；BaseTask 支持 optimizer_cfg/scheduler_cfg/loss/metrics。
  - 三大任务全部配置化；DetectionTask 改用 GridAssigner + GridDetectionLoss；
    验证循环返回 preds/targets 并接入 MetricCollection。
- [x] 阶段 C：统一 Trainer 增强
  - 验证指标进 epoch 日志；梯度累积 epoch 末 flush；`torch.set_grad_enabled` 改上下文；
    删除 train_logs/val_logs 死字段，fit 返回 history；删除 LearningRateScheduler 双轨逻辑。
- [x] 阶段 D：日志/可视化统一到 utils
  - 新建 `utils/logging/`（logger/tensorboard/training）；TrainingLogger（console+file+JSONL+TB）；
    callbacks 全走 logger；移除 TensorBoardLogger 回调与 Visualizer。
- [x] 阶段 E：多卡训练（DDP）
  - Trainer 支持 strategy=ddp/devices；DistributedSampler（drop_last=True）、rank0 保存、指标聚合；
    新增 test_ddp_smoke.py（无 GPU 自动 skip，多卡机器 torchrun 验证）。
- [x] 阶段 F：工作目录隔离 + 自动续训
  - workdir.py（默认 ~/dvisionix_runs/<exp>/<ts>，代码库外）；resume 三态 + 完整状态 + config dump。
- [x] 阶段 G：端到端接线、迁移与收尾
  - tools/train.py 重写（work_dir/resume/devices/strategy/loss 接线）；三个 demo 端到端跑通
    （分类 acc、检测 mAP、分割 mIoU 均出现在验证日志）。
  - 修复历史 bug：合成检测数据 boxes 应为 numpy；BaseDataset 支持 mask 路径加载。
  - demos：删除 train_from_config.py（引用已删除的 TaskType）与 cifar10_demo.py（旧 Visualizer）；
    修复 train_detection/segmentation 对 TensorBoardLogger 的引用。
  - 版本号 0.3.0；全量测试 144 passed（含 1 个无卡自动跳过的 DDP 冒烟）。

## 四、遗留事项（后续可选）

- [ ] 多卡实机验证：`torchrun --nproc_per_node=2 tools/train.py --config ... --devices 0,1`
      + `tests/test_training/test_ddp_smoke.py`。
- [ ] CI 落地：GitHub Actions 跑 ruff/black/pytest --cov（ruff 当前环境未安装）。
- [ ] 检测算法升级：GeneralizedModel 检测分支补 decode/评估；assigner 扩展（ATSS/FCOS 风格）。

---

# 结构/配置/导出优化（v0.3.1）

> 三项收尾优化（用户确认全部按推荐方案）：training 目录重组、config schema 增强、export 重构。

## 一、training 目录重组
- 目标：目录表达"任务/回调是组件族"的架构语义，扩展点清晰（新增任务/回调 = 新增一个文件）。
- 结果：
  ```
  training/
    trainer.py            # 统一 Trainer
    builder.py / workdir.py / evaluation.py
    tasks/                # BaseTask + 分类/检测/分割 + build_task（主扩展点）
    callbacks/            # CallbackList + ProgressBar/ModelCheckpoint/EarlyStopping（主扩展点）
    optim/                # OPTIMIZERS / SCHEDULERS 注册表 + build_*
  ```
- 顶层 API 不变（`from dvisionix.training import Trainer, ClassificationTask, build_task, ...`）；
  不保留转发垫片，仅同步 verify_all_modules.py 与内部导入。
- 测试：training 相关 51 passed + 1 skipped。

## 二、config 增强
- 新增 `dvisionix/config/schema.py`：轻量 schema 校验（必填/类型/取值/未知键告警/别名提示），
  无重依赖；`Config.validate_schema(task_type)` 入口接入 tools/train.py 并记录告警。
- `_parse_cli_value` 支持 YAML 子集（`[0,1]` / `{...}`），`--cfg-options training.devices=[0,1]` 可用；
  `dvisionix.config` 导出 `parse_cli_options`。
- 死键清理：删除 `logging.tensorboard/log_dir`、`data.mean/std`、`model.pretrained/backbone`
  （产物统一进 work_dir，归一化走 transforms 默认，保持单一事实来源）。
- 双通道归一：`training.learning_rate/weight_decay` 降级为便捷别名（同时存在 optimizer.lr 时告警），
  权威字段 `training.optimizer.lr/weight_decay`；默认配置与 demo 配置已迁移。
- 测试：tests/test_config/test_schema.py（未知键告警/类型错误/别名/CLI list-dict）。

## 三、export 重构
- 实测发现的问题：多输入模型无法导出（TypeError）；dict 输出能导但输出名自动生成、verify 静默丢输出；
  verify 仅单输入单输出；无任务感知；无测试覆盖。
- 结果：`ONNXExporter` 支持 input_shapes / 自定义 dummy_inputs；输出自适应
  （Tensor / tuple / list / dict 按键命名）；多输入多输出 verify；`backend='trace'|'dynamo'`
  （dynamo 需 onnxscript，缺失给清晰提示）；normalize/metadata 写入 ONNX metadata_props。
- 新增 tests/test_export/test_onnx_export.py（8 passed + 1 skipped）。
- docs/model_export.md 更新：分类 / 检测（说明导出 raw preds）/ 自定义多输入 / dict 输出四类示例。

## 四、验证
- 全量测试：见下方最终结果。
- README 架构树 / 项目结构章节同步更新。

## 五、遗留事项
- [ ] 多卡实机验证（torchrun + tests/test_training/test_ddp_smoke.py）。
- [ ] CI 落地（ruff/black/pytest --cov；当前环境未装 ruff）。
- [ ] export 的 dynamo 后端需在安装 onnxscript 的环境补跑 tests/test_export。
- [ ] 检测算法升级：GeneralizedModel 检测分支补 decode/评估；assigner 扩展（ATSS/FCOS 风格）。