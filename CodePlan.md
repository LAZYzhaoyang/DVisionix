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

---

# 收尾清理与文档同步（v0.3.2）

## 一、脚本清理
- 删除已失效/冗余脚本：
  - `demos/train_detection.py`、`demos/train_segmentation.py`（配置键已迁移到 optimizer.lr 后失效，
    且与 `tools/train.py` 重复；训练入口统一为 tools/train.py）。
  - `verify_all_modules.py`、`verify_data_module.py`（已被 pytest 取代）。
  - 空目录 `demos/{classification,detection,segmentation}`。
- 保留并修复：`demos/train_cifar10_new_trainer.py`（build_dataset 改为 dict 调用、monitor 键 val_acc→accuracy、
  移除 BOM 与 DatasetFactory 旧引用）；`demos/export_onnx_demo.py` 验证可用。
- 清理仓库根残留产物：`checkpoints/ logs/ exports/ .cache/`（gitignored 运行时产物）。

## 二、文档同步
- 重写 `docs/README.md`（索引对齐现有文档与 demo）。
- 重写 `docs/quick_start.md`（tools/train.py 入口、--resume auto、多卡、编程式示例修正）。
- 重写 `docs/logging.md`（TrainingLogger / JSONL / TensorBoard，移除失效的 cfg.logging.log_dir 说明）。
- 修正 `docs/detection.md` / `docs/segmentation.md`（Sample 字段、transforms、tools/train.py 入口、
  配置化 optimizer/loss、移除失效 API 示例）。
- 修正 `docs/data.md`（移除 verify_data_module 引用）、`docs/config_system.md`（补 schema 校验与 CLI 覆盖）。
- 更新 `README.md`：文档索引表对齐真实文档；快速开始（optimizer.lr、--resume auto）；尾部迁移章节
  替换为「版本演进」（0.2.0 / 0.3.0 / 0.3.1）。

## 三、代码卫生
- 修复 `data/transforms/base.py` 重复 import。

## 四、验证
- 全量测试 161 passed + 2 skipped（DDP/dynamo 环境跳过）。
- `demos/export_onnx_demo.py` 端到端通过（导出 + onnxruntime verify，max_diff ~1e-8）。

---

# 模型模块丰富（v0.4.0）

> 用户决策：删除 GeneralizedModel（三合一万能模型，检测分支半成品），拆分为具体模型 + 共享脚手架；
> FCOS 与 RetinaNet 两种检测结构都保留（anchor-free / anchor-based）；ATSS assigner 一并实现（即插即用）；
> 分割做 DeepLabV3 / UNet / FCN；分类加 LinearClassifier / ArcFace / MultiLabel；教学模型迁入 models.toy。

## 一、目录结构（models）
```
models/
  base.py            # BaseModel（仅契约）
  classifiers.py     # LinearClassifier（backbone + 分类头组合）
  segmenters.py      # SegmentationModel（backbone + 分割头组合）
  toy/               # 教学模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel）
  backbones/  necks/  heads/  layers/  losses/  postprocess.py
  detectors/
    base.py          # SingleStageDetector（装配脚手架）
    anchors.py       # AnchorGenerator + bbox delta 编解码
    fcos.py          # FCOSDetector（anchor-free）
    retinanet.py     # RetinaNetDetector（anchor-based）
  heads/
    cls_head.py      # ClsHead / ArcFaceHead / MultiLabelHead
    seg_head.py      # SegHead / FCNHead / DeepLabV3Head(ASPP) / UNetDecoder
    det_head.py      # DetHead / FCOSHead / RetinaNetHead
  losses/detection/
    assigner.py      # GridAssigner / FCOSAssigner / MaxIoUAssigner / ATSSAssigner
    losses.py        # Objectness / GridDetection / SigmoidFocal / FCOSDetection / RetinaNetLoss
    box_loss.py      # GIoU / CIoU / L1Box
```

## 二、关键设计
1. **检测即插即用**：detector（forward 原始多尺度输出 + decode）+ assigner（可换）+ loss（可换）三者解耦；
   DetectionTask 统一契约（`model.decode` + `loss(preds, batch, image_hw, device)`），FCOS / RetinaNet / Grid 无缝接入。
2. **assigner 族**：FCOSAssigner（中心采样+尺度约束+min_pos 回退）、MaxIoUAssigner（RetinaNet 默认）、
   ATSSAssigner（自适应 top-k + mean/std 阈值，FCOS/RetinaNet 通用）。
3. **分割头族**：SegHead(1x1) / FCNHead / DeepLabV3Head(ASPP) / UNetDecoder（多尺度跳跃连接），
   SegmentationModel 自动根据 head 类型注入 in_channels 或 in_channels_list。
4. **教学模型独立**：toy 子包；顶层 re-export 兼容旧导入（dvisionix.models.SimpleCNN 仍可用）。
5. **配置 `_delete_` 语义**：父配置类型专属参数残留问题，用 `_delete_: true` 整体替换（mmcv 风格）。

## 三、新增配置示例
- configs/detection/fcos_synthetic.yaml（timm resnet18 + FPN + FCOSHead，loss fcos_detection）
- configs/detection/retinanet_synthetic.yaml（retinanet_head + AnchorGenerator，loss retinanet_detection / assigner max_iou）
- configs/segmentation/deeplabv3_synthetic.yaml（segmentation_model + deeplabv3_head）
- 三者均已端到端跑通（FCOS/RetinaNet 验证日志输出 mAP，DeepLabV3 输出 mIoU）。

## 四、修复的问题
- FCOS reg_loss NaN：min_pos 回退把框外位置选为正样本导致 log(负数)；修复为 clamp 非负 + exp 输出 clamp。
- 配置继承键残留：父配置 model/loss 段类型专属参数被合并进子配置（如 grid_detection 的 width 传入 fcos）；
  引入 `_delete_: true` 解决。

## 五、验证
- 全量测试：见最终结果（新增 test_detectors / test_segmenters / test_panet / test_classifiers / test_delete_marker）。

## 六、遗留 / 下一步
- DETR 系列（DETR / Deformable DETR / RT-DETR）与 YOLO 系列（YOLOv8 风格）作为下一步计划。
- 更多分割头：Mask2Former / SegFormer（语义+实例分割统一）。
- 更多分类头：度量学习变体（CosFace / SphereFace）、多标签任务化。
- GeneralizedModel 相关文档/测试引用清理已完成；README 版本演进已补充。

---

# 前沿模型扩展（v0.5.0）

> 用户决策：优先级 A(YOLO)→B(DETR)→C(分割头)→D(度量学习头)；全部完成后统一测试；
> heads 模块按任务重组为子包、每头一个文件。

## 一、新增内容
- 检测：`YOLODetector`（YOLOv8 风格，`YOLOHead` 解耦头 + `TaskAlignedAssigner` + `YOLOLoss` + `yolo_decode`）；
  `DETRDetector`（`DETRHead` transformer + `HungarianMatcher` + `DETRLoss` + `detr_decode` + `PositionEmbeddingSine`）。
- 分割：`SegFormerHead`（MLP 解码）、`MaskFormerHead`（query 掩码解码，compact，输出语义 logits 可直接用 SegmentationTask）；
  `SegmentationModel` 支持列表输入头（UNet/SegFormer/MaskFormer）。
- 分类：`CosFaceHead` / `SphereFaceHead` / `AdaFaceHead`（度量学习变体）；`MultiLabelTask`（多标签训练任务）。
- heads 目录重组：`classification/ segmentation/ detection/` 子包，每头一个文件，互不影响。

## 二、修复
- HungarianMatcher 在 n>m（query 多于 gt）时无增广路径导致死循环 → 代价矩阵补齐为方阵后求解。
- 清理超时遗留的孤儿 python 进程（Windows torch DLL 偶发加载失败与此相关）。

## 三、验证
- 阶段 A-D 各自冒烟通过（YOLO/DETR loss 下降、decode 形状正确；SegFormer/MaskFormer/度量头/多标签任务 forward 正确）。
- 统一全量测试：195 passed + 2 skipped。

## 四、遗留 / 下一步
- Mask2Former 完整版：mask 监督（匈牙利 mask 匹配）+ 实例/全景分割评估（COCO mask mAP）——当前为 compact 语义版。
- YOLO 配置示例（configs/detection/yolo_synthetic.yaml）可按需补充。
- 知识蒸馏回调（DistillCallback）可选。
- DETR 系列进阶：Deformable DETR / RT-DETR。

---

# 训练工程与 Mask2Former 完整版（v0.6.0）

## 一、训练工程
- Callback 新增 `on_validation_begin / on_validation_end` 钩子（EMA 等换权重用）。
- `EMA` 回调：影子权重 + 验证换入/恢复，随 checkpoint 保存 shadow。
- `DistillCallback` + `DistillationLoss`（CE + alpha*KL，温度缩放）；teacher logits 挂 `trainer.teacher_logits`。
- ModelCheckpoint 保留策略：`save_every_n_epochs` 周期存档 + `max_epoch_checkpoints` 上限清理。
- Trainer 训练结束导出 `work_dir/history.csv`。

## 二、Mask2Former 完整版
- MaskFormerHead 支持 `output_mode="full"`：返回 `pred_logits / pred_masks / semantic_logits`。
- `MaskFormerLoss`：匈牙利匹配（类别 + Dice 代价）+ CE + mask BCE + Dice。
- SegmentationModel 对 dict 输出透传；语义版（默认）仍可直接用 SegmentationTask。
- 实例/全景分割评估（COCO mask mAP）仍为后续计划。

## 三、CI
- `.github/workflows/ci.yml`：push/PR 触发，Python 3.10/3.11 CPU 矩阵，ruff + black + pytest。
- 本机未安装 ruff/black，CI 需在 GitHub 上跑通（本地仅保证代码可测）。

## 四、多卡验证（实现但不验证，隔离）
- DDP 代码路径已隔离：`strategy='auto'` 在无多卡时自动降级 `none`；`strategy='ddp'` 且无 CUDA 时明确报错；
  `tests/test_training/test_ddp_smoke.py` 无多卡自动 skip，不影响正常测试（200 passed + 2 skipped）。
- 多卡验证步骤（待用户多卡机器执行）：
  ```
  torchrun --nproc_per_node=2 tools/train.py --config configs/classification/demo_synthetic.yaml --devices 0,1
  torchrun --nproc_per_node=2 -m pytest tests/test_training/test_ddp_smoke.py -q
  ```

## 五、验证
- 新增 tests/test_training/test_v05_engineering.py（EMA / history.csv / checkpoint 保留 / 蒸馏 / MaskFormerLoss）。
- 统一全量测试：200 passed + 2 skipped。

---

# RT-DETR / mask mAP / 本地 lint（v0.7.0）

## 一、DETR 进阶（RT-DETR-lite）
- `RTDETRHead`：多尺度特征混合编码器（投影+融合）-> 类别分数 top-k query 选择 -> transformer 解码器；
  输出契约与 DETR 一致（logits/boxes），直接复用 `DETRLoss` 与 `detr_decode`。
- `RTDETRDetector` 注册为 `rtdetr`；`SingleStageDetector` 支持列表输入头（in_channels_list 自动注入）。
- 说明：为 compact 实现（conv 融合替代可变形注意力）；真 Deformable DETR 的可变形注意力留作后续。

## 二、实例/语义 mask mAP
- `MaskAveragePrecision`（metrics/segmentation.py）：COCO 风格 mask mAP（IoU 0.5:0.95，101-point 插值）。
- `maskformer_decode`（models/heads/segmentation/maskformer.py）：MaskFormerHead full 模式 -> (masks, scores, labels)。
- `evaluate_mask_ap`（training/evaluation）：端到端评估（目标 mask 自动对齐预测分辨率）。

## 三、本地 lint 工具链落地
- 安装 ruff 0.16.1 + black 26.5.1 到 dvisionix 环境。
- ruff：修复 400+ 问题（未用导入/变量、E741、分号、尾随空白、导入排序），per-file ignore 仅限
  __init__ 的 E402 与 tests 的 E402/E702/E741；最终 `ruff check dvisionix tools tests` 全绿。
- black：全仓格式化（93 文件），`black --check` 通过，与 CI 一致。

## 四、回调透传 batch
- `on_batch_begin/end` 增加可选 `batch` 参数（Trainer 传入），DistillCallback 据此计算 teacher logits。

## 五、验证
- 新增 tests/test_models/test_v07_rtdetr_mask.py（RT-DETR forward/decode/loss 下降、mask mAP 完美/错误、evaluate_mask_ap）。
- 全量测试 204 passed + 2 skipped；ruff / black 全绿。


---

# decode 归位与组合性验证（v0.7.1）

> 用户决策：模型专属 decode 从共享 postprocess 移回各模型文件（便于定位与维护）；
> 验证 heads 与不同 backbone/neck 的组合性；同步 CodePlan 与使用文档。

## 一、decode 归位（postprocess 只保留共享原语）

- 背景：postprocess.py 曾集中承载各模型专属 decode，模型逻辑分散、定位维护困难。
- 结果：
  - `postprocess.py` 仅保留共享原语：`nms / batched_nms / box_iou`（不依赖 torchvision.ops）。
  - `detr_decode` → `models/detectors/base.py`（DETR / RT-DETR 共用，输出契约一致）；
    `fcos_decode` → `models/detectors/fcos.py`；`retinanet_decode` → `models/detectors/retinanet.py`；
    `yolo_decode` → `models/detectors/yolo.py`；`maskformer_decode` → `models/heads/segmentation/maskformer.py`。
  - 各检测器 `decode()` 方法就近调用本文件 decode；`training/evaluation.py` 改为从模型侧导入。
  - 顶层 API 兼容：`dvisionix.models.fcos_decode / retinanet_decode / yolo_decode / detr_decode / maskformer_decode`
    仍可直接导入（detectors / heads 子包导出 + models/__init__ re-export）。
  - 顺带修复重构隐患：retinanet_decode 内错误的惰性导入路径（`.detectors.anchors` 会导致运行时 ImportError，
    且与模块级导入重复）；统一 detectors/base.py 的 `__all__`；补齐 fcos/yolo/retinanet 缺失的 `import torch`。

## 二、heads 与 backbone/neck 组合性验证

- 冒烟验证 27 组组合全部通过（CPU）：
  - 检测：FCOS / RetinaNet / YOLO × {SequentialBackbone, TimmBackbone(resnet18)} × {无 neck, FPN, PANet}；
    DETR × {seq, timm} × {无 neck, FPN}；RT-DETR × {seq, timm}（直连骨干多尺度）。
  - 分割：Seg / FCN / DeepLabV3 / UNetDecoder / SegFormer / MaskFormer × {seq, timm}。
  - 分类：LinearClassifier × {timm, seq} × {ClsHead / ArcFace / CosFace / SphereFace / AdaFace / MultiLabel}。
- 结论：组件化装配器（SingleStageDetector / SegmentationModel / LinearClassifier）对不同 backbone / neck / head
  即插即用，无需为具体模型改造；多尺度头（UNetDecoder / SegFormerHead / MaskFormerHead / RTDETRHead）自动注入
  `in_channels_list`，单尺度头注入 `in_channels`。

## 三、已知边界与后续优化

- ~~RT-DETR 暂不支持 neck~~ ✅ 已解决（阶段 A：支持 neck，多尺度通道自动对齐 neck 输出）。
- ~~多尺度头识别靠装配器硬编码名单~~ ✅ 已解决（阶段 A：head 类属性 `input_style="multi_scale"` 自声明，装配器统一读取）。
- ~~head 注册名风格不统一~~ ✅ 已解决（阶段 A：分类头补 `_head` 后缀别名 `arcface_head / cosface_head / sphereface_head / adaface_head / multi_label_head`，旧名保留兼容）。

## 四、验证

- 全量测试 204 passed + 2 skipped；ruff / black 全绿（与 CI 一致）。
- decode 函数位置断言：detr/fcos/retinanet/yolo/maskformer decode 均从对应模型文件导出。

## 五、下一步计划
## 六、阶段 A 实施记录（已完成）

> 用户确认后实施：A1 统一 decode 契约 / A2 input_style 自声明 / A3 RT-DETR neck + 注册名统一。

### A1：统一 `.decode()` 契约
- `MaskFormerHead` 新增 `decode()` 实例方法（委托模块级 `maskformer_decode`，纯函数实现 + 实例桥接）。
- `SegmentationModel` 新增 `decode()` 透传（head 支持解码时委托；否则 `NotImplementedError` 清晰报错）。
- `evaluate_mask_ap` 优先复用 `model.decode()`（`getattr(model, "decode", None) or maskformer_decode` 兜底）。

### A2：多尺度头 input_style 自声明机制
- `UNetDecoder / SegFormerHead / MaskFormerHead / RTDETRHead` 增加类属性 `input_style = "multi_scale"`。
- `SingleStageDetector` / `SegmentationModel` 装配器统一读取该属性决定注入 `in_channels_list` / `in_channels`，
  删除 `_LIST_INPUT_HEADS` 硬编码名单与 `rtdetr_head` 特判；`SegmentationModel.forward` 的多尺度分发同样改为读取属性。
- 效果：新增多尺度头只需声明 `input_style`，注册即用、零装配器改动。

### A3：RT-DETR 可选 neck + 注册名统一
- RT-DETR 现可接 FPN / PANet：装配器按 neck 输出通道自动推导 `in_channels_list`
  （int 通道 × num_outs；列表通道直接用），修复了此前 RT-DETR + neck 的通道错配隐患。
- 分类头补 `_head` 后缀别名（`arcface_head` 等 5 个），旧名（`arcface` 等）保留兼容，现有配置不受影响。

### 验证
- 新增 `tests/test_models/test_v071_phase_a.py`（6 项）：input_style 属性、neck 注入、maskformer 经模型 decode、
  单尺度头 decode 报错、RT-DETR + FPN、分类头别名。
- 组合冒烟补充：RT-DETR + PANet、MaskFormer + FPN 通过（neck 通道自动对齐）。
- 全量测试 210 passed + 2 skipped；ruff / black 全绿。


- 多卡实机验证：`torchrun --nproc_per_node=2 tools/train.py --config ... --devices 0,1` + test_ddp_smoke.py（DDP 已隔离实现）。
- DETR 系列：Deformable DETR（可变形注意力）、RT-DETR 完整版（neck 对齐 / 更强编码器）。
- 分割：Mask2Former 实例/全景分割评估打通（mask 监督已就绪）；更多分割头（按任务目录每头一文件扩展）。
- 检测：YOLO 系列扩展（YOLOv5 / v7 / v9 风格变体）、更多 anchor-free 检测器。
- 指标：分类 / 分割指标全面迁移 torchmetrics（可选后端）。


---

# 阶段 B：模型库扩充（v0.8.0）

> 用户确认实施：按"算法工具箱"方向扩充模型库，全部模块化、即插即用；教学模型独立；规模大不是问题。

## 一、B-分割：新分割头 + 实例/全景
- 新头（均注册 HEADS，接入 SegmentationModel 即插即用）：
  - `PSPHead`（psp_head）：金字塔场景解析池化（1/2/3/6 bins），单尺度输入。
  - `UPerNetHead`（upernet_head）：FPN 风格自顶向下多尺度融合 + 顶层 PPM，input_style="multi_scale"。
  - `DeepLabV3PlusHead`（deeplabv3plus_head）：高层 ASPP + 低层特征解码器（取 in_channels_list[-2]），input_style="multi_scale"。
- `MaskFormerTask`（实例分割任务）：MaskFormerHead full 模式 + `MaskFormerLoss` + mask mAP 验证；已注册 TASKS 并在 training 顶层导出。
- 全景分割评估打通：
  - `PanopticQuality`（metrics/panoptic.py，注册 panoptic_quality）：标准 PQ / SQ / RQ（按类 IoU>=0.5 贪心匹配）。
  - `panoptic_decode` / `evaluate_panoptic`（training/evaluation.py）：full 模式 preds -> 全景 id 图 -> PQ 评估（GT 可取 batch["panoptic"] 或退化为语义）。

## 二、B-检测：YOLO 系列 + Deformable DETR
- 新层（注册 LAYERS，可直接用于 SequentialBackbone stages）：
  - `CSPLayer`（csp_layer）：CSP 瓶颈块（YOLOv5 风格）。
  - `ELANLayer`（elan_layer）：高效聚合网络块（YOLOv7/v9 风格）。
- 新配置示例：`configs/detection/yolov5_synthetic.yaml`（CSP 骨干 + PANet + YOLOHead）、
  `yolov9_synthetic.yaml`（ELAN 骨干 + PANet + YOLOHead）。
- `DeformableDETRHead`（deformable_detr_head）+ `DeformableDETRDetector`（deformable_detr）：
  纯 PyTorch 多尺度可变形注意力（`MultiScaleDeformableAttention`，layers 注册），输出契约与 DETR 一致，
  复用 DETRLoss 与 detr_decode（compact，head 间取平均简化）。

## 三、B-分类：新度量学习头
- `NormFaceHead`（normface_head）：归一化特征/权重 + 缩放，无 margin。
- `CurricularFaceHead`（curricularface_head）：课程式自适应 margin（compact：困难样本 margin 更大）。
- `PartialFCHead`（partial_fc_head）：大规模类别采样子集 softmax；默认全量可用（配合 LinearClassifier +
  ClassificationTask），采样模式返回 (logits_subset, sampled_indices) + `remap_labels` 辅助。

## 四、验证
- 新增 `tests/test_models/test_v08_phase_b.py`（13 项）：新分割头前向/多尺度注入、MaskFormerTask 训练/验证、
  PanopticQuality 完美/错误、panoptic_decode 形状、CSP/ELAN 层、YOLOv5 风格检测器、Deformable DETR 前向/解码/损失下降、
  NormFace/CurricularFace、PartialFC 全量/采样模式。
- 全量测试 223 passed + 2 skipped；ruff / black 全绿。
- 修复过程中问题：UPerNet 融合循环逐级对齐；panoptic_decode 输出上采样到图像分辨率；
  PartialFC 采样从非 batch 类别中抽取保证恰好 num_sample_classes。

## 五、下一步计划（阶段 C/D）
- 多卡实机验证（torchrun + test_ddp_smoke.py；DDP 已隔离实现）。
- 指标：分类/分割指标迁移 torchmetrics 可选后端。
- 训练/评估收尾：实例/全景评估接入训练验证循环（evaluate_panoptic 已就绪，可挂进 MaskFormerTask 或自定义回调）。
- 文档/CI 收尾：README/CodePlan 同步、版本号、CI 全绿。


---

# 阶段 C：推荐组合实施（v0.9.0）

> 用户确认：按推荐组合实施（Mask2Former 完整版 + RT-DETR 增强版 + 全景评估接入验证循环）；
> 指标迁移 torchmetrics 列为未来优化项，暂不实施。

## 一、C1：全景评估接入验证循环
- `MaskFormerTask` 新增 `panoptic: bool = False` / `id_scale` 开关：
  - `validation_step` 在 mask mAP 基础上追加 `panoptic_decode` 全景 id 图（preds 扩展为 4 元组，targets 为 3 元组）；
  - GT 优先取 `batch["panoptic"]`，缺省退化为 `batch["mask"] * id_scale`；
  - `update_metrics` 同时喂 `PanopticQuality`；`on_validation_epoch_end` 合并输出 `PQ / SQ / RQ`；
  - `reset_metrics` 用 getattr 保护（BaseTask.__init__ 在子类属性初始化前调用）。
- 顺带修复 mask mAP 评估的真值格式：`MaskAveragePrecision` 期望 `target_masks: List[(M,H,W)]`，
  `evaluate_mask_ap` 与 `MaskFormerTask` 现统一构造 (1,H,W) 单实例掩码（或 instance_masks），并把 GT 对齐到预测分辨率。

## 二、C2：RT-DETR 增强版
- 新增 `RTDETRFullHead`（rtdetr_full_head）/ `RTDETRFullDetector`（rtdetr_full），保留 compact 版（rtdetr）不变：
  - 多尺度**可变形编码器**（复用 MultiScaleDeformableAttention，替代简单卷积融合）；
  - **IoU-aware query selection**：选择头预测类别 + 框 + IoU，score = class_score * iou 取 top-k；
  - 解码器以选择出的 token 内容为初始 tgt、预测框中心为参考点，框经细化输出；
  - 输出契约与 DETR 一致，复用 DETRLoss / detr_decode。

## 三、C3：Mask2Former 完整版
- 新增 `Mask2FormerHead`（mask2former_head）：
  - FPN 风格像素解码器 + **mask attention 解码器**（交叉注意力受上一轮预测掩码约束，逐层细化）；
  - 输出契约与 MaskFormerHead full 模式一致（pred_logits / pred_masks / semantic_logits），
    复用 MaskFormerLoss / maskformer_decode / panoptic_decode；
- `MaskFormerLoss` 支持真实实例 GT：`batch["instance_masks"]`（(N,H,W)）+ `batch["instance_labels"]`（(N,)），
  缺省回退语义连通域近似。

## 四、验证
- 新增 `tests/test_models/test_v09_phase_c.py`（6 项）：panoptic 验证循环开/关、RT-DETR 增强前向/解码/损失下降、
  Mask2Former 前向/解码/损失下降（语义 GT）、实例 GT 损失。
- 全量测试 229 passed + 2 skipped；ruff / black 全绿。
- 修复问题：MaskAveragePrecision 真值格式（List[(M,H,W)]）与分辨率对齐；BaseTask 初始化时序下的 reset_metrics 保护。

## 五、未来优化项（暂不实施，已记录）
- 分类/分割指标迁移 torchmetrics 可选后端。
- 多卡实机验证（DDP 已隔离实现）。
- YOLOv7/v10、CenterNet、更多分割头/度量头等模型扩充（按需排期）。


---

# 方向 3：模型继续扩充（v0.10.0）

> 用户确认方向 3：YOLOv7/v10、CenterNet、更多分割头、更多度量头。

## 一、YOLO 系列
- `EELANLayer`（eelan_layer）：E-ELAN 交叉融合层（YOLOv7 风格），注册 LAYERS，可直接作骨干 stage。
- `NMSFreeYOLOHead`（yolo_v10_head）+ `NMSFreeYOLODetector`（yolo_v10）：
  - 结构与 YOLOHead 一致，训练配合 `OneToOneYOLOLoss`（yolo_v10_detection，每个 GT 只匹配一个最高质量预测，分类 BCE + 框 GIoU）；
  - 推理免 NMS：逐层解码 + 跨层 top-k。
- 配置示例：`configs/detection/yolov7_synthetic.yaml`（E-ELAN + PANet + YOLOHead）、`yolov10_synthetic.yaml`（NMS-free）。

## 二、CenterNet
- `CenterNetHead`（centernet_head）：中心点热图 + 宽高 + 偏移（单尺度，兼容列表输入取末层）。
- `CenterNetLoss`（centernet_detection）：penalty-reduced Focal（热图）+ L1（宽高/偏移，仅中心点）。
- `CenterNetDetector`（centernet）：3x3 max-pool 峰值过滤 + 偏移修正 + 像素框解码。
- 配置示例：`configs/detection/centernet_synthetic.yaml`（stride-8 骨干）。

## 三、分割头
- `BiSeNetHead`（bisenet_head）：轻量实时分割（细节分支 + 全局上下文分支融合）。

## 四、分类头 / 损失
- `CircleLossHead`（circle_loss_head）+ `CircleLoss`（circle_loss）：Circle Loss 自适应 margin。
- `SimCLRHead`（simclr_head，MLP 投影头）+ `InfoNCELoss`（info_nce）：SimCLR 风格对比学习（双视角 InfoNCE）。

## 五、验证
- 新增 `tests/test_models/test_v010_direction3.py`（7 项）：E-ELAN、yolo_v10 前向/解码/损失下降、
  centernet 前向/解码/损失下降、新配置加载、BiSeNet、CircleLoss、SimCLR/InfoNCE。
- 全量测试 236 passed + 2 skipped；ruff / black 全绿。


---

# D-1 骨干库 / D-2 SimCLR / D-4 分割增强（v0.11.0）

> 用户确认：D-1 内置骨干（ConvNeXt + CSPDarknet + MobileNetV3；ViT 等 Transformer 骨干列入后续）；
> D-2 SimCLR 端到端按计划实施；D-3 YOLOv9-lite 仅出详细计划暂不实施；D-4 SwinUNet 风格 + SegFormer 变体实施；
> 多卡验证 / torchmetrics 迁移继续列为后续。

## 一、D-1：内置骨干库
- `FeatureBackboneBase`（feature.py）：内置骨干通用基类（stage 列表 + dry-run 通道推导 + features_only / out_channels / num_features）。
- `ConvNeXtBackbone`（convnext_backbone）：LN + 深度可分离 + 层缩放；stem 4x4 stride4 + 4 个 stage（stride 4/8/16/32）。
- `CSPDarknetBackbone`（cspdarknet_backbone）：YOLOv5 风格（6x6 stem + 4 个 3x3 stride2 + CSP，SiLU）。
- `MobileNetV3Backbone`（mobilenetv3_backbone）：MBConv 倒残差 + SE（stride 2/4/8/16/32）。
- 新增层：`ConvNeXtBlock`（convnext_block）、`MBConvBlock`（mbconv_block），均注册 LAYERS。
- 全部支持分类 / 检测 / 分割即插即用（与 LinearClassifier / SingleStageDetector / SegmentationModel 组合验证）。

## 二、D-2：SimCLR 端到端
- `SimCLRTransforms`（simclr_transforms）：双视角增强（随机缩放裁剪 / 翻转 / 色彩抖动 / 归一化），输出 image1/image2。
- `SimCLRTask`（training/tasks/simclr.py）：InfoNCELoss 自监督训练（兼容装配器 num_classes 注入）。
- `tools/train.py` 支持 task_type="simclr"（变换 + 任务映射 + 合成数据）。
- 配置示例：`configs/classification/simclr_synthetic.yaml`。

## 三、D-4：分割增强
- `SegFormerV2Head`（segformer_v2_head）：overlap patch embed（4x4 stride2）+ MixFFN（LN + 3x3 DWConv + MLP）。
- `SwinUNetDecoder`（swin_unet_decoder）：PatchExpand（LN + Linear(2x) + PixelShuffle(2)）逐级上采样 + 跳连融合。

## 四、D-3：YOLOv9-lite 详细计划（待实施）
目标：NMS-free 之外的检测能力再进一步，用 PGI（Programmable Gradient Information）提升浅层梯度质量。
1. **可逆分支（RevCol 风格简化）**：新增 `layers/reversible.py` —— `ReversibleBlock`（add 耦合 + 显式保存激活，训练时可重计算，节省显存）；骨干末端加 1 层可逆块组。
2. **PGI 辅助损失**：浅层特征加辅助预测头（`YOLOAuxHead`，结构与 YOLOHead 一致），损失与主头按比例组合（`YOLOAuxLoss`：主损失 + aux_weight * 辅助损失），只参与训练、推理时丢弃。
3. **检测器**：`YOLOv9Detector`（yolo_v9）：E-ELAN 骨干（已有）+ 可逆分支 + YOLOHead（主）+ aux 头（训练）。
4. **配置**：`configs/detection/yolov9_synthetic.yaml`（复用现有合成数据）。
5. **验证**：forward / decode / 主+辅损失下降；aux 头在推理 forward 中不参与输出。
6. 工作量：中-大（可逆块 + 辅助头 + 损失装配）。

## 五、后续计划（未实施）
- ViT / Swin 等 Transformer 骨干（BACKBONES 扩展）。
- 多卡实机验证（DDP 已隔离实现）。
- 分类 / 分割指标迁移 torchmetrics 可选后端。

## 六、验证
- 新增 `tests/test_models/test_v011_d14.py`（8 项）：三个内置骨干分类+检测组合、SegFormerV2 / SwinUNet、SimCLRTransforms 双视角、SimCLRTask 训练下降、simclr 配置加载。
- 全量测试 244 passed + 2 skipped；ruff / black 全绿。
