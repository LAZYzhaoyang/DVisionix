# 变更日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
完整的规划与缺陷清单见 [CodePlan.md](CodePlan.md)。

## [Unreleased] — v1.1 稳定性与工程优化（进行中）

阶段 0-3 已完成；阶段 4（打包与 CI）、阶段 5（性能）待办。

### 新增

- **装配层门禁**：`tests/test_e2e_configs.py` 对 `configs/` 下每个可训练配置跑 1 个 epoch
  （此前所有「配置加载」测试都止步于 `build_model()`，导致 3 个官方配置在 286 条测试全绿时无法训练）。
- **模型契约门禁**：`tests/test_model_contracts.py` 断言所有检测器的 `decode()`
  接受统一关键字参数。
- **CPU gloo 双进程 DDP 一致性测试**：`tests/test_ddp_cpu.py`，无需 GPU 即可在 CI 长期运行。
- **ONNX 导出契约测试**、**checkpoint 契约测试**、**统一 Evaluator 契约测试**、
  **损失语义与裁剪契约测试**。
- `training.ema.*` 配置项（EMA 此前只有编程式入口）。
- LICENSE、CHANGELOG、Dependabot、发布工作流。

### 修复

| 编号 | 问题 |
|---|---|
| D1 | `centernet` / `nmsfree_yolo` 的 `decode()` 缺少 `iou_threshold`，验证阶段必然抛 `TypeError` |
| D2 | `yolov10_synthetic.yaml` 向 `OneToOneYOLOLoss` 传它不接受的 `topk` |
| D3 | `simclr` 不在合法任务类型中；且合成数据缺少 simclr 分支 |
| D4 | DINO 用 `feats[0].shape * 4` 当图像尺寸，官方配置下偏差 2 倍，去噪分支与主分支坐标系不一致 |
| D5 | DDP 聚合把检测的 `(boxes, scores, labels)` 3-tuple 拍平成 6-tuple |
| D6 | FCOS / YOLO 回归损失重复累加 L1（`use_giou=True` 实为 GIoU+L1，`False` 实为 2×L1） |
| D7 | 检测损失未按正样本归一化，量级随 batch size 与层数漂移 |
| D8 | `Trainer.validate()` 无 DDP 分支且不触发验证回调（EMA 权重交换失效） |
| D9 | 检测管线先 resize 到目标尺寸再同尺寸裁剪，随机裁剪完全失效 |
| D10 | `ToTensor` 可能把 mask 转成 float，破坏 `CrossEntropyLoss` 的 long 契约 |
| D11 | 首图空预测时 mask AP 的目标尺寸退化为 `(1, 1)` |
| D12 | ONNX 导出不支持嵌套输出；导出会原地修改调用者模型的设备与 train/eval 状态 |
| D13 | 合成数据 train/val 共用文件名序列，验证集是训练集的子集 |
| D14 | `dvisionix/config/defaults/*.yaml` 不进 wheel，非 editable 安装下 `Config.from_default()` 必然失败 |
| D15 | 依赖与版本号在 `setup.py` / `requirements.txt` 双源且已漂移 |
| D16 | CI 无 mypy / coverage / 安全检查，仅测 3.10 与 3.11 |
| D17 | EMA / `DistillCallback` 在配置驱动路径下不可达 |
| D18 | 任务类型常量存在三份且不一致（其中一份是从未被引用的死常量） |
| D19 | 3 处 docstring 机械损坏 |
| D20 | 测试只「构建模型」不「运行配置」，配置是否可用无人验证 |

另修复：梯度累积尾窗分母错误、`fit()` 重复调用 `on_validation_epoch_end()` 导致
history 中出现基于空累加器算出的指标列、两个「loss 下降」测试的顺序相关偶发失败。

### 变更（可能影响历史结果可比性）

- **损失语义**：FCOS / YOLO 默认改为 **GIoU only**（`giou_weight=1.0, l1_weight=0.0`）；
  旧的 `use_giou` 开关按字面意图迁移并发出 `DeprecationWarning`。
  v1.0.0 的检查点不可直接续训，既有超参需重新标定。
- **指标口径**：检测损失各分量改为真实均值；`val_loss` 与 v1.0.0 不可直接比较。
- **输出格式**：`history.csv` 的 epoch 级指标统一为 `val_` 前缀
  （`accuracy` → `val_accuracy`，与 `val_loss` 一致）。
- **增强生效**：检测训练管线改为先放大到 1.1× 再随机裁剪（此前增强完全未生效）。
- **裁剪契约**：`RandomCrop` / `CenterCrop` / `BoxSyncRandomCrop` 在输入过小时默认报错
  （`on_small="error"`），不再静默返回错误尺寸。
- **数据口径**：合成数据 train / val 使用不同 split 与独立随机种子（消除泄漏），
  且数据变为可复现。
- **resume 校验**：`load_checkpoint` 默认拒绝配置哈希 / 任务类型 / 模型类型不匹配的
  checkpoint，并拒绝纯 `state_dict`；可用 `allow_config_mismatch=True` 显式放行。
- **打包**：`setup.py` 已删除，元数据统一到 `pyproject.toml`。

## [1.0.0]

功能基线。配置驱动 + 组件化模型库 + 统一训练引擎 + 工具链，API 冻结进入稳定期。

- 12 种骨干、3 种颈部、35 个头、12 个检测器、25+ 损失
- 统一 Trainer：DDP、AMP、梯度累积、完整 resume、torch.compile、channels_last
- 分类 / 检测 / 语义分割 / 实例与全景分割 / 自监督对比学习 + 线性评估
- 286 条测试、`ruff` / `black` 全绿、10 篇专题文档

## [0.17.0]

训练工程 P2（超参搜索 / 特征蒸馏）+ P3（性能开关 / 实验管理）+ DINO look-forward-twice。

## [0.16.0]

DINO-lite、线性评估闭环、训练工程 P1。

## [0.15.0]

组合器目录化、model 分层重构、SwinV2 / DeformableV2 / SegFormerV3。

## [0.14.0]

中期模型扩充：ConvNeXtV2 / EfficientNetLite / MiT 骨干 + SwinUNet + YOLOv11。

## [0.13.0]

model 分层重构 + layers 统一 + 调用规则 R1-R6 入册。

## [0.12.0]

ViT / Swin 骨干、YOLOv9-lite(PGI)。

## [0.11.0]

内置骨干体系、SimCLR 端到端、分割增强。

## [0.10.0]

YOLOv7 / YOLOv10、CenterNet、BiSeNet、Circle / SimCLR 分类头。

## [0.9.0]

全景评估、RT-DETR-full、Mask2Former 完整版。

## [0.8.0]

PSP / UPerNet / DeepLabV3+、DeformableDETR、度量学习头。

## [0.7.x]

decode 归位到模型文件、组合性验证。

## [0.6.0]

Mask2Former 完整版、EMA / 蒸馏回调、CI。

## [0.5.0]

YOLO / DETR、SegFormer / MaskFormer、度量学习。

## [0.4.0]

模型模块丰富（FCOS / RetinaNet / UNet / DeepLab 等）。

## [0.3.0]

训练子系统重构：Task 组件化、loss 迁移到模型层、DDP / resume / work_dir。

## [0.2.0]

组件化重构：Registry + 配置驱动入口。
