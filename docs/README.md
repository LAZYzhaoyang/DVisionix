# DVisionix 文档索引

## 快速开始
- [快速开始](quick_start.md) - 安装、Config 驱动训练、编程式训练
- [配置系统](config_system.md) - Config 加载、继承、CLI 覆盖、schema 校验

## 核心模块
- [数据模块](data.md) - Sample 协议、BaseDataset、原子变换、公开数据集工具箱与自定义任务的数据集
- [骨干网络 (timm)](backbones.md) - TimmBackbone / TimmClassifier 用法
- [自定义 Layer 与 Model](custom_models.md) - layers 模块、封装 timm 层、注册与配置驱动组装
- [指标 (Metrics)](metrics.md) - 原子指标、组合容器、预设与自定义
- [日志系统](logging.md) - 结构化日志 / JSONL / TensorBoard（TrainingLogger）
- [模型导出 (ONNX)](model_export.md) - ONNXExporter 导出与精度验证
- [语义分割任务](segmentation.md) - 分割数据格式与端到端训练
- [目标检测任务](detection.md) - 组件化检测器（FCOS/RetinaNet/YOLO/DETR/RT-DETR）、decode 与 mAP 评估

## 示例代码 (demos/)
- `tools/train.py` - 配置驱动统一训练入口（分类/检测/分割，支持 resume / DDP / work_dir）
- `demos/train_cifar10_new_trainer.py` - 编程式 CIFAR-10 训练示例
- `demos/export_onnx_demo.py` - ONNX 导出示例

## 说明
文档随功能持续补充。运行任何脚本前请先激活 conda 环境 `dvisionix`；测试使用 `pytest tests/`。