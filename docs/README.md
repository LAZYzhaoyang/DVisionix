# DVisionix 文档索引

## 快速开始
- [快速开始](quick_start.md) - 几分钟上手 DVisionix

## 核心模块
- [配置系统](config_system.md) - Config 加载、继承、合并、验证
- [骨干网络（timm）](backbones.md) - TimmBackbone / TimmClassifier 用法
- [自定义 Layer 与 Model](custom_models.md) - layers 模块、封装 timm 层、注册与配置驱动组装
- [日志系统](logging.md) - 结构化日志与按阶段记录指标
- [模型导出（ONNX）](model_export.md) - ONNXExporter 导出与精度验证
- [语义分割任务](segmentation.md) - 分割数据格式与端到端训练
- [目标检测任务](detection.md) - 网格检测器、collate 与端到端训练

## 示例代码（demos/）
- `demos/train_from_config.py` - Config 驱动的端到端训练
- `demos/cifar10_demo.py` - CIFAR-10 分类示例
- `demos/export_onnx_demo.py` - ONNX 导出示例
- `demos/train_segmentation.py` - 语义分割端到端训练
- `demos/train_detection.py` - 目标检测端到端训练

## 说明
文档随功能持续补充。运行任何脚本前请先激活 conda 环境 `dvisionix`。
