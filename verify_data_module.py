# D:\ZhaoyangProject\DVisionix\verify_data_module.py

"""
验证数据模块是否正常工作
"""

import os
import sys
import tempfile
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("DVisionix 数据模块验证")
print("=" * 60)

# 1. 测试导入
print("\n1. 测试模块导入...")
try:
    from dvisionix.data import (
        BaseDataset,
        TaskType,
        DataFormat,
        CustomDataset,
        DatasetFactory,
        ClassificationTransforms,
    )
    print("   ✅ 导入成功")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    sys.exit(1)

# 2. 测试基础类型
print("\n2. 测试基础类型...")
try:
    assert TaskType.CLASSIFICATION.value == "classification"
    df = DataFormat(num_classes=10)
    assert df.num_classes == 10
    print("   ✅ 基础类型正常")
except Exception as e:
    print(f"   ❌ 基础类型错误: {e}")

# 3. 测试自定义数据集
print("\n3. 测试自定义数据集...")
try:
    import cv2
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试图像
        for i in range(5):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(tmpdir, f"img_{i}.png"), img)
        
        samples = [
            {"image_path": os.path.join(tmpdir, f"img_{i}.png"), "label": i % 2}
            for i in range(5)
        ]
        
        dataset = CustomDataset(
            task_type="classification",
            samples=samples,
            num_classes=2,
        )
        
        assert len(dataset) == 5
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape[0] == 3  # CHW 格式
        
        print(f"   ✅ 数据集大小: {len(dataset)}")
        print(f"   ✅ 图像形状: {sample['image'].shape}")
        print(f"   ✅ 标签: {sample['label'].item()}")
except Exception as e:
    print(f"   ❌ 自定义数据集错误: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试变换
print("\n4. 测试数据变换...")
try:
    transforms = ClassificationTransforms(train=True, image_size=32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img_path = os.path.join(tmpdir, "test.png")
        cv2.imwrite(img_path, img)
        
        samples = [{"image_path": img_path, "label": 0}]
        dataset = CustomDataset(
            task_type="classification",
            samples=samples,
            num_classes=2,
            transforms=transforms,
        )
        
        sample = dataset[0]
        assert sample["image"].shape[1] == 32
        assert sample["image"].shape[2] == 32
        print(f"   ✅ 变换后图像形状: {sample['image'].shape}")
except Exception as e:
    print(f"   ❌ 变换错误: {e}")

print("\n" + "=" * 60)
print("✅ 数据模块验证完成！")
print("=" * 60)