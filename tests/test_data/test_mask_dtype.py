# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: mask dtype 契约测试（ToTensor / MaskToTensor 必须稳定输出 torch.long）。
"""mask dtype 契约测试（CodePlan 7.4 步骤 2-4 / 7.1.2 D10）。

分割标签必须满足 ``CrossEntropyLoss`` 的 long 契约。v1.0.0 有两个漏洞：

- ``ToTensor`` 只在 **2 维** 分支处理 dtype，3 维 mask 会走 float 分支变成 float32；
- ``MaskToTensor`` 只在输入**不是** Tensor 时才转换，已经是 float Tensor 的 mask
  会被原样放过去。

两者都会把错误推迟到 loss 计算时才炸（或更糟：静默断掉梯度），因此这里把
"mask 一律 long" 作为可执行契约固定下来。
"""

import numpy as np
import pytest
import torch

from dvisionix.data.transforms import MaskToTensor, ToTensor


@pytest.mark.unit
class TestToTensorMaskDtype:
    def test_2d_mask_becomes_long(self):
        out = ToTensor(keys=("image", "mask"))(
            {"image": np.zeros((8, 8, 3), dtype=np.uint8), "mask": np.zeros((8, 8), dtype=np.uint8)}
        )
        assert out["mask"].dtype == torch.long
        assert out["mask"].shape == (8, 8)

    def test_3d_mask_becomes_long(self):
        """v1.0.0 的漏洞：3 维 mask 会走 float 分支。"""
        out = ToTensor(keys=("mask",))({"mask": np.zeros((8, 8, 2), dtype=np.int32)})
        assert out["mask"].dtype == torch.long
        assert out["mask"].shape == (8, 8, 2)

    def test_float_mask_becomes_long(self):
        out = ToTensor(keys=("mask",))({"mask": np.zeros((8, 8), dtype=np.float32)})
        assert out["mask"].dtype == torch.long

    def test_image_still_becomes_float_chw(self):
        out = ToTensor(keys=("image",))({"image": np.zeros((8, 8, 3), dtype=np.uint8)})
        assert out["image"].dtype == torch.float32
        assert out["image"].shape == (3, 8, 8)


@pytest.mark.unit
class TestMaskToTensorContract:
    def test_already_long_tensor_is_kept(self):
        mask = torch.zeros(4, 4, dtype=torch.long)
        assert MaskToTensor()({"mask": mask})["mask"].dtype == torch.long

    def test_float_tensor_is_coerced_to_long(self):
        """v1.0.0 的漏洞：已经是 Tensor 的 float mask 会被原样保留。"""
        out = MaskToTensor()({"mask": torch.zeros(4, 4, dtype=torch.float32)})
        assert out["mask"].dtype == torch.long

    def test_numpy_mask_is_converted(self):
        out = MaskToTensor()({"mask": np.zeros((4, 4), dtype=np.uint8)})
        assert out["mask"].dtype == torch.long

    def test_missing_mask_is_noop(self):
        sample = {"image": torch.zeros(3, 4, 4)}
        assert MaskToTensor()(sample) is sample

    def test_non_integral_float_mask_raises(self):
        with pytest.raises(ValueError, match="非整数"):
            MaskToTensor()({"mask": torch.tensor([[0.5, 1.0]])})

    def test_negative_values_raise(self):
        with pytest.raises(ValueError, match="负值"):
            MaskToTensor()({"mask": torch.tensor([[-1, 0], [1, 2]])})

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="形状"):
            MaskToTensor()({"mask": torch.zeros(4)})

    @pytest.mark.parametrize("shape", [(4, 4), (4, 4, 3)])
    def test_valid_shapes_accepted(self, shape):
        out = MaskToTensor()({"mask": torch.zeros(*shape, dtype=torch.long)})
        assert tuple(out["mask"].shape) == shape
