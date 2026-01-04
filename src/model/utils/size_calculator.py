import torch.nn as nn
from typing import Tuple, List, Union

def calculate_conv2d_size(
    input_size: Tuple[int, int, int],
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    output_channels: int = None
) -> Tuple[int, int, int]:
    """计算卷积层输出尺寸

    Args:
        input_size: 输入尺寸 (C, H, W)
        kernel_size: 卷积核尺寸
        stride: 步长
        padding: 填充
        dilation: 空洞率
        output_channels: 输出通道数，如果为None则保持与输入相同

    Returns:
        输出尺寸 (C, H, W)
    """
    C, H, W = input_size
    
    # 标准化参数
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # 计算输出尺寸
    H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    
    # 输出通道数
    C_out = output_channels if output_channels is not None else C
    
    return (C_out, H_out, W_out)

def calculate_pool2d_size(
    input_size: Tuple[int, int, int],
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1
) -> Tuple[int, int, int]:
    """计算池化层输出尺寸

    Args:
        input_size: 输入尺寸 (C, H, W)
        kernel_size: 池化核尺寸
        stride: 步长，如果为None则等于kernel_size
        padding: 填充
        dilation: 空洞率

    Returns:
        输出尺寸 (C, H, W)
    """
    C, H, W = input_size
    
    # 标准化参数
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # 计算输出尺寸
    H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    
    return (C, H_out, W_out)

def calculate_upsample_size(
    input_size: Tuple[int, int, int],
    scale_factor: Union[int, Tuple[int, int]] = 2,
    mode: str = 'nearest'
) -> Tuple[int, int, int]:
    """计算上采样层输出尺寸

    Args:
        input_size: 输入尺寸 (C, H, W)
        scale_factor: 缩放因子
        mode: 上采样模式

    Returns:
        输出尺寸 (C, H, W)
    """
    C, H, W = input_size
    
    if isinstance(scale_factor, int):
        scale_factor = (scale_factor, scale_factor)
    
    H_out = H * scale_factor[0]
    W_out = W * scale_factor[1]
    
    return (C, H_out, W_out)

def calculate_linear_size(
    input_size: Tuple[int, int, int],
    output_features: int
) -> Tuple[int, int, int]:
    """计算线性层输出尺寸（假设输入已展平）

    Args:
        input_size: 输入尺寸 (C, H, W)
        output_features: 输出特征数

    Returns:
        输出尺寸 (output_features, 1, 1)
    """
    return (output_features, 1, 1)

def calculate_sequential_size(
    input_size: Tuple[int, int, int],
    layers: List[nn.Module]
) -> Tuple[int, int, int]:
    """计算序列层输出尺寸

    Args:
        input_size: 输入尺寸 (C, H, W)
        layers: 层列表

    Returns:
        输出尺寸 (C, H, W)
    """
    current_size = input_size
    
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            current_size = calculate_conv2d_size(
                current_size,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.out_channels
            )
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            current_size = calculate_pool2d_size(
                current_size,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation
            )
        elif isinstance(layer, nn.Upsample):
            current_size = calculate_upsample_size(
                current_size,
                layer.scale_factor,
                layer.mode
            )
        elif isinstance(layer, nn.Linear):
            current_size = calculate_linear_size(current_size, layer.out_features)
        # 对于其他层，假设不改变尺寸
    
    return current_size

def check_size_compatibility(
    output_size: Tuple[int, int, int],
    input_size: Tuple[int, int, int]
) -> bool:
    """检查两个尺寸是否兼容

    Args:
        output_size: 输出尺寸 (C, H, W)
        input_size: 输入尺寸 (C, H, W)

    Returns:
        是否兼容
    """
    return output_size == input_size

def print_size_info(name: str, input_size: Tuple[int, int, int], output_size: Tuple[int, int, int]):
    """打印尺寸信息

    Args:
        name: 层名称
        input_size: 输入尺寸
        output_size: 输出尺寸
    """
    print(f"{name}:")
    print(f"  输入: {input_size}")
    print(f"  输出: {output_size}")
    print(f"  兼容性: {'✓' if check_size_compatibility(output_size, input_size) else '✗'}")
    print()