# D:\ZhaoyangProject\DVisionix\dvisionix\utils\device.py

"""
设备管理工具

提供 CPU/GPU 的自动检测和无缝切换。
"""

import torch
from typing import Optional, Dict, Any


def get_device(device: str = "auto") -> torch.device:
    """
    自动选择设备
    
    Args:
        device: 设备类型
            - "auto": 自动检测（优先使用 GPU）
            - "cuda": 使用 CUDA GPU
            - "cpu": 使用 CPU
            - "cuda:0", "cuda:1": 指定 GPU 序号
    
    Returns:
        torch.device 对象
        
    Examples:
        >>> device = get_device("auto")  # 自动选择
        >>> model = model.to(device)
        >>> images = images.to(device)
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # 支持 Apple Silicon GPU
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    return torch.device(device)


def get_device_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    获取设备详细信息
    
    Args:
        device: torch.device 对象，为 None 时使用自动选择的设备
    
    Returns:
        设备信息字典
    """
    if device is None:
        device = get_device("auto")
    
    info = {
        "device": str(device),
        "type": device.type,
    }
    
    if device.type == "cuda":
        info.update({
            "cuda_available": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_memory_allocated": f"{torch.cuda.memory_allocated(device) / 1024**3:.2f} GB",
            "cuda_memory_reserved": f"{torch.cuda.memory_reserved(device) / 1024**3:.2f} GB",
        })
    elif device.type == "mps":
        info.update({
            "mps_available": True,
        })
    else:
        info.update({
            "cpu_only": True,
        })
    
    return info


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子（保证可复现性）
    
    Args:
        seed: 随机种子值
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    递归地将数据移动到指定设备
    
    支持 Tensor、字典、列表、元组等类型。
    
    Args:
        data: 要移动的数据
        device: 目标设备
    
    Returns:
        移动到目标设备后的数据
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(x, device) for x in data)
    else:
        return data