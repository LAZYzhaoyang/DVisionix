from ..base_dataset import BaseDataset
from typing import Dict, Any, List, Tuple

class DetDataset(BaseDataset):
    """Base class for detection datasets"""
    
    def __init__(self, *args, **kwargs):
        # Store category information
        self.categories: List[str] = []
        # Default input size for network
        self.input_size: Tuple[int, int] = (1024, 1024)
        # Minimum object size to be considered
        self.min_size: int = 1
        super().__init__(*args, **kwargs)
    
    def _format_bbox(self, bbox: List[float], format: str = 'xyxy') -> List[float]:
        """Convert bbox to target format (xyxy, xywh, cxcywh)"""
        return bbox
    
    def get_img_info(self, index: int) -> Dict[str, Any]:
        """Return image basic information (size, file_name, etc.)"""
        return self.data_infos[index]