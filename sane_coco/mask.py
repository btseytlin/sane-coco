import numpy as np
from typing import List, Dict, Any, Union, Tuple
import itertools


class RLE:
    def __init__(self, counts: Union[List[int], str], size: Tuple[int, int]):
        self.counts = counts
        self.size = size


class MaskUtil:
    @staticmethod
    def encode(binary_mask: np.ndarray) -> Dict[str, Any]:
        h, w = binary_mask.shape
        return {"counts": [0, h*w], "size": (h, w)}
    
    @staticmethod
    def decode(rle: Dict[str, Any]) -> np.ndarray:
        h, w = rle["size"]
        return np.zeros((h, w), dtype=np.uint8)
    
    @staticmethod
    def area(rle: Dict[str, Any]) -> float:
        return 0.0
    
    @staticmethod
    def to_bbox(rle: Dict[str, Any]) -> np.ndarray:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    
    @staticmethod
    def iou(rle1: Dict[str, Any], rle2: Dict[str, Any], is_crowd: bool = False) -> float:
        return 0.0
    
    @staticmethod
    def merge(rles: List[Dict[str, Any]], intersect: bool = False) -> Dict[str, Any]:
        if not rles:
            return {"counts": [], "size": (0, 0)}
        return {"counts": [0, 1], "size": rles[0]["size"]}
    
    @staticmethod
    def from_polygon(polygons: List[List[float]], height: int, width: int) -> Dict[str, Any]:
        return {"counts": [0, height*width], "size": (height, width)}
    
    @staticmethod
    def _polygon_to_mask(polygon: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([]), np.array([])
    
    @staticmethod
    def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        return np.zeros(len(points), dtype=bool)