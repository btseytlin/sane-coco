import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class RLE:
    counts: List[int]
    shape: Tuple[int, int]
    
    @property
    def mask(self) -> 'Mask':
        array = np.zeros(self.shape, dtype=np.uint8)
        
        # This is a simplified placeholder for RLE decoding
        # In a real implementation, this would decode the RLE into a binary mask
        
        return Mask(array=array)
    
    @classmethod
    def from_mask(cls, mask: 'Mask') -> 'RLE':
        # This is a simplified placeholder for RLE encoding
        # In a real implementation, this would encode the binary mask as RLE
        
        return cls(
            counts=[0, mask.area],
            shape=mask.shape
        )
    
    def __eq__(self, other: 'RLE') -> bool:
        if not isinstance(other, RLE):
            return False
        return (self.shape == other.shape and 
                np.array_equal(self.counts, other.counts))


@dataclass
class Mask:
    array: np.ndarray
    
    def __post_init__(self):
        if self.array.dtype != np.uint8:
            self.array = self.array.astype(np.uint8)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape
    
    @property
    def height(self) -> int:
        return self.shape[0]
    
    @property
    def width(self) -> int:
        return self.shape[1]
    
    @property
    def area(self) -> int:
        return int(np.sum(self.array))
    
    @property
    def rle(self) -> RLE:
        return RLE.from_mask(self)
    
    def __and__(self, other: 'Mask') -> 'Mask':
        if self.shape != other.shape:
            raise ValueError(f"Mask shapes don't match: {self.shape} vs {other.shape}")
        return Mask(array=np.logical_and(self.array, other.array).astype(np.uint8))
    
    def __or__(self, other: 'Mask') -> 'Mask':
        if self.shape != other.shape:
            raise ValueError(f"Mask shapes don't match: {self.shape} vs {other.shape}")
        return Mask(array=np.logical_or(self.array, other.array).astype(np.uint8))
    
    def __xor__(self, other: 'Mask') -> 'Mask':
        if self.shape != other.shape:
            raise ValueError(f"Mask shapes don't match: {self.shape} vs {other.shape}")
        return Mask(array=np.logical_xor(self.array, other.array).astype(np.uint8))
    
    def __invert__(self) -> 'Mask':
        return Mask(array=np.logical_not(self.array).astype(np.uint8))
    
    def sum(self) -> int:
        return self.area
    
    def intersection(self, other: 'Mask') -> 'Mask':
        return self & other
    
    def union(self, other: 'Mask') -> 'Mask':
        return self | other
    
    def iou(self, other: 'Mask') -> float:
        intersection = (self & other).area
        union = (self | other).area
        return intersection / union if union > 0 else 0.0
    
    def crop(self, x: int, y: int, width: int, height: int) -> 'Mask':
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        width = max(1, min(width, self.width - x))
        height = max(1, min(height, self.height - y))
        
        return Mask(array=self.array[y:y+height, x:x+width])
    
    def resize(self, width: int, height: int) -> 'Mask':
        from PIL import Image
        pil_img = Image.fromarray(self.array)
        resized = pil_img.resize((width, height), Image.NEAREST)
        return Mask(array=np.array(resized))
    
    def to_polygons(self) -> List[List[float]]:
        # This is a simplified placeholder for contour extraction
        # In a real implementation, this would find contours in the mask
        # and convert them to COCO polygon format
        
        # Placeholder implementation
        return [[0, 0, 0, 10, 10, 10, 10, 0]]
    
    @classmethod
    def from_polygons(cls, polygons: List[List[float]], shape: Tuple[int, int]) -> 'Mask':
        # This is a simplified placeholder for polygon rendering
        # In a real implementation, this would render polygons into a binary mask
        
        array = np.zeros(shape, dtype=np.uint8)
        return cls(array=array)
    
    @classmethod
    def zeros(cls, height: int, width: int) -> 'Mask':
        return cls(array=np.zeros((height, width), dtype=np.uint8))
    
    @classmethod
    def ones(cls, height: int, width: int) -> 'Mask':
        return cls(array=np.ones((height, width), dtype=np.uint8))