"""
sane_coco: A more Pythonic implementation of pycocotools.

This package provides modern, intuitive interfaces for working with
COCO (Common Objects in Context) datasets.
"""

from .dataset import CocoDataset, Image, Annotation, Category, BBox
from .mask import Mask, RLE
from .eval import evaluate_detections, evaluate_segmentations, plot_precision_recall

__version__ = "0.1.0"

__all__ = [
    'CocoDataset', 
    'Image', 
    'Annotation', 
    'Category', 
    'BBox',
    'Mask',
    'RLE',
    'evaluate_detections',
    'evaluate_segmentations',
    'plot_precision_recall'
]