import copy
import json
import numpy as np
from pathlib import Path
import pytest
from pycocotools.coco import COCO as OldCOCO
from pycocotools.cocoeval import COCOeval
from sane_coco import COCODataset, BBox


def test_bbox_operations():
    bbox = BBox(100, 100, 50, 100)
    assert bbox.area == 5000
    assert bbox.corners == (100, 100, 150, 200)
