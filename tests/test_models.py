import copy
import json
import numpy as np
from pathlib import Path
import pytest
from pycocotools.coco import COCO as OldCOCO
from pycocotools.cocoeval import COCOeval
from sane_coco import COCODataset, BBox, Mask, RLE, Polygon


def test_bbox_operations():
    bbox = BBox(100, 100, 50, 100)
    assert bbox.area == 5000
    assert bbox.corners == (100, 100, 150, 200)


def test_mask_operations():
    # Create a simple binary mask
    mask_array = np.zeros((10, 10), dtype=bool)
    mask_array[2:7, 2:7] = True
    mask = Mask(mask_array)

    # Test area calculation
    assert mask.area == 25

    # Test conversion to RLE
    rle = mask.to_rle()
    assert isinstance(rle, RLE)
    assert rle.size == (10, 10)

    # Test conversion to polygon
    polygon = mask.to_polygon()
    assert isinstance(polygon, Polygon)

    # Test round-trip conversion
    reconstructed_mask = rle.to_mask()
    assert np.array_equal(mask.array, reconstructed_mask.array)


def test_rle_operations():
    # Create an RLE representation
    counts = [0, 10, 5, 5, 5, 5]
    size = (10, 10)
    rle = RLE(counts=counts, size=size)

    # Test area calculation
    assert rle.area == 20

    # Test conversion to mask
    mask = rle.to_mask()
    assert isinstance(mask, Mask)
    assert mask.array.shape == size

    # Test conversion to polygon
    polygon = rle.to_polygon()
    assert isinstance(polygon, Polygon)

    # Test round-trip conversion
    reconstructed_rle = mask.to_rle()
    assert np.array_equal(rle.to_mask().array, reconstructed_rle.to_mask().array)


def test_polygon_operations():
    # Create a simple square polygon
    points = [10, 10, 20, 10, 20, 20, 10, 20]
    polygon = Polygon(points)

    # Test area calculation
    assert polygon.area == 100.0

    # Test conversion to mask
    mask = polygon.to_mask(size=(30, 30))
    assert isinstance(mask, Mask)
    assert mask.array.shape == (30, 30)

    # Test conversion to RLE
    rle = polygon.to_rle(size=(30, 30))
    assert isinstance(rle, RLE)
    assert rle.size == (30, 30)

    # Test round-trip conversion
    reconstructed_polygon = mask.to_polygon()
    reconstructed_mask = reconstructed_polygon.to_mask(size=(30, 30))
    assert np.sum(mask.array) == np.sum(reconstructed_mask.array)
