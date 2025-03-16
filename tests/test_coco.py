import copy
import json
import numpy as np
from pathlib import Path
import pytest
from pycocotools.coco import COCO as OldCOCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
from sane_coco import COCODataset, BBox


def test_complex_queries(sample_data):
    dataset = COCODataset.from_dict(sample_data)

    images_with_dog_and_cat = [
        img
        for img in dataset.images
        if "dog" in [ann.category.name for ann in img.annotations]
        and "cat" in [ann.category.name for ann in img.annotations]
    ]

    images_with_dog = [
        img
        for img in dataset.images
        if any(ann.category.name == "dog" for ann in img.annotations)
    ]

    images_with_dog_or_cat = [
        img
        for img in dataset.images
        if any(ann.category.name == "dog" for ann in img.annotations)
        or any(ann.category.name == "cat" for ann in img.annotations)
    ]

    crowded_images = [
        img
        for img in dataset.images
        if len([ann for ann in img.annotations if ann.category.name == "person"]) >= 2
    ]

    animal_annotations = [
        ann for ann in dataset.annotations if ann.category.name in ["dog", "cat"]
    ]

    assert len(images_with_dog_and_cat) == 0
    assert len(images_with_dog) == 1
    assert len(images_with_dog_or_cat) == 2
    assert len(crowded_images) == 0
    assert len(animal_annotations) == 2


def test_to_pandas(sample_data):
    dataset = COCODataset.from_dict(sample_data)

    # Test annotation-level dataframe
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(dataset.annotations)
    assert "image_id" in df.columns
    assert "image_file_name" in df.columns
    assert "image_width" in df.columns
    assert "image_height" in df.columns
    assert "category_id" in df.columns
    assert "category_name" in df.columns
    assert "category_supercategory" in df.columns
    assert "annotation_id" in df.columns
    assert "bbox_x" in df.columns
    assert "bbox_y" in df.columns
    assert "bbox_width" in df.columns
    assert "bbox_height" in df.columns
    assert "annotation_area" in df.columns
    assert "annotation_iscrowd" in df.columns

    # Test image-level dataframe
    img_df = dataset.to_pandas(group_by_image=True)
    assert isinstance(img_df, pd.DataFrame)
    assert len(img_df) == len(dataset.images)
    assert "image_id" in img_df.columns
    assert "image_file_name" in img_df.columns
    assert "image_width" in img_df.columns
    assert "image_height" in img_df.columns
    assert "annotations" in img_df.columns


class TestCOCODatasetValidation:
    def test_missing_image_fields(self):
        with pytest.raises(
            ValueError, match="Image missing required fields: id, width, height"
        ):
            COCODataset.from_dict(
                {
                    "images": [{"file_name": "test.jpg"}],
                    "categories": [],
                    "annotations": [],
                }
            )

    def test_missing_category_fields(self):
        with pytest.raises(
            ValueError, match="Category missing required fields: supercategory"
        ):
            COCODataset.from_dict(
                {
                    "images": [
                        {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                    ],
                    "categories": [{"id": 1, "name": "test"}],
                    "annotations": [],
                }
            )

    def test_invalid_bbox_format(self):
        with pytest.raises(ValueError, match="Invalid bbox format"):
            COCODataset.from_dict(
                {
                    "images": [
                        {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                    ],
                    "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3]}
                    ],
                }
            )

    def test_invalid_bbox_dimensions(self):
        with pytest.raises(ValueError, match="Invalid bbox dimensions"):
            COCODataset.from_dict(
                {
                    "images": [
                        {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                    ],
                    "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [10, 10, -5, 20],
                        }
                    ],
                }
            )

    def test_invalid_image_reference(self):
        with pytest.raises(
            ValueError, match="Annotation 1 references non-existent image 999"
        ):
            COCODataset.from_dict(
                {
                    "images": [
                        {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                    ],
                    "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 999,
                            "category_id": 1,
                            "bbox": [1, 2, 3, 4],
                        }
                    ],
                }
            )

    def test_invalid_category_reference(self):
        with pytest.raises(
            ValueError, match="Annotation 1 references non-existent category 999"
        ):
            COCODataset.from_dict(
                {
                    "images": [
                        {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                    ],
                    "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 999,
                            "bbox": [1, 2, 3, 4],
                        }
                    ],
                }
            )

    def test_duplicate_image_ids(self):
        with pytest.raises(ValueError, match="Duplicate image id: 1"):
            COCODataset.from_dict(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "test1.jpg",
                            "width": 100,
                            "height": 100,
                        },
                        {
                            "id": 1,
                            "file_name": "test2.jpg",
                            "width": 100,
                            "height": 100,
                        },
                    ],
                    "categories": [],
                    "annotations": [],
                }
            )

    def test_duplicate_category_ids(self):
        with pytest.raises(ValueError, match="Duplicate category id: 1"):
            COCODataset.from_dict(
                {
                    "images": [],
                    "categories": [
                        {"id": 1, "name": "test1", "supercategory": "test"},
                        {"id": 1, "name": "test2", "supercategory": "test"},
                    ],
                    "annotations": [],
                }
            )

    def test_duplicate_annotation_ids(self):
        with pytest.raises(ValueError, match="Duplicate annotation id: 1"):
            COCODataset.from_dict(
                {
                    "images": [
                        {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100},
                        {
                            "id": 2,
                            "file_name": "test2.jpg",
                            "width": 100,
                            "height": 100,
                        },
                    ],
                    "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [0, 0, 10, 10],
                        },
                        {
                            "id": 1,
                            "image_id": 2,
                            "category_id": 1,
                            "bbox": [0, 0, 10, 10],
                        },
                    ],
                }
            )

    def test_invalid_image_dimensions(self):
        with pytest.raises(ValueError, match="Invalid image dimensions: 0x100"):
            COCODataset.from_dict(
                {
                    "images": [
                        {"id": 1, "file_name": "test.jpg", "width": 0, "height": 100}
                    ],
                    "categories": [],
                    "annotations": [],
                }
            )

    def test_missing_annotation_fields(self):
        with pytest.raises(
            ValueError, match="Annotation missing required fields: bbox"
        ):
            COCODataset.from_dict(
                {
                    "images": [
                        {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}
                    ],
                    "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
                    "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
                }
            )
