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


class TestCOCODatasetDuckDBConversion:
    def test_to_duckdb(self, sample_data):
        dataset = COCODataset.from_dict(sample_data)
        con = dataset.to_duckdb()

        # Verify table existence and schema
        tables = con.sql("SHOW TABLES").df()
        assert set(tables["name"]) == {"categories", "images", "annotations"}

        # Test categories table
        cat_df = con.sql("SELECT * FROM categories").df()
        assert len(cat_df) == len(dataset.categories)
        assert set(cat_df.columns) == {"category_id", "name", "supercategory"}
        first_cat = cat_df.iloc[0]
        assert first_cat["category_id"] == 1
        assert first_cat["name"] == "person"
        assert first_cat["supercategory"] == "person"

        # Test images table
        img_df = con.sql("SELECT * FROM images").df()
        assert len(img_df) == len(dataset.images)
        assert set(img_df.columns) == {"image_id", "file_name", "width", "height"}
        first_img = img_df.iloc[0]
        assert first_img["image_id"] == 1
        assert first_img["file_name"] == "000000001.jpg"
        assert first_img["width"] == 640
        assert first_img["height"] == 480

        # Test annotations table with foreign keys
        ann_df = con.sql(
            """
            SELECT a.*, i.file_name, i.width as image_width, i.height as image_height,
                   c.name as category_name, c.supercategory
            FROM annotations a
            JOIN images i ON a.image_id = i.image_id
            JOIN categories c ON a.category_id = c.category_id
        """
        ).df()
        assert len(ann_df) == len(dataset.annotations)
        first_ann = ann_df.iloc[0]
        assert first_ann["annotation_id"] == 1
        assert first_ann["image_id"] == 1
        assert first_ann["category_id"] == 1
        assert first_ann["bbox_x"] == 100
        assert first_ann["bbox_y"] == 100
        assert first_ann["bbox_width"] == 50
        assert first_ann["bbox_height"] == 100
        assert first_ann["area"] == 5000
        assert first_ann["iscrowd"] == False
        assert first_ann["file_name"] == "000000001.jpg"
        assert first_ann["category_name"] == "person"

    def test_from_duckdb(self, sample_data):
        dataset = COCODataset.from_dict(sample_data)
        con = dataset.to_duckdb()

        new_dataset = COCODataset.from_duckdb(con)
        assert len(new_dataset.images) == len(dataset.images)
        assert len(new_dataset.categories) == len(dataset.categories)
        assert len(new_dataset.annotations) == len(dataset.annotations)

        first_img = new_dataset.images[0]
        assert first_img.id == 1
        assert first_img.file_name == "000000001.jpg"
        assert first_img.width == 640
        assert first_img.height == 480

        first_ann = first_img.annotations[0]
        assert first_ann.id == 1
        assert first_ann.category.id == 1
        assert first_ann.category.name == "person"
        assert first_ann.category.supercategory == "person"
        assert first_ann.bbox.x == 100
        assert first_ann.bbox.y == 100
        assert first_ann.bbox.width == 50
        assert first_ann.bbox.height == 100
        assert first_ann.area == 5000
        assert first_ann.iscrowd == False


class TestCOCODatasetPandasConversion:
    def test_to_pandas(self, sample_data):
        dataset = COCODataset.from_dict(sample_data)

        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(dataset.annotations)
        first_row = df.iloc[0]
        assert first_row["image_id"] == 1
        assert first_row["image_file_name"] == "000000001.jpg"
        assert first_row["image_width"] == 640
        assert first_row["image_height"] == 480
        assert first_row["category_id"] == 1
        assert first_row["category_name"] == "person"
        assert first_row["category_supercategory"] == "person"
        assert first_row["annotation_id"] == 1
        assert first_row["bbox_x"] == 100
        assert first_row["bbox_y"] == 100
        assert first_row["bbox_width"] == 50
        assert first_row["bbox_height"] == 100
        assert first_row["annotation_area"] == 5000
        assert first_row["annotation_iscrowd"] == False

        img_df = dataset.to_pandas(group_by_image=True)
        assert isinstance(img_df, pd.DataFrame)
        assert len(img_df) == len(dataset.images)
        first_img_row = img_df.iloc[0]
        assert first_img_row["image_id"] == 1
        assert first_img_row["image_file_name"] == "000000001.jpg"
        assert first_img_row["image_width"] == 640
        assert first_img_row["image_height"] == 480
        assert len(first_img_row["annotations"]) == 2
        first_ann = first_img_row["annotations"][0]
        assert first_ann["id"] == 1
        assert first_ann["image_id"] == 1
        assert first_ann["category_id"] == 1
        assert first_ann["bbox"] == [100, 100, 50, 100]
        assert first_ann["area"] == 5000
        assert first_ann["iscrowd"] == 0

    def test_from_pandas(self, sample_data):
        dataset = COCODataset.from_dict(sample_data)
        df = dataset.to_pandas()

        new_dataset = COCODataset.from_pandas(df)
        assert len(new_dataset.images) == len(dataset.images)
        assert len(new_dataset.categories) == len(dataset.categories)
        assert len(new_dataset.annotations) == len(dataset.annotations)

        first_img = new_dataset.images[0]
        assert first_img.id == 1
        assert first_img.file_name == "000000001.jpg"
        assert first_img.width == 640
        assert first_img.height == 480

        first_ann = first_img.annotations[0]
        assert first_ann.id == 1
        assert first_ann.category.id == 1
        assert first_ann.category.name == "person"
        assert first_ann.category.supercategory == "person"
        assert first_ann.bbox.x == 100
        assert first_ann.bbox.y == 100
        assert first_ann.bbox.width == 50
        assert first_ann.bbox.height == 100
        assert first_ann.area == 5000
        assert first_ann.iscrowd == False

        img_df = dataset.to_pandas(group_by_image=True)
        new_dataset = COCODataset.from_pandas(img_df, group_by_image=True)
        assert len(new_dataset.images) == len(dataset.images)
        assert len(new_dataset.categories) == len(dataset.categories)
        assert len(new_dataset.annotations) == len(dataset.annotations)


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
