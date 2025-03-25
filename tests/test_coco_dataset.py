import copy
import json
import numpy as np
from pathlib import Path
import pytest
from pycocotools.coco import COCO as OldCOCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
from sane_coco import COCODataset, BBox, RLE


class TestCOCODatasetOperations:
    def test_complex_queries(self, sample_data):
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
            if len([ann for ann in img.annotations if ann.category.name == "person"])
            >= 2
        ]

        animal_annotations = [
            ann for ann in dataset.annotations if ann.category.name in ["dog", "cat"]
        ]

        assert len(images_with_dog_and_cat) == 0
        assert len(images_with_dog) == 1
        assert len(images_with_dog_or_cat) == 2
        assert len(crowded_images) == 0
        assert len(animal_annotations) == 2

    def test_duckdb_operations(self, sample_data):
        dataset = COCODataset.from_dict(sample_data)
        con = dataset.to_duckdb()

        large_objects = con.sql(
            """
            SELECT i.file_name, a.bbox_width * a.bbox_height as obj_size, c.name
            FROM annotations a
            JOIN images i ON a.image_id = i.image_id
            JOIN categories c ON a.category_id = c.category_id
            WHERE a.bbox_width * a.bbox_height > 2500
            ORDER BY obj_size DESC
        """
        ).df()
        assert len(large_objects) > 0

        category_stats = con.sql(
            """
            SELECT c.name, COUNT(*) as count, 
                   AVG(a.bbox_width * a.bbox_height) as avg_size
            FROM annotations a
            JOIN categories c ON a.category_id = c.category_id
            GROUP BY c.name
            ORDER BY count DESC
        """
        ).df()
        assert len(category_stats) == len(dataset.categories)

        crowded_images = con.sql(
            """
            SELECT i.file_name, COUNT(*) as obj_count
            FROM images i
            JOIN annotations a ON i.image_id = a.image_id
            GROUP BY i.file_name
            HAVING COUNT(*) >= 2
            ORDER BY obj_count DESC
        """
        ).df()
        assert len(crowded_images) > 0

        overlapping_objects = con.sql(
            """
            SELECT i.file_name, COUNT(DISTINCT a1.annotation_id) as overlap_count
            FROM annotations a1
            JOIN annotations a2 ON a1.image_id = a2.image_id 
                AND a1.annotation_id < a2.annotation_id
                AND (a1.bbox_x < a2.bbox_x + a2.bbox_width)
                AND (a1.bbox_x + a1.bbox_width > a2.bbox_x)
                AND (a1.bbox_y < a2.bbox_y + a2.bbox_height)
                AND (a1.bbox_y + a1.bbox_height > a2.bbox_y)
            JOIN images i ON a1.image_id = i.image_id
            GROUP BY i.file_name
        """
        ).df()
        assert isinstance(overlapping_objects, pd.DataFrame)


class TestCOCODatasetDuckDBConversion:
    def test_to_duckdb(self, sample_data):
        dataset = COCODataset.from_dict(sample_data)
        con = dataset.to_duckdb()

        tables = con.sql("SHOW TABLES").df()
        assert set(tables["name"]) == {"categories", "images", "annotations"}

        cat_df = con.sql("SELECT * FROM categories").df()
        assert len(cat_df) == len(dataset.categories)
        assert set(cat_df.columns) == {"category_id", "name", "supercategory"}
        first_cat = cat_df.iloc[0]
        assert first_cat["category_id"] == 1
        assert first_cat["name"] == "person"
        assert first_cat["supercategory"] == "person"

        img_df = con.sql("SELECT * FROM images").df()
        assert len(img_df) == len(dataset.images)
        assert set(img_df.columns) == {"image_id", "file_name", "width", "height"}
        first_img = img_df.iloc[0]
        assert first_img["image_id"] == 1
        assert first_img["file_name"] == "000000001.jpg"
        assert first_img["width"] == 640
        assert first_img["height"] == 480

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

    def test_from_simple_dict(self):
        simple_data = [
            {
                "image_path": "img1.jpg",
                "annotations": [
                    {
                        "category": "person",
                        "bbox": [100, 100, 50, 100],
                    },
                    {
                        "category": "dog",
                        "bbox": [200, 200, 30, 60],
                    },
                ],
            },
            {
                "image_path": "img2.jpg",
                "width": 800,
                "height": 600,
                "annotations": [
                    {
                        "category": "cat",
                        "bbox": [300, 300, 40, 80],
                    }
                ],
            },
        ]

        dataset = COCODataset.from_simple_dict(simple_data)

        assert len(dataset.images) == 2
        assert len(dataset.categories) == 3
        assert len(dataset.annotations) == 3

        assert dataset.images[0].file_name == "img1.jpg"
        assert len(dataset.images[0].annotations) == 2
        assert dataset.images[0].annotations[0].category.name == "person"
        assert dataset.images[0].annotations[1].category.name == "dog"

        assert dataset.images[1].file_name == "img2.jpg"
        assert dataset.images[1].width == 800
        assert dataset.images[1].height == 600
        assert len(dataset.images[1].annotations) == 1
        assert dataset.images[1].annotations[0].category.name == "cat"
        assert dataset.images[1].annotations[0].area == 40 * 80

    def test_from_simple_dict_with_area(self):
        simple_data = [
            {
                "image_path": "img1.jpg",
                "annotations": [
                    {
                        "category": "person",
                        "bbox": [100, 100, 50, 100],
                    },
                    {
                        "category": "dog",
                        "bbox": [200, 200, 30, 60],
                    },
                ],
            },
            {
                "image_path": "img2.jpg",
                "width": 800,
                "height": 600,
                "annotations": [
                    {
                        "category": "cat",
                        "bbox": [300, 300, 40, 80],
                        "area": 30 * 30,
                    }
                ],
            },
        ]

        dataset = COCODataset.from_simple_dict(simple_data)

        assert len(dataset.images) == 2
        assert len(dataset.categories) == 3
        assert len(dataset.annotations) == 3

        assert dataset.images[0].file_name == "img1.jpg"
        assert len(dataset.images[0].annotations) == 2
        assert dataset.images[0].annotations[0].category.name == "person"
        assert dataset.images[0].annotations[1].category.name == "dog"

        assert dataset.images[1].file_name == "img2.jpg"
        assert dataset.images[1].width == 800
        assert dataset.images[1].height == 600
        assert len(dataset.images[1].annotations) == 1
        assert dataset.images[1].annotations[0].category.name == "cat"
        assert dataset.images[1].annotations[0].area == 30 * 30

    def test_from_simple_dict_with_rle(self):
        simple_data = [
            {
                "image_path": "img1.jpg",
                "width": 10,
                "height": 10,
                "annotations": [
                    {
                        "category": "person",
                        "bbox": [1, 1, 5, 5],
                        "segmentation": {
                            "counts": [0, 25, 10, 5, 10],
                            "size": [10, 10],
                        },
                    },
                    {
                        "category": "dog",
                        "bbox": [6, 6, 3, 3],
                        "segmentation": {"counts": [45, 5, 0], "size": [10, 10]},
                    },
                ],
            }
        ]

        dataset = COCODataset.from_simple_dict(simple_data)

        assert len(dataset.images) == 1
        assert len(dataset.categories) == 2
        assert len(dataset.annotations) == 2

        ann1 = dataset.images[0].annotations[0]
        assert ann1.category.name == "person"
        assert isinstance(ann1.segmentation, RLE)
        assert ann1.segmentation.counts == [0, 25, 10, 5, 10]
        assert ann1.segmentation.size == [10, 10]

        ann2 = dataset.images[0].annotations[1]
        assert ann2.category.name == "dog"
        assert isinstance(ann2.segmentation, RLE)
        assert ann2.segmentation.counts == [45, 5, 0]
        assert ann2.segmentation.size == [10, 10]
