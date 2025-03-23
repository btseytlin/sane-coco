try:
    import pandas as pd
except ImportError:
    pass

try:
    import duckdb
except ImportError:
    pass

from typing import Dict, List, Union, Any
from .models import Annotation, BBox, Category, Image, Polygon, RLE
from .validation import (
    validate_sections_exist,
    validate_images,
    validate_annotations,
    validate_unique_ids,
)
from .util import group_annotations_by_image


class COCODataset:
    def __init__(
        self,
        categories: Dict[str, Category],
        images: List[Image],
        annotations: List[Annotation],
    ):
        self.categories = categories
        self.images = images
        self.annotations = annotations

        self.link_images_and_annotations()

    def get_category_by_id(self, category_id: int) -> Category:
        try:
            return next(c for c in self.categories.values() if c.id == category_id)
        except StopIteration:
            raise ValueError(f"Category {category_id} not found")

    @classmethod
    def find_category_by_id(
        cls, category_id: int, categories: Dict[str, Category], annotation_id: int
    ) -> Category:
        try:
            return next(c for c in categories.values() if c.id == category_id)
        except StopIteration:
            raise ValueError(
                f"Annotation {annotation_id} references non-existent category {category_id}"
            )

    @classmethod
    def parse_category(cls, cat_data: dict) -> Category:
        take_keys = ["id", "name", "supercategory"]
        return Category(**{k: v for k, v in cat_data.items() if k in take_keys})

    @classmethod
    def parse_image(cls, img_data: dict) -> Image:
        take_keys = ["id", "file_name", "width", "height"]
        return Image(
            annotations=[], **{k: v for k, v in img_data.items() if k in take_keys}
        )

    @classmethod
    def parse_annotation(
        cls,
        ann_data: dict,
        image_map: Dict[int, Image],
        categories: Dict[str, Category],
    ) -> Annotation:
        image = image_map[ann_data["image_id"]]
        category = cls.find_category_by_id(
            ann_data["category_id"], categories, ann_data["id"]
        )

        bbox = BBox(*ann_data["bbox"])

        # Handle segmentation if present
        segmentation = None
        area = ann_data.get("area")
        if "segmentation" in ann_data:
            seg_data = ann_data["segmentation"]
            if isinstance(seg_data, list):
                # Polygon format
                if len(seg_data) > 0 and isinstance(seg_data[0], list):
                    # Multiple polygons, use the first one for now
                    segmentation = Polygon(seg_data[0])
                elif len(seg_data) >= 6:  # At least 3 points (x,y pairs)
                    segmentation = Polygon(seg_data)
            elif isinstance(seg_data, dict):
                # RLE format
                seg_data["area"] = area
                segmentation = RLE.from_dict(seg_data)

        # Handle crowd flag
        iscrowd = ann_data.get("iscrowd", 0) == 1

        return Annotation(
            id=ann_data["id"],
            bbox=bbox,
            category=category,
            image=image,
            segmentation=segmentation,
            iscrowd=iscrowd,
            saved_area=area,
        )

    @classmethod
    def parse_categories(cls, category_data: List[dict]) -> Dict[str, Category]:
        categories = {}
        for cat_data in category_data:
            category = cls.parse_category(cat_data)
            categories[cat_data["name"]] = category
        return categories

    @classmethod
    def parse_images(
        cls, image_data: List[dict]
    ) -> tuple[List[Image], Dict[int, Image]]:
        images = []
        image_map = {}
        for img_data in image_data:
            image = cls.parse_image(img_data)
            images.append(image)
            image_map[image.id] = image
        return images, image_map

    @classmethod
    def parse_annotations(
        cls,
        annotation_data: List[dict],
        image_map: Dict[int, Image],
        categories: Dict[str, Category],
    ) -> List[Annotation]:
        annotations = []
        for ann_data in annotation_data:
            annotation = cls.parse_annotation(ann_data, image_map, categories)
            annotations.append(annotation)
        return annotations

    def link_images_and_annotations(
        self,
    ) -> None:
        for image in self.images:
            image.annotations = []

        grouped = group_annotations_by_image(self.annotations)
        for image in self.images:
            image.annotations = grouped.get(image.id, [])

    @classmethod
    def validate_data_dict(cls, data: dict) -> None:
        validate_sections_exist(data)
        validate_images(data["images"])
        validate_annotations(data["annotations"], data["images"], data["categories"])

        validate_unique_ids(data["images"], "image")
        validate_unique_ids(data["categories"], "category")
        validate_unique_ids(data["annotations"], "annotation")

    @classmethod
    def from_dict(cls, data: dict) -> "COCODataset":
        cls.validate_data_dict(data)

        categories = cls.parse_categories(data["categories"])
        images, image_map = cls.parse_images(data["images"])
        annotations = cls.parse_annotations(data["annotations"], image_map, categories)

        instance = cls(categories, images, annotations)
        return instance

    def to_pandas(self, group_by_image: bool = False):
        if group_by_image:
            return self.to_pandas_by_image()
        return self.to_pandas_by_annotation()

    def to_pandas_by_image(self):
        data = []
        for img in self.images:
            data.append(self.get_image_row(img))
        return pd.DataFrame(data)

    def get_image_row(self, img):
        return {
            "image_id": img.id,
            "image_file_name": img.file_name,
            "image_width": img.width,
            "image_height": img.height,
            "annotations": [ann.to_dict() for ann in img.annotations],
        }

    def to_pandas_by_annotation(self):
        data = []
        for ann in self.annotations:
            data.append(self.get_annotation_row(ann))
        return pd.DataFrame(data)

    def get_annotation_row(self, ann):
        return {
            "image_id": ann.image.id,
            "image_file_name": ann.image.file_name,
            "image_width": ann.image.width,
            "image_height": ann.image.height,
            "category_id": ann.category.id,
            "category_name": ann.category.name,
            "category_supercategory": ann.category.supercategory,
            "annotation_id": ann.id,
            "bbox_x": ann.bbox.x,
            "bbox_y": ann.bbox.y,
            "bbox_width": ann.bbox.width,
            "bbox_height": ann.bbox.height,
            "annotation_area": ann.area,
            "annotation_iscrowd": ann.iscrowd,
        }

    def get_annotation_dicts(self) -> List[List[dict]]:
        return [
            [
                {
                    "category": ann.category.name,
                    "bbox": ann.bbox.xywh,
                    "area": ann.area if ann.area else None,
                    "iscrowd": 1 if ann.iscrowd else 0,
                }
                for ann in img.annotations
            ]
            for img in self.images
        ]

    def to_dict(self) -> dict:
        categories_list = [category.to_dict() for category in self.categories.values()]
        images_list = [image.to_dict() for image in self.images]
        annotations_list = [ann.to_dict() for ann in self.annotations]

        return {
            "categories": categories_list,
            "images": images_list,
            "annotations": annotations_list,
        }

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, group_by_image: bool = False
    ) -> "COCODataset":
        if group_by_image:
            return cls.from_pandas_by_image(df)
        return cls.from_pandas_by_annotation(df)

    @classmethod
    def from_pandas_by_image(cls, df: pd.DataFrame) -> "COCODataset":
        categories = {}
        images = []
        annotations = []

        for _, row in df.iterrows():
            img = Image(
                id=row["image_id"],
                file_name=row["image_file_name"],
                width=row["image_width"],
                height=row["image_height"],
                annotations=[],
            )
            images.append(img)

            for ann_data in row["annotations"]:
                cat_id = ann_data["category_id"]
                if cat_id not in categories:
                    categories[cat_id] = Category(
                        id=cat_id,
                        name=ann_data.get("category_name", str(cat_id)),
                        supercategory=ann_data.get("category_supercategory", "default"),
                    )

                ann = Annotation(
                    id=ann_data["id"],
                    bbox=BBox(*ann_data["bbox"]),
                    category=categories[cat_id],
                    image=img,
                    iscrowd=ann_data.get("iscrowd", 0) == 1,
                )
                annotations.append(ann)
                img.annotations.append(ann)

        return cls(categories, images, annotations)

    @classmethod
    def from_pandas_by_annotation(cls, df: pd.DataFrame) -> "COCODataset":
        categories = {}
        images = {}
        annotations = []

        for _, row in df.iterrows():
            img_id = row["image_id"]
            if img_id not in images:
                images[img_id] = Image(
                    id=img_id,
                    file_name=row["image_file_name"],
                    width=row["image_width"],
                    height=row["image_height"],
                    annotations=[],
                )

            cat_id = row["category_id"]
            if cat_id not in categories:
                categories[cat_id] = Category(
                    id=cat_id,
                    name=row["category_name"],
                    supercategory=row["category_supercategory"],
                )

            bbox = BBox(
                x=row["bbox_x"],
                y=row["bbox_y"],
                width=row["bbox_width"],
                height=row["bbox_height"],
            )

            ann = Annotation(
                id=row["annotation_id"],
                bbox=bbox,
                category=categories[cat_id],
                image=images[img_id],
                iscrowd=row["annotation_iscrowd"],
            )
            annotations.append(ann)
            images[img_id].annotations.append(ann)

        return cls(
            {cat.name: cat for cat in categories.values()},
            list(images.values()),
            annotations,
        )

    @classmethod
    def from_pycocotools(cls, coco) -> "COCODataset":
        data = {
            "categories": list(coco.cats.values()),
            "images": list(coco.imgs.values()),
            "annotations": list(coco.anns.values()),
        }
        return cls.from_dict(data)

    def to_duckdb(self):
        con = duckdb.connect(":memory:")
        self.create_categories_table(con)
        self.create_images_table(con)
        self.create_annotations_table(con)
        return con

    def create_categories_table(self, con):
        con.execute(
            "CREATE TABLE categories (category_id INTEGER, name VARCHAR, supercategory VARCHAR)"
        )
        for c in self.categories.values():
            con.execute(
                "INSERT INTO categories VALUES (?, ?, ?)",
                [c.id, c.name, c.supercategory],
            )

    def create_images_table(self, con):
        con.execute(
            "CREATE TABLE images (image_id INTEGER, file_name VARCHAR, width INTEGER, height INTEGER)"
        )
        for i in self.images:
            con.execute(
                "INSERT INTO images VALUES (?, ?, ?, ?)",
                [i.id, i.file_name, i.width, i.height],
            )

    def create_annotations_table(self, con):
        con.execute(
            """CREATE TABLE annotations (
            annotation_id INTEGER, image_id INTEGER, category_id INTEGER,
            bbox_x FLOAT, bbox_y FLOAT, bbox_width FLOAT, bbox_height FLOAT,
            area FLOAT, iscrowd BOOLEAN)"""
        )
        for ann in self.annotations:
            bbox = ann.bbox.xywh
            con.execute(
                "INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    ann.id,
                    ann.image.id,
                    ann.category.id,
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    ann.area,
                    ann.iscrowd,
                ],
            )

    @classmethod
    def from_duckdb(cls, con) -> "COCODataset":
        categories = cls.load_categories(con)
        images, image_map = cls.load_images(con)
        annotations = cls.load_annotations(con, image_map, categories)
        return cls(categories, images, annotations)

    @staticmethod
    def load_categories(con) -> Dict[str, Category]:
        return {
            cat.name: cat
            for cat in (
                Category(id=row[0], name=row[1], supercategory=row[2])
                for row in con.execute("SELECT * FROM categories").fetchall()
            )
        }

    @staticmethod
    def load_images(con) -> tuple[List[Image], Dict[int, Image]]:
        images, image_map = [], {}
        for row in con.execute("SELECT * FROM images").fetchall():
            img = Image(
                id=row[0], file_name=row[1], width=row[2], height=row[3], annotations=[]
            )
            images.append(img)
            image_map[img.id] = img
        return images, image_map

    @staticmethod
    def load_annotations(
        con, image_map: Dict[int, Image], categories: Dict[str, Category]
    ) -> List[Annotation]:
        annotations = []
        for row in con.execute("SELECT * FROM annotations").fetchall():
            cat = next(c for c in categories.values() if c.id == row[2])
            bbox = BBox(x=row[3], y=row[4], width=row[5], height=row[6])
            ann = Annotation(
                id=row[0],
                image=image_map[row[1]],
                category=cat,
                bbox=bbox,
                iscrowd=row[8],
            )
            annotations.append(ann)
        return annotations

    @staticmethod
    def parse_simple_image(img_data: dict, image_id: int) -> Image:
        return Image(
            id=image_id,
            file_name=img_data["image_path"],
            width=img_data.get("width"),
            height=img_data.get("height"),
            annotations=[],
        )

    @staticmethod
    def get_or_create_category(
        cat_name: str, categories: dict, next_cat_id: int
    ) -> tuple[Category, int]:
        if cat_name not in categories:
            categories[cat_name] = Category(
                id=next_cat_id,
                name=cat_name,
                supercategory=cat_name,
            )
            next_cat_id += 1
        return categories[cat_name], next_cat_id

    @staticmethod
    def parse_simple_annotation(
        ann_data: dict,
        annotation_id: int,
        image: Image,
        category: Category,
    ) -> Annotation:
        bbox = BBox(*ann_data["bbox"])
        segmentation = None
        if "segmentation" in ann_data:
            segmentation = RLE.from_dict(ann_data["segmentation"])

        return Annotation(
            id=annotation_id,
            bbox=bbox,
            category=category,
            image=image,
            segmentation=segmentation,
            iscrowd=False,
            saved_area=ann_data.get("area"),
        )

    @classmethod
    def from_simple_dict(cls, data: List[dict]) -> "COCODataset":
        categories = {}
        images = []
        annotations = []
        next_cat_id = next_img_id = next_ann_id = 1

        for img_data in data:
            image = cls.parse_simple_image(img_data, next_img_id)
            images.append(image)

            for ann_data in img_data["annotations"]:
                category, next_cat_id = cls.get_or_create_category(
                    ann_data["category"], categories, next_cat_id
                )
                annotation = cls.parse_simple_annotation(
                    ann_data, next_ann_id, image, category
                )
                annotations.append(annotation)
                image.annotations.append(annotation)
                next_ann_id += 1

            next_img_id += 1

        return cls(categories, images, annotations)
