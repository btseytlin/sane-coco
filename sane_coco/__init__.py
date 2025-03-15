from dataclasses import dataclass
from typing import Dict, List, Iterator
import pytest


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def corners(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class Category:
    id: int
    name: str
    supercategory: str


@dataclass
class Annotation:
    id: int
    bbox: BBox
    category: Category
    image: "Image"


@dataclass
class Image:
    id: int
    file_name: str
    width: int
    height: int
    annotations: List[Annotation]


class COCODataset:
    def __init__(self, categories: Dict[str, Category], images: List[Image]):
        self.categories = categories
        self.images = images

    @staticmethod
    def validate_required_fields(data: dict, required: List[str], entity: str) -> None:
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"{entity} missing required fields: {', '.join(missing)}")

    @staticmethod
    def validate_unique_id(id: int, used_ids: set[int], entity: str) -> None:
        if id in used_ids:
            raise ValueError(f"duplicate {entity} id: {id}")
        used_ids.add(id)

    @staticmethod
    def parse_category(cat_data: dict, category_ids: set[int]) -> Category:
        COCODataset.validate_required_fields(
            cat_data, ["id", "name", "supercategory"], "category"
        )
        COCODataset.validate_unique_id(cat_data["id"], category_ids, "category")
        return Category(**cat_data)

    @staticmethod
    def parse_image(img_data: dict, image_ids: set[int]) -> Image:
        COCODataset.validate_required_fields(
            img_data, ["id", "width", "height"], "image"
        )
        COCODataset.validate_unique_id(img_data["id"], image_ids, "image")
        if img_data["width"] <= 0 or img_data["height"] <= 0:
            raise ValueError(
                f"invalid image dimensions: {img_data['width']}x{img_data['height']}"
            )
        return Image(annotations=[], **img_data)

    @staticmethod
    def validate_bbox(bbox: List[float]) -> None:
        if len(bbox) != 4:
            raise ValueError("invalid bbox format")
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("invalid bbox dimensions")

    @staticmethod
    def find_category_by_id(
        category_id: int, categories: Dict[str, Category]
    ) -> Category:
        try:
            return next(c for c in categories.values() if c.id == category_id)
        except StopIteration:
            raise ValueError(
                f"annotation 1 references non-existent category {category_id}"
            )

    @staticmethod
    def parse_annotation(
        ann_data: dict,
        annotation_ids: set[int],
        image_map: Dict[int, Image],
        categories: Dict[str, Category],
    ) -> Annotation:
        COCODataset.validate_required_fields(
            ann_data, ["id", "image_id", "category_id", "bbox"], "annotation"
        )
        COCODataset.validate_unique_id(ann_data["id"], annotation_ids, "annotation")

        if ann_data["image_id"] not in image_map:
            raise ValueError(
                f"annotation {ann_data['id']} references non-existent image {ann_data['image_id']}"
            )

        image = image_map[ann_data["image_id"]]
        category = COCODataset.find_category_by_id(ann_data["category_id"], categories)
        COCODataset.validate_bbox(ann_data["bbox"])

        return Annotation(
            id=ann_data["id"],
            bbox=BBox(*ann_data["bbox"]),
            category=category,
            image=image,
        )

    @staticmethod
    def parse_categories(category_data: List[dict]) -> Dict[str, Category]:
        categories = {}
        category_ids = set()
        for cat_data in category_data:
            category = COCODataset.parse_category(cat_data, category_ids)
            categories[cat_data["name"]] = category
        return categories

    @staticmethod
    def parse_images(image_data: List[dict]) -> tuple[List[Image], Dict[int, Image]]:
        images = []
        image_map = {}
        image_ids = set()
        for img_data in image_data:
            image = COCODataset.parse_image(img_data, image_ids)
            images.append(image)
            image_map[image.id] = image
        return images, image_map

    @staticmethod
    def parse_annotations(
        annotation_data: List[dict],
        image_map: Dict[int, Image],
        categories: Dict[str, Category],
    ) -> List[Annotation]:
        annotation_ids = set()
        annotations = []
        for ann_data in annotation_data:
            annotation = COCODataset.parse_annotation(
                ann_data, annotation_ids, image_map, categories
            )
            annotations.append(annotation)
        return annotations

    @staticmethod
    def group_annotations_by_image(
        annotations: List[Annotation],
    ) -> Dict[int, List[Annotation]]:
        grouped = {}
        for annotation in annotations:
            if annotation.image.id not in grouped:
                grouped[annotation.image.id] = []
            grouped[annotation.image.id].append(annotation)
        return grouped

    def link_annotations(self, annotations: List[Annotation]) -> None:
        grouped = self.group_annotations_by_image(annotations)
        for image in self.images:
            image.annotations = grouped.get(image.id, [])

    @classmethod
    def from_dict(cls, data: dict) -> "COCODataset":
        categories = cls.parse_categories(data["categories"])
        images, image_map = cls.parse_images(data["images"])
        annotations = cls.parse_annotations(data["annotations"], image_map, categories)

        instance = cls(categories, images)
        instance.link_annotations(annotations)
        return instance
