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

    @property
    def iter_images(self) -> Iterator[Image]:
        return iter(self.images)

    @classmethod
    def from_dict(cls, data: dict) -> "COCODataset":
        categories = {}
        category_ids = set()
        for cat_data in data["categories"]:
            required_fields = ["id", "name", "supercategory"]
            missing = [f for f in required_fields if f not in cat_data]
            if missing:
                raise ValueError(
                    f"category missing required fields: {', '.join(missing)}"
                )
            if cat_data["id"] in category_ids:
                raise ValueError(f"duplicate category id: {cat_data['id']}")
            category_ids.add(cat_data["id"])
            categories[cat_data["name"]] = Category(**cat_data)

        images = []
        image_map = {}
        image_ids = set()

        for img_data in data["images"]:
            required_fields = ["id", "width", "height"]
            missing = [f for f in required_fields if f not in img_data]
            if missing:
                raise ValueError(f"image missing required fields: {', '.join(missing)}")
            if img_data["id"] in image_ids:
                raise ValueError(f"duplicate image id: {img_data['id']}")
            if img_data["width"] <= 0 or img_data["height"] <= 0:
                raise ValueError(
                    f"invalid image dimensions: {img_data['width']}x{img_data['height']}"
                )
            image_ids.add(img_data["id"])
            image = Image(annotations=[], **img_data)
            images.append(image)
            image_map[image.id] = image

        annotation_ids = set()
        for ann_data in data["annotations"]:
            required_fields = ["id", "image_id", "category_id", "bbox"]
            missing = [f for f in required_fields if f not in ann_data]
            if missing:
                raise ValueError(
                    f"annotation missing required fields: {', '.join(missing)}"
                )
            if ann_data["id"] in annotation_ids:
                raise ValueError(f"duplicate annotation id: {ann_data['id']}")
            annotation_ids.add(ann_data["id"])

            if ann_data["image_id"] not in image_map:
                raise ValueError(
                    f"annotation {ann_data['id']} references non-existent image {ann_data['image_id']}"
                )
            image = image_map[ann_data["image_id"]]

            try:
                category = next(
                    c for c in categories.values() if c.id == ann_data["category_id"]
                )
            except StopIteration:
                raise ValueError(
                    f"annotation {ann_data['id']} references non-existent category {ann_data['category_id']}"
                )

            if len(ann_data["bbox"]) != 4:
                raise ValueError("invalid bbox format")
            if ann_data["bbox"][2] <= 0 or ann_data["bbox"][3] <= 0:
                raise ValueError("invalid bbox dimensions")
            bbox = BBox(*ann_data["bbox"])
            annotation = Annotation(
                id=ann_data["id"], bbox=bbox, category=category, image=image
            )
            image.annotations.append(annotation)

        return cls(categories, images)
