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
        self._images = images

    @property
    def images(self) -> Iterator[Image]:
        return iter(self._images)

    @classmethod
    def from_dict(cls, data: dict) -> "COCODataset":
        categories = {c["name"]: Category(**c) for c in data["categories"]}
        images = []
        image_map = {}

        for img_data in data["images"]:
            image = Image(annotations=[], **img_data)
            images.append(image)
            image_map[image.id] = image

        for ann_data in data["annotations"]:
            image = image_map[ann_data["image_id"]]
            category = next(
                c for c in categories.values() if c.id == ann_data["category_id"]
            )
            bbox = BBox(*ann_data["bbox"])
            annotation = Annotation(
                id=ann_data["id"], bbox=bbox, category=category, image=image
            )
            image.annotations.append(annotation)

        return cls(categories, images)
