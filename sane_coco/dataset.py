from typing import Dict, List
from .models import Annotation, BBox, Category, Image
from .validation import (
    validate_sections_exist,
    validate_images,
    validate_categories,
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
        return Category(**cat_data)

    @classmethod
    def parse_image(cls, img_data: dict) -> Image:
        return Image(annotations=[], **img_data)

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

        return Annotation(
            id=ann_data["id"],
            bbox=BBox(*ann_data["bbox"]),
            category=category,
            image=image,
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
        validate_categories(data["categories"])
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

    def get_annotation_dicts(self):
        result = []
        for img in self.images:
            img_annotations = [
                {"category": ann.category.name, "bbox": ann.bbox.xywh}
                for ann in img.annotations
            ]
            result.append(img_annotations)
        return result
