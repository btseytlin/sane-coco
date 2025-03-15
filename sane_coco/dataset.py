from typing import Dict, List
from .models import Annotation, BBox, Category, Image
from .validation import validate_required_fields, validate_bbox


class COCODataset:
    def __init__(self, categories: Dict[str, Category], images: List[Image]):
        self.categories = categories
        self.images = images
        self.annotations: List[Annotation] = []

    @classmethod
    def find_category_by_id(
        cls, category_id: int, categories: Dict[str, Category], annotation_id: int
    ) -> Category:
        try:
            return next(c for c in categories.values() if c.id == category_id)
        except StopIteration:
            raise ValueError(
                f"annotation {annotation_id} references non-existent category {category_id}"
            )

    @classmethod
    def parse_category(cls, cat_data: dict, category_ids: set[int]) -> Category:
        return Category(**cat_data)

    @classmethod
    def parse_image(cls, img_data: dict, image_ids: set[int]) -> Image:
        return Image(annotations=[], **img_data)

    @classmethod
    def parse_annotation(
        cls,
        ann_data: dict,
        annotation_ids: set[int],
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
        category_ids = set()
        for cat_data in category_data:
            category = cls.parse_category(cat_data, category_ids)
            categories[cat_data["name"]] = category
        return categories

    @classmethod
    def parse_images(
        cls, image_data: List[dict]
    ) -> tuple[List[Image], Dict[int, Image]]:
        images = []
        image_map = {}
        image_ids = set()
        for img_data in image_data:
            image = cls.parse_image(img_data, image_ids)
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
        annotation_ids = set()
        annotations = []
        for ann_data in annotation_data:
            annotation = cls.parse_annotation(
                ann_data, annotation_ids, image_map, categories
            )
            annotations.append(annotation)
        return annotations

    @classmethod
    def group_annotations_by_image(
        cls,
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
        self.annotations = annotations

    @classmethod
    def from_dict(cls, data: dict) -> "COCODataset":
        cls.ensure_sections_exist(data)
        cls.validate_images(data["images"])
        cls.validate_categories(data["categories"])
        cls.validate_annotations(
            data["annotations"], data["images"], data["categories"]
        )

        cls.validate_unique_ids(data["images"], "image")
        cls.validate_unique_ids(data["categories"], "category")

        categories = cls.parse_categories(data["categories"])
        images, image_map = cls.parse_images(data["images"])
        annotations = cls.parse_annotations(data["annotations"], image_map, categories)

        instance = cls(categories, images)
        instance.link_annotations(annotations)
        return instance

    @classmethod
    def ensure_sections_exist(cls, data: dict) -> None:
        for section in ["images", "categories", "annotations"]:
            if section not in data:
                data[section] = []

    @classmethod
    def validate_images(cls, images: List[dict]) -> None:
        for img_data in images:
            validate_required_fields(img_data, ["id", "width", "height"], "image")
            if img_data["width"] <= 0 or img_data["height"] <= 0:
                raise ValueError(
                    f"invalid image dimensions: {img_data['width']}x{img_data['height']}"
                )

    @classmethod
    def validate_categories(cls, categories: List[dict]) -> None:
        for cat_data in categories:
            validate_required_fields(
                cat_data, ["id", "name", "supercategory"], "category"
            )

    @classmethod
    def validate_annotations(
        cls, annotations: List[dict], images: List[dict], categories: List[dict]
    ) -> None:
        image_ids = {img["id"] for img in images}
        category_ids = {cat["id"] for cat in categories}

        for ann_data in annotations:
            validate_required_fields(
                ann_data, ["id", "image_id", "category_id", "bbox"], "annotation"
            )
            validate_bbox(ann_data["bbox"])
            if ann_data["image_id"] not in image_ids:
                raise ValueError(
                    f"annotation {ann_data['id']} references non-existent image {ann_data['image_id']}"
                )
            if ann_data["category_id"] not in category_ids:
                raise ValueError(
                    f"annotation {ann_data['id']} references non-existent category {ann_data['category_id']}"
                )

        cls.validate_unique_ids(annotations, "annotation")

    @classmethod
    def validate_unique_ids(cls, items: List[dict], entity: str) -> None:
        ids = set()
        for item in items:
            if item["id"] in ids:
                raise ValueError(f"duplicate {entity} id: {item['id']}")
            ids.add(item["id"])
