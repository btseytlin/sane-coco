from typing import List, Set


def validate_required_fields(data: dict, required: List[str], entity: str) -> None:
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(
            f"{entity.capitalize()} missing required fields: {', '.join(missing)}"
        )


def validate_bbox(bbox: List[float]) -> None:
    if len(bbox) != 4:
        raise ValueError("Invalid bbox format")
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError("Invalid bbox dimensions")


def validate_sections_exist(data: dict) -> None:
    for section in ["images", "categories", "annotations"]:
        if section not in data:
            raise ValueError(f"Missing section: {section}")


def validate_images(images: List[dict]) -> None:
    for img_data in images:
        validate_required_fields(img_data, ["id", "width", "height"], "image")
        if img_data["width"] <= 0 or img_data["height"] <= 0:
            raise ValueError(
                f"Invalid image dimensions: {img_data['width']}x{img_data['height']}"
            )


def validate_categories(categories: List[dict]) -> None:
    for cat_data in categories:
        validate_required_fields(cat_data, ["id", "name", "supercategory"], "category")


def validate_annotations(
    annotations: List[dict], images: List[dict], categories: List[dict]
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
                f"Annotation {ann_data['id']} references non-existent image {ann_data['image_id']}"
            )
        if ann_data["category_id"] not in category_ids:
            raise ValueError(
                f"Annotation {ann_data['id']} references non-existent category {ann_data['category_id']}"
            )

    validate_unique_ids(annotations, "annotation")


def validate_unique_ids(items: List[dict], entity: str) -> None:
    ids = set()
    for item in items:
        if item["id"] in ids:
            raise ValueError(f"Duplicate {entity} id: {item['id']}")
        ids.add(item["id"])
