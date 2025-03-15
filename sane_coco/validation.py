from typing import List, Set, Dict, Any, Union


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


def validate_segmentation(segmentation: Union[List, Dict], ann_id: int) -> None:
    if isinstance(segmentation, list):
        # Polygon format
        if len(segmentation) == 0:
            raise ValueError(f"Empty segmentation in annotation {ann_id}")

        if isinstance(segmentation[0], list):
            # Multiple polygons
            for poly in segmentation:
                if len(poly) < 6:  # At least 3 points (x,y pairs)
                    raise ValueError(f"Invalid polygon points in annotation {ann_id}")
        elif len(segmentation) < 6:  # At least 3 points (x,y pairs)
            raise ValueError(f"Invalid polygon points in annotation {ann_id}")
    elif isinstance(segmentation, dict):
        # RLE format
        if "counts" not in segmentation or "size" not in segmentation:
            raise ValueError(f"Invalid RLE format in annotation {ann_id}")
        if not isinstance(segmentation["size"], list) or len(segmentation["size"]) != 2:
            raise ValueError(f"Invalid RLE size in annotation {ann_id}")
    else:
        raise ValueError(f"Invalid segmentation format in annotation {ann_id}")


def validate_crowd_annotation(ann_data: Dict[str, Any]) -> None:
    if ann_data.get("iscrowd", 0) == 1:
        # Crowd annotations should use RLE segmentation
        if "segmentation" not in ann_data:
            raise ValueError(f"Crowd annotation {ann_data['id']} missing segmentation")

        if not isinstance(ann_data["segmentation"], dict):
            raise ValueError(f"Crowd annotations must use RLE segmentation")

        # Validate area matches RLE area
        if "area" in ann_data and "segmentation" in ann_data:
            # Estimate RLE area from counts
            seg = ann_data["segmentation"]
            if "counts" in seg:
                counts = seg["counts"]
                if isinstance(counts, list):
                    # Simple RLE format
                    rle_area = 0
                    if len(counts) % 2 == 1:
                        counts = counts[1:]  # Skip first 0 if odd length
                    for i in range(1, len(counts), 2):
                        rle_area += counts[i]

                    # Check if area matches RLE area
                    if (
                        abs(ann_data["area"] - rle_area) > 1
                    ):  # Allow small rounding errors
                        raise ValueError(f"Crowd annotation area must match RLE area")
                # For compressed RLE, we can't easily compute area here


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

        # Validate segmentation if present
        if "segmentation" in ann_data:
            validate_segmentation(ann_data["segmentation"], ann_data["id"])

        # Validate crowd annotations
        if "iscrowd" in ann_data:
            validate_crowd_annotation(ann_data)

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
