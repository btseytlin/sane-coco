from typing import Any, Dict, List, Tuple
import numpy as np
from sane_coco.models import Annotation


def group_annotations_by_image(
    annotations: List[Annotation],
) -> Dict[int, List[Annotation]]:
    grouped = {}
    for annotation in annotations:
        if annotation.image.id not in grouped:
            grouped[annotation.image.id] = []
        grouped[annotation.image.id].append(annotation)
    return grouped


def calculate_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    n, m = boxes1.shape[0], boxes2.shape[0]
    ious = np.zeros((n, m), dtype=np.float32)
    eps = np.finfo(np.float32).eps

    for i in range(n):
        x1, y1, w1, h1 = boxes1[i]
        x1_end, y1_end = x1 + w1, y1 + h1
        area1 = w1 * h1

        for j in range(m):
            x2, y2, w2, h2 = boxes2[j]
            x2_end, y2_end = x2 + w2, y2 + h2
            area2 = w2 * h2

            xi = max(x1, x2)
            yi = max(y1, y2)
            xi_end = min(x1_end, x2_end)
            yi_end = min(y1_end, y2_end)

            if xi_end > xi and yi_end > yi:
                intersection = (xi_end - xi) * (yi_end - yi)
                union = area1 + area2 - intersection
                if union < eps:
                    ious[i, j] = 1.0 if abs(area1 - area2) < eps else 0.0
                else:
                    ious[i, j] = intersection / union

    return ious


def validate_and_get_annotation_format(annotation: dict[str, Any]):
    coco_keys = ["category", "bbox"]
    torchmetrics_keys = ["labels", "boxes"]
    if all(key in annotation for key in coco_keys):
        return "coco"
    elif all(key in annotation for key in torchmetrics_keys):
        return "torchmetrics"
    else:
        raise ValueError(
            f"""Invalid annotation format: {annotation}. 
            
            Expected either: 
            1. COCO style dict with keys 'category', 'bbox', 'score' or 
            2. Torchmetrics style dict with keys 'labels', 'boxes', 'scores'"""
        )


def validate_annotations_and_get_format(
    annotations: list[dict[str, Any]] | list[list[dict[str, Any]]],
) -> list[str]:
    formats = []
    for img_annotations in annotations:
        if isinstance(img_annotations, dict):
            formats.append(validate_and_get_annotation_format(img_annotations))
        else:
            for ann in img_annotations:
                formats.append(validate_and_get_annotation_format(ann))

    if len(set(formats)) > 1:
        raise ValueError("Mixed annotation formats")

    return formats


def convert_torchmetrics_format_to_coco(
    annotations: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    coco_annotations = []
    for img_annotations in annotations:
        coco_img_annotations = []
        boxes = img_annotations["boxes"]
        labels = img_annotations["labels"]
        scores = img_annotations.get("scores", [None] * len(boxes))
        for box, label, score in zip(boxes, labels, scores):
            annotation = {"category": label, "bbox": box}
            if score is not None:
                annotation["score"] = score
            coco_img_annotations.append(annotation)
        coco_annotations.append(coco_img_annotations)
    return coco_annotations
