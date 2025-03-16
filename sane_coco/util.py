from typing import Dict, List, Tuple
import numpy as np
from .models import Annotation


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

    for i in range(n):
        x1, y1, w1, h1 = boxes1[i]
        x1_end, y1_end = x1 + w1, y1 + h1

        for j in range(m):
            x2, y2, w2, h2 = boxes2[j]
            x2_end, y2_end = x2 + w2, y2 + h2

            xi = max(x1, x2)
            yi = max(y1, y2)
            xi_end = min(x1_end, x2_end)
            yi_end = min(y1_end, y2_end)

            if xi_end > xi and yi_end > yi:
                intersection = (xi_end - xi) * (yi_end - yi)
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection
                ious[i, j] = intersection / union

    return ious


def calculate_iou(
    box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]
) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_end, y1_end = x1 + w1, y1 + h1
    x2_end, y2_end = x2 + w2, y2 + h2

    xi = max(x1, x2)
    yi = max(y1, y2)
    xi_end = min(x1_end, x2_end)
    yi_end = min(y1_end, y2_end)

    if xi_end <= xi or yi_end <= yi:
        return 0.0

    intersection = (xi_end - xi) * (yi_end - yi)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union = box1_area + box2_area - intersection

    return intersection / union
