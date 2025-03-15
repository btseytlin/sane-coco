from typing import Dict, List
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
