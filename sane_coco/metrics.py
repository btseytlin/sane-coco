from __future__ import annotations
import copy
from typing import Any
import numpy as np

from sane_coco.types import COCOBBox
from sane_coco.matching import (
    match_predictions_to_ground_truth,
)
from sane_coco.util import (
    convert_torchmetrics_format_to_coco,
    validate_annotations_and_get_format,
)


DEFAULT_IOU_THRESHOLDS = np.linspace(
    0.5,
    0.95,
    int(np.round((0.95 - 0.5) / 0.05)) + 1,
    endpoint=True,
)

DEFAULT_REC_THRESHOLDS = np.linspace(
    0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
)

DEFAULT_AREA_RANGES = {
    "all": (0, float("inf")),
    "small": (0, 32**2),
    "medium": (32**2, 96**2),
    "large": (96**2, float("inf")),
}


class MeanAveragePrecision:
    def __init__(
        self,
        annotations_true: list[list[dict[str, Any]]] | None = None,
        annotations_pred: list[list[dict[str, Any]]] | None = None,
        iou_thresholds: list[float] | None = None,
        max_detections: int = 100,
        min_area: float | None = None,
        max_area: float | None = None,
    ):
        self.annotations_true = annotations_true or []
        self.annotations_pred = annotations_pred or []
        self.iou_thresholds = iou_thresholds or list(DEFAULT_IOU_THRESHOLDS)
        self.max_detections = max_detections
        self.min_area = min_area
        self.max_area = max_area

    def reset(self):
        self.annotations_true = []
        self.annotations_pred = []

    def update(self, annotations_true, annotations_pred):
        formats_true = validate_annotations_and_get_format(annotations_true)
        formats_pred = validate_annotations_and_get_format(annotations_pred)
        if formats_true and formats_pred and formats_true[0] != formats_pred[0]:
            raise ValueError("Mixed annotation formats")

        if formats_true and formats_true[0] == "torchmetrics":
            annotations_true = convert_torchmetrics_format_to_coco(annotations_true)
            annotations_pred = convert_torchmetrics_format_to_coco(annotations_pred)

        self.annotations_true.extend(annotations_true)
        self.annotations_pred.extend(annotations_pred)

    def forward(self, annotations_true, annotations_pred):
        self.reset()
        self.update(annotations_true, annotations_pred)
        return self.compute()

    def __call__(self, annotations_true, annotations_pred):
        return self.forward(annotations_true, annotations_pred)

    def compute(
        self,
    ) -> dict[str, float | dict[str, float]]:
        categories = get_categories(self.annotations_true)
        per_category_metrics = {}

        for category in categories:
            category_annotations_true = filter_by_category(
                self.annotations_true, category
            )
            category_annotations_pred = filter_by_category(
                self.annotations_pred, category
            )

            per_category_metrics[category] = average_precision(
                category_annotations_true,
                category_annotations_pred,
                iou_thresholds=self.iou_thresholds,
                max_detections=self.max_detections,
                min_area=self.min_area,
                max_area=self.max_area,
            )

        metrics = {
            "ap": {},
            "ar": {},
            "map": None,
            "mar": None,
            "per_category": per_category_metrics,
        }

        for iou in self.iou_thresholds:
            ap_values = [m["ap"][iou] for m in per_category_metrics.values()]
            ar_values = [m["ar"][iou] for m in per_category_metrics.values()]
            metrics["ap"][iou] = float(np.mean(ap_values)) if ap_values else 0.0
            metrics["ar"][iou] = float(np.mean(ar_values)) if ar_values else 0.0

        metrics["map"] = float(np.mean(list(metrics["ap"].values())))
        metrics["mar"] = float(np.mean(list(metrics["ar"].values())))

        return metrics


def average_precision(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    iou_thresholds: list[float],
    max_detections: int = 100,
    min_area: float | None = None,
    max_area: float | None = None,
) -> dict[str, float | dict[str, float]]:

    if min_area is not None or max_area is not None:
        min_area = min_area or 0
        max_area = max_area or float("inf")
        assert min_area <= max_area, "min_area must be less than or equal to max_area"
        annotations_true = filter_annotations_by_area(
            annotations_true, min_area, max_area
        )
        annotations_pred = filter_annotations_by_area(
            annotations_pred, min_area, max_area
        )

    metrics = {
        "ap": {},
        "ar": {},
    }

    bboxes_true = [[ann["bbox"] for ann in img_true] for img_true in annotations_true]
    bboxes_pred = [[ann["bbox"] for ann in img_pred] for img_pred in annotations_pred]
    scores_pred = [[ann["score"] for ann in img_pred] for img_pred in annotations_pred]

    for iou in iou_thresholds:
        metrics["ap"][iou], metrics["ar"][iou] = compute_ap_ar_at_iou(
            bboxes_true,
            bboxes_pred,
            scores_pred,
            iou,
            max_detections,
        )

    return metrics


def mean_average_precision(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    iou_thresholds: list[float] | None = None,
    max_detections: int = 100,
    min_area: float | None = None,
    max_area: float | None = None,
) -> dict[str, float | dict[str, float]]:
    metric = MeanAveragePrecision(
        iou_thresholds=iou_thresholds,
        max_detections=max_detections,
        min_area=min_area,
        max_area=max_area,
    )
    return metric(annotations_true, annotations_pred)


def compute_precision_recall(tp: np.ndarray, fp: np.ndarray, total_true: int):
    if len(tp) == 0:
        return np.zeros(101), np.linspace(0, 1, 101)

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = (
        tp_cumsum / float(total_true) if total_true > 0 else np.zeros_like(tp_cumsum)
    )
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(float).eps)

    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    i = np.where(recall[1:] != recall[:-1])[0]

    ap_x = np.linspace(0, 1, 101)
    ap_y = np.zeros_like(ap_x)

    for j in range(len(i)):
        recall_i = recall[i[j]]
        recall_i_plus = recall[i[j] + 1]
        precision_i = precision[i[j] + 1]

        mask = (ap_x >= recall_i) & (ap_x <= recall_i_plus)
        ap_y[mask] = precision_i

    return ap_y, ap_x


def filter_annotations_by_area(
    annotations_true: list[list[dict[str, Any]]],
    min_area: float | None,
    max_area: float | None,
) -> list[list[dict[str, Any]]]:
    filtered = []
    for img_true in annotations_true:
        img_filtered = []
        for ann in img_true:
            area = ann.get("area", None)
            if not area:
                x, y, w, h = ann["bbox"]
                area = w * h
            if min_area <= area < max_area:
                img_filtered.append(ann)
        filtered.append(img_filtered)
    return filtered


def get_categories(annotations_true: list[list[dict[str, Any]]]) -> set[str]:
    return {ann["category"] for img_true in annotations_true for ann in img_true}


def filter_by_category(
    annotations: list[list[dict[str, Any]]], category: str
) -> list[list[dict[str, Any]]]:
    return [
        [ann for ann in img_anns if ann["category"] == category]
        for img_anns in annotations
    ]


def calculate_ap(
    precision: np.ndarray,
    recall: np.ndarray,
    num_points: int = 101,
) -> float:
    ap = 0
    for t in np.linspace(0, 1, num_points):
        mask = recall >= t
        if mask.any():
            ap += np.max(precision[mask]) / num_points
    return min(float(ap), 1.0)


def calculate_ar(
    bboxes_true: list[list[COCOBBox]],
    bboxes_pred: list[list[COCOBBox]],
    scores_pred: list[list[float]],
    iou_threshold: float,
    max_detections: int,
) -> float:
    if not bboxes_true:
        return 0.0

    recall_per_image = []

    for img_idx, (image_bboxes_true, image_bboxes_pred, image_scores_pred) in enumerate(
        zip(bboxes_true, bboxes_pred, scores_pred)
    ):
        if not image_bboxes_true:
            continue

        argsort = np.argsort(image_scores_pred)[::-1][:max_detections]
        image_bboxes_pred = [image_bboxes_pred[i] for i in argsort]
        image_scores_pred = [image_scores_pred[i] for i in argsort]
        tp, _, _ = match_predictions_to_ground_truth(
            image_bboxes_true, image_bboxes_pred, image_scores_pred, iou_threshold
        )

        img_recall = sum(tp) / len(image_bboxes_true)
        recall_per_image.append(img_recall)

    mean_recall = float(np.mean(recall_per_image))
    return min(mean_recall, 1.0)


def compute_ap_ar_at_iou(
    bboxes_true: list[list[COCOBBox]],
    bboxes_pred: list[list[COCOBBox]],
    scores_pred: list[list[float]],
    iou_threshold: float,
    max_detections: int = 100,
) -> tuple[float, float]:
    """
    Compute average precision and average recall at a given IoU threshold for a category.

    Args:
        bboxes_true: List of lists of ground truth bounding boxes for images of a category.
        bboxes_pred: List of lists of predicted bounding boxes for images of a category.
        scores_pred: List of lists of scores for the predicted bounding boxes for images of a category.
        iou_threshold: IoU threshold for matching.
        max_detections: Maximum number of detections to consider.

    Returns:
        ap: Average precision.
        ar: Average recall.
    """

    total_true = sum([len(img_true) for img_true in bboxes_true])
    if total_true == 0:
        return 0.0, 0.0

    tp, fp, scores = [], [], []
    print(scores_pred)
    for image_bboxes_true, image_bboxes_pred, image_scores_pred in zip(
        bboxes_true, bboxes_pred, scores_pred
    ):
        print(image_scores_pred)
        img_tp, img_fp, img_scores = match_predictions_to_ground_truth(
            image_bboxes_true, image_bboxes_pred, image_scores_pred, iou_threshold
        )
        tp.extend(img_tp)
        fp.extend(img_fp)
        scores.extend(img_scores)

    if not scores:
        return 0.0, 0.0

    indices = np.argsort(scores)[::-1][:max_detections]
    tp = np.array(tp)[indices]
    fp = np.array(fp)[indices]
    precision, recall = compute_precision_recall(tp, fp, total_true)
    ap = calculate_ap(precision, recall)
    ar = calculate_ar(
        bboxes_true, bboxes_pred, scores_pred, iou_threshold, max_detections
    )
    return ap, ar
