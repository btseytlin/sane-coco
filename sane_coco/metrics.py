from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

try:
    from .numba import calculate_iou_batch_numba as calculate_iou_batch
except ImportError:
    from .util import calculate_iou_batch

DEFAULT_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

DEFAULT_AREA_RANGES = {
    "all": [0, float("inf")],
    "small": [0, 32**2],
    "medium": [32**2, 96**2],
    "large": [96**2, float("inf")],
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
    ) -> dict[str, float]:
        return average_precision(
            self.annotations_true,
            self.annotations_pred,
            iou_thresholds=self.iou_thresholds,
            max_detections=self.max_detections,
            min_area=self.min_area,
            max_area=self.max_area,
        )


def average_precision(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    iou_thresholds: list[float],
    max_detections: int = 100,
    min_area: float | None = None,
    max_area: float | None = None,
) -> dict[str, float]:
    annotations_true = precompute_annotation_areas(copy.deepcopy(annotations_true))
    annotations_pred = precompute_annotation_areas(copy.deepcopy(annotations_pred))

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
        "map": {},
        "mar": {},
    }

    for iou in iou_thresholds:
        metrics["ap"][iou], metrics["ar"][iou] = compute_ap_ar_at_iou(
            annotations_true,
            annotations_pred,
            iou,
            max_detections,
        )

    metrics["map"] = float(np.mean(list(metrics["ap"].values())))
    metrics["mar"] = float(np.mean(list(metrics["ar"].values())))

    return metrics


def mean_average_precision(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    iou_thresholds: list[float] | None = None,
    max_detections: int | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
) -> dict[str, float]:
    metric = MeanAveragePrecision(
        iou_thresholds=iou_thresholds,
        max_detections=max_detections,
        min_area=min_area,
        max_area=max_area,
    )
    return metric(annotations_true, annotations_pred)


def match_predictions_to_ground_truth(
    true_image_annotations: list[dict[str, Any]],
    pred_image_annotations: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[int], list[int], list[float]]:
    if not pred_image_annotations:
        return [], [], []

    pred_image_annotations = sorted(
        pred_image_annotations, reverse=True, key=lambda x: x["score"]
    )
    true_matched = [False] * len(true_image_annotations)
    tp, fp, scores = [], [], []

    true_boxes = np.array([b["bbox"] for b in true_image_annotations])
    pred_boxes = np.array([b["bbox"] for b in pred_image_annotations])

    if len(pred_boxes) > 0:
        if len(true_boxes) == 0:
            tp = [0] * len(pred_image_annotations)
            fp = [1] * len(pred_image_annotations)
            scores = [b["score"] for b in pred_image_annotations]
            return tp, fp, scores

        ious = calculate_iou_batch(pred_boxes, true_boxes)
        for pred_idx, pred in enumerate(pred_image_annotations):
            scores.append(pred["score"])

            valid_gt_indices = [
                i
                for i, gt in enumerate(true_image_annotations)
                if gt["category"] == pred["category"]
            ]

            if len(true_boxes) == 0 or not valid_gt_indices:
                tp.append(0)
                fp.append(1)
                continue

            valid_ious = ious[pred_idx][valid_gt_indices]

            best_iou_idx = valid_gt_indices[np.argmax(valid_ious)]
            max_iou = ious[pred_idx][best_iou_idx]
            if max_iou >= iou_threshold and not true_matched[best_iou_idx]:
                tp.append(1)
                fp.append(0)
                true_matched[best_iou_idx] = True
            else:
                tp.append(0)
                fp.append(1)

    return tp, fp, scores


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
            area = ann["area"]
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


def calculate_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    ap = 0
    for t in np.linspace(0, 1, 101):
        mask = recall >= t
        if mask.any():
            ap += np.max(precision[mask]) / 101
    return min(float(ap), 1.0)


def calculate_ar(
    tp: np.ndarray,
    cat_true: list[list[dict]],
    max_detections: int,
) -> float:
    n_imgs = len(cat_true)
    if n_imgs == 0:
        return 0.0

    total_gt = sum(len(img_true) for img_true in cat_true)
    if total_gt == 0:
        return 0.0

    tp_cumsum = np.cumsum(tp)
    if len(tp_cumsum) == 0:
        return 0.0

    recall = tp_cumsum[-1] / total_gt if total_gt > 0 else 0.0
    return min(float(recall), 1.0)


def get_ap_and_ar_for_category(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    category: str,
    iou_threshold: float,
    max_detections: int,
) -> tuple[float, float]:
    annotations_true = filter_by_category(annotations_true, category)
    annotations_pred = filter_by_category(annotations_pred, category)

    total_true = sum([len(img_true) for img_true in annotations_true])
    if total_true == 0:
        return 0.0, 0.0

    tp, fp, scores = [], [], []
    for img_true, img_pred in zip(annotations_true, annotations_pred):
        img_tp, img_fp, img_scores = match_predictions_to_ground_truth(
            img_true, img_pred, iou_threshold
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
    ar = calculate_ar(tp, annotations_true, max_detections)

    return ap, ar


def compute_ap_ar_at_iou(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    iou_threshold: float,
    max_detections: int = 100,
) -> tuple[float, float]:
    categories = get_categories(annotations_true)

    ap_per_category = []
    ar_per_category = []

    for category in categories:
        ap, ar = get_ap_and_ar_for_category(
            annotations_true,
            annotations_pred,
            category,
            iou_threshold,
            max_detections,
        )
        ap_per_category.append(ap)
        ar_per_category.append(ar)

    ap = float(np.mean(ap_per_category)) if ap_per_category else 0.0
    ar = float(np.mean(ar_per_category)) if ar_per_category else 0.0

    return ap, ar


def precompute_annotation_areas(annotations: list[list[dict[str, Any]]]) -> list[float]:
    for img_annotations in annotations:
        for annotation in img_annotations:
            if "area" in annotation:
                continue

            if "bbox" in annotation:
                x, y, w, h = annotation["bbox"]
                annotation["area"] = w * h
            else:
                raise ValueError("Annotation must have either 'bbox' or 'area'")

    return annotations
