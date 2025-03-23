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
        area_ranges: dict[str, tuple[float, float]] | None = None,
    ):
        self.annotations_true = annotations_true or []
        self.annotations_pred = annotations_pred or []

        self.iou_thresholds = iou_thresholds or list(DEFAULT_IOU_THRESHOLDS)

        self.max_detections = max_detections
        self.area_ranges = area_ranges or dict(DEFAULT_AREA_RANGES)

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

    def compute(self) -> dict[str, float]:
        return average_precision(
            self.annotations_true,
            self.annotations_pred,
            iou_thresholds=self.iou_thresholds,
            max_detections=self.max_detections,
            area_ranges=self.area_ranges,
        )


def match_predictions_to_ground_truth(
    img_true: list[dict[str, Any]],
    img_pred: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[int], list[int], list[float]]:
    if not img_pred:
        return [], [], []

    true_boxes = np.array([b["bbox"] for b in img_true])
    pred_boxes = np.array([b["bbox"] for b in img_pred])

    ious = calculate_iou_batch(pred_boxes, true_boxes)

    img_pred = sorted(img_pred, key=lambda x: x["score"], reverse=True)
    true_matched = [False] * len(img_true)
    tp, fp, scores = [], [], []

    true_boxes = np.array([b["bbox"] for b in img_true])
    pred_boxes = np.array([b["bbox"] for b in img_pred])

    if len(true_boxes) > 0 and len(pred_boxes) > 0:
        ious = calculate_iou_batch(pred_boxes, true_boxes)
        for pred_idx, pred in enumerate(img_pred):
            scores.append(pred["score"])
            if len(img_true) > 0:
                valid_gt_indices = [
                    i
                    for i, gt in enumerate(img_true)
                    if gt["category"] == pred["category"]
                ]
                if valid_gt_indices:
                    valid_ious = ious[pred_idx][valid_gt_indices]

                    if len(valid_ious) > 0:
                        best_iou_idx = valid_gt_indices[np.argmax(valid_ious)]
                        max_iou = ious[pred_idx][best_iou_idx]
                        if max_iou >= iou_threshold and not true_matched[best_iou_idx]:
                            tp.append(1)
                            fp.append(0)
                            true_matched[best_iou_idx] = True
                            continue
                tp.append(0)
                fp.append(1)
            else:
                tp.append(0)
                fp.append(1)

    return tp, fp, scores


def compute_precision_recall(tp: np.ndarray, fp: np.ndarray, total_true: int):
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / total_true if total_true > 0 else np.zeros_like(tp_cumsum)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(float).eps)

    precision = np.concatenate(([1], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    return precision, recall


def filter_annotations_by_area(
    annotations_true: list[list[dict[str, Any]]], area_range: list[float] | None
) -> list[list[dict[str, Any]]]:
    if not area_range:
        return annotations_true
    min_area, max_area = area_range
    return [
        [ann for ann in img_true if min_area <= ann["area"] < max_area]
        for img_true in annotations_true
    ]


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
    return ap


def calculate_ar(
    tp: np.ndarray,
    cat_true: list[list[dict]],
    max_detections: int,
) -> float:
    n_imgs = len(cat_true)
    if n_imgs == 0:
        return 0

    recalls = []
    for i in range(n_imgs):
        start_idx = i * max_detections
        end_idx = (i + 1) * max_detections
        if start_idx >= len(tp):
            recalls.append(0)
        else:
            n_true = len(cat_true[i])
            if n_true == 0:
                recalls.append(0)
            else:
                recalls.append(np.sum(tp[start_idx:end_idx]) / n_true)
    return float(np.mean(recalls))


def get_ap_and_ar_for_category(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    category: str,
    iou_threshold: float,
    max_detections: int,
) -> tuple[float, float]:
    annotations_true = filter_by_category(annotations_true, category)
    annotations_pred = filter_by_category(annotations_pred, category)

    total_true = sum(len(img_true) for img_true in annotations_true)
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

    indices = np.argsort(scores)[::-1]
    tp = np.array(tp)[indices]
    ar = calculate_ar(tp, annotations_true, max_detections)

    return ap, ar


def compute_ap_at_iou(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    iou_threshold: float,
    max_detections: int = 100,
    area_range: list[float] | None = None,
) -> tuple[float, float]:
    annotations_true = filter_annotations_by_area(annotations_true, area_range)
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


def average_precision(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    iou_thresholds: list[float] | None = None,
    max_detections: int = 100,
    area_ranges: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = list(DEFAULT_IOU_THRESHOLDS)

    if area_ranges is None:
        area_ranges = dict(DEFAULT_AREA_RANGES)

    annotations_true = precompute_annotation_areas(copy.deepcopy(annotations_true))
    annotations_pred = precompute_annotation_areas(copy.deepcopy(annotations_pred))

    metrics = {
        "ap": {},
        "ar": {},
        "size": {size: {} for size in area_ranges.keys()},
        "map": {},
        "mar": {},
    }

    for iou in iou_thresholds:
        metrics["ap"][iou], metrics["ar"][iou] = compute_ap_at_iou(
            annotations_true, annotations_pred, iou, max_detections, area_ranges["all"]
        )

        for size, area_range in area_ranges.items():
            if size != "all":
                _, metrics["size"][size][iou] = compute_ap_at_iou(
                    annotations_true,
                    annotations_pred,
                    iou,
                    max_detections,
                    area_range,
                )

    metrics["map"] = float(np.mean(list(metrics["ap"].values())))
    metrics["mar"] = float(np.mean(list(metrics["ar"].values())))

    return metrics


def mean_average_precision(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    iou_thresholds: list[float] | None = None,
    max_detections: int | None = None,
    area_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    metric = MeanAveragePrecision(
        iou_thresholds=iou_thresholds,
        max_detections=max_detections,
        area_ranges=area_ranges,
    )
    return metric(annotations_true, annotations_pred)
