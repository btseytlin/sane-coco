from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

try:
    from .numba import calculate_iou_batch_numba as calculate_iou_batch
except ImportError:
    from .util import calculate_iou_batch

DEFAULT_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

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


def process_image_predictions(
    img_true: list[dict[str, Any]],
    img_pred: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[int], list[int], list[float]]:
    if not img_pred:
        return [], [], []

    img_pred = sorted(img_pred, key=lambda x: x["score"], reverse=True)
    true_matched = [False] * len(img_true)
    tp, fp, scores = [], [], []

    true_boxes = np.array(
        [[b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]] for b in img_true]
    )
    pred_boxes = np.array(
        [[b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]] for b in img_pred]
    )

    if len(true_boxes) > 0 and len(pred_boxes) > 0:
        ious = calculate_iou_batch(pred_boxes, true_boxes)
        for pred_idx, pred in enumerate(img_pred):
            scores.append(pred["score"])
            if len(img_true) > 0:
                max_iou_idx = np.argmax(ious[pred_idx])
                max_iou = ious[pred_idx][max_iou_idx]
                if (
                    max_iou >= iou_threshold
                    and not true_matched[max_iou_idx]
                    and pred["category"] == img_true[max_iou_idx]["category"]
                ):
                    tp.append(1)
                    fp.append(0)
                    true_matched[max_iou_idx] = True
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                tp.append(0)
                fp.append(1)

    return tp, fp, scores


def compute_precision_recall(
    tp: np.ndarray,
    fp: np.ndarray,
    total_true: int,
) -> tuple[np.ndarray, np.ndarray]:
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / total_true

    precision = np.concatenate(([1], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    return precision, recall


def compute_ap_at_iou(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    iou_threshold: float,
    max_detections: int = 100,
) -> float:
    if not annotations_true or not annotations_pred:
        return 0.0

    total_true = sum(len(img_true) for img_true in annotations_true)
    if total_true == 0:
        return 0.0

    tp, fp, scores = [], [], []
    for img_true, img_pred in zip(annotations_true, annotations_pred):
        img_pred = sorted(img_pred, key=lambda x: x["score"], reverse=True)[
            :max_detections
        ]
        img_tp, img_fp, img_scores = process_image_predictions(
            img_true, img_pred, iou_threshold
        )
        tp.extend(img_tp)
        fp.extend(img_fp)
        scores.extend(img_scores)

    if not scores:
        return 0.0

    indices = np.argsort(scores)[::-1]
    tp = np.array(tp)[indices]
    fp = np.array(fp)[indices]

    precision, recall = compute_precision_recall(tp, fp, total_true)
    recall_diffs = recall[1:] - recall[:-1]
    return float(np.sum(precision[1:] * recall_diffs))


def compute_ar_at_iou(
    boxes_true: list[dict[str, Any]],
    boxes_pred: list[dict[str, Any]],
    iou_threshold: float,
    max_detections: int = 100,
    area_range: tuple[float, float] | None = None,
) -> float:
    raise NotImplementedError


def average_precision(
    annotations_true: list[list[dict[str, Any]]],
    annotations_pred: list[list[dict[str, Any]]],
    iou_thresholds: list[float] | None = None,
    max_detections: int = 100,
    area_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = list(DEFAULT_IOU_THRESHOLDS)

    if area_ranges is None:
        area_ranges = dict(DEFAULT_AREA_RANGES)

    metrics = {"ap": {}, "ar": {}, "size": {"small": {}, "medium": {}, "large": {}}}

    # TODO: Implement

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
