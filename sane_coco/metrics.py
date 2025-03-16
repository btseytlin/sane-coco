from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

try:
    from .numba import calculate_iou_batch_numba as calculate_iou_batch
except ImportError:
    from .util import calculate_iou_batch


class MeanAveragePrecision:
    def __init__(
        self,
        iou_thresholds: list[float] | None = None,
        max_detections: int = 100,
        area_ranges: dict[str, tuple[float, float]] | None = None,
        annotations_true: list[list[dict[str, Any]]] | None = None,
        annotations_pred: list[list[dict[str, Any]]] | None = None,
    ):
        super().__init__()
        self.iou_thresholds = iou_thresholds or [
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ]
        self.max_dets = max_detections
        self.area_ranges = area_ranges or {
            "all": (0, float("inf")),
            "small": (0, 32**2),
            "medium": (32**2, 96**2),
            "large": (96**2, float("inf")),
        }

        self.annotations_true = annotations_true or []
        self.annotations_pred = annotations_pred or []

    def reset(self):
        self.annotations_true = []
        self.annotations_pred = []

    def update(self, annotations_true, annotations_pred):
        self.annotations_true.append(annotations_true)
        self.annotations_pred.append(annotations_pred)

    def forward(self, gt_annotations, predictions):
        self.reset()
        self.update(gt_annotations, predictions)
        return self.compute()

    def __call__(self, gt_annotations, predictions):
        return self.forward(gt_annotations, predictions)

    def compute(self) -> dict[str, float]:
        return average_precision(
            self.gt_annotations,
            self.predictions,
            iou_thresholds=self.iou_thresholds,
            max_dets=self.max_dets,
            area_ranges=self.area_ranges,
        )


def compute_ap_at_iou(
    gt_boxes: list[dict[str, Any]],
    pred_boxes: list[dict[str, Any]],
    iou_threshold: float,
) -> float:
    raise NotImplementedError


def compute_ar_at_iou(
    gt_boxes: list[dict[str, Any]],
    pred_boxes: list[dict[str, Any]],
    iou_threshold: float,
    max_dets: int = 100,
    area_range: tuple[float, float] | None = None,
) -> float:
    raise NotImplementedError


def average_precision(
    gt_annotations: list[list[dict[str, Any]]],
    predictions: list[list[dict[str, Any]]],
    iou_thresholds: list[float] | None = None,
    max_dets: int = 100,
    area_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    if area_ranges is None:
        area_ranges = {
            "all": (0, float("inf")),
            "small": (0, 32**2),
            "medium": (32**2, 96**2),
            "large": (96**2, float("inf")),
        }

    metrics = {"ap": {}, "ar": {}, "size": {"small": {}, "medium": {}, "large": {}}}

    # TODO: Implement

    return metrics


def mean_average_precision(
    gt_annotations: list[list[dict[str, Any]]],
    predictions: list[list[dict[str, Any]]],
    iou_thresholds: list[float] | None = None,
    max_dets: int = 100,
    area_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    metric = MeanAveragePrecision(
        iou_thresholds=iou_thresholds,
        max_dets=max_dets,
        area_ranges=area_ranges,
    )
    return metric(gt_annotations, predictions)
