from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np

try:
    from .numba import calculate_iou_batch_numba as calculate_iou_batch
except ImportError:
    from .util import calculate_iou_batch


class DetectionMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.gt_annotations = []
        self.predictions = []

    def update(self, gt_annotations, predictions):
        self.gt_annotations.append(gt_annotations)
        self.predictions.append(predictions)

    def compute(self):
        raise NotImplementedError("Subclasses must implement compute method")

    def forward(self, gt_annotations, predictions):
        self.reset()
        self.update(gt_annotations, predictions)
        return self.compute()

    def __call__(self, gt_annotations, predictions):
        return self.forward(gt_annotations, predictions)


class MeanAveragePrecision(DetectionMetric):
    """
    Compute Mean Average Precision (mAP) and Mean Average Recall (mAR) for object detection.

    This metric follows a similar interface to torchmetrics, with update(), compute(), and forward() methods.

    Args:
        iou_thresholds: List of IoU thresholds for evaluation
        max_dets: Maximum number of detections per image
        area_ranges: Dictionary mapping area range names to (min_area, max_area) tuples

    Example:
        >>> from sane_coco.metrics import MeanAveragePrecision
        >>> # Ground truth annotations per image
        >>> gt_annotations = [
        ...     [  # First image
        ...         {"category": "person", "bbox": [100, 100, 50, 100]},
        ...         {"category": "dog", "bbox": [200, 150, 80, 60]}
        ...     ],
        ...     [  # Second image
        ...         {"category": "person", "bbox": [300, 200, 40, 90]}
        ...     ]
        ... ]
        >>> # Predictions per image
        >>> predictions = [
        ...     [  # First image
        ...         {"category": "person", "bbox": [102, 98, 48, 102], "score": 0.9},
        ...         {"category": "dog", "bbox": [198, 152, 82, 58], "score": 0.8}
        ...     ],
        ...     [  # Second image
        ...         {"category": "person", "bbox": [305, 195, 38, 92], "score": 0.95}
        ...     ]
        ... ]
        >>> # Method 1: Using update and compute
        >>> metric = MeanAveragePrecision()
        >>> metric.update(gt_annotations, predictions)
        >>> results = metric.compute()
        >>> print(f"mAP: {results.ap:.4f}, mAP@0.5: {results.ap50:.4f}")
        >>> # Method 2: Using forward (or __call__)
        >>> metric = MeanAveragePrecision()
        >>> results = metric(gt_annotations, predictions)
        >>> print(f"mAP: {results.ap:.4f}, mAP@0.5: {results.ap50:.4f}")
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        max_dets: int = 100,
        area_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
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
        self.max_dets = max_dets
        self.area_ranges = area_ranges or {
            "all": (0, float("inf")),
            "small": (0, 32**2),
            "medium": (32**2, 96**2),
            "large": (96**2, float("inf")),
        }

    def compute(self) -> Dict[str, float]:
        if (
            self.gt_annotations
            and isinstance(self.gt_annotations[0], list)
            and len(self.gt_annotations) == 1
        ):
            gt_annotations = self.gt_annotations[0]
        else:
            gt_annotations = self.gt_annotations

        if (
            self.predictions
            and isinstance(self.predictions[0], list)
            and len(self.predictions) == 1
        ):
            predictions = self.predictions[0]
        else:
            predictions = self.predictions

        return average_precision(
            gt_annotations,
            predictions,
            iou_thresholds=self.iou_thresholds,
            max_dets=self.max_dets,
            area_ranges=self.area_ranges,
        )


def compute_ap_at_iou(
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    iou_threshold: float,
) -> float:
    if not gt_boxes or not pred_boxes:
        return 0.0

    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    # Sort predictions by score
    pred_boxes = sorted(pred_boxes, key=lambda x: x["score"], reverse=True)

    # Initialize arrays for precision-recall calculation
    tp = [0] * num_pred
    fp = [0] * num_pred
    gt_matched = [False] * num_gt

    # Match predictions to ground truth
    for i, pred in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(gt_boxes):
            if gt["category"] != pred["category"]:
                continue

            iou = calculate_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou and not gt_matched[j]:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1

    # Compute precision and recall
    tp_cumsum = []
    fp_cumsum = []
    running_tp_sum = 0
    running_fp_sum = 0
    for i in range(num_pred):
        running_tp_sum += tp[i]
        running_fp_sum += fp[i]
        tp_cumsum.append(running_tp_sum)
        fp_cumsum.append(running_fp_sum)

    recalls = [tp_sum / max(num_gt, 1) for tp_sum in tp_cumsum]
    precisions = [
        tp_sum / (tp_sum + fp_sum) if tp_sum + fp_sum > 0 else 1.0
        for tp_sum, fp_sum in zip(tp_cumsum, fp_cumsum)
    ]

    # Add sentinel values
    precisions = [1.0] + precisions
    recalls = [0.0] + recalls

    # Compute average precision
    ap = 0.0
    for i in range(len(precisions) - 1):
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]

    return float(ap)


def compute_ar_at_iou(
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    iou_threshold: float,
    max_dets: int = 100,
    area_range: Optional[Tuple[float, float]] = None,
) -> float:
    if not gt_boxes or not pred_boxes:
        return 0.0

    # Filter by area if specified
    if area_range:
        min_area, max_area = area_range
        gt_boxes = [
            gt for gt in gt_boxes if min_area <= gt.get("area", float("inf")) < max_area
        ]

    if not gt_boxes:
        return 0.0

    num_gt = len(gt_boxes)

    # Sort predictions by score and limit to max_dets
    pred_boxes = sorted(pred_boxes, key=lambda x: x["score"], reverse=True)[:max_dets]
    num_pred = len(pred_boxes)

    # Initialize arrays for recall calculation
    tp = [0] * num_pred
    gt_matched = [False] * num_gt

    # Match predictions to ground truth
    for i, pred in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(gt_boxes):
            if gt["category"] != pred["category"]:
                continue

            iou = calculate_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou and not gt_matched[j]:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = True

    # Compute recall
    recall = sum(gt_matched) / max(num_gt, 1)
    return float(recall)


def average_precision(
    gt_annotations: List[List[Dict[str, Any]]],
    predictions: List[List[Dict[str, Any]]],
    iou_thresholds: Optional[List[float]] = None,
    max_dets: int = 100,
    area_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    if area_ranges is None:
        area_ranges = {
            "all": (0, float("inf")),
            "small": (0, 32**2),
            "medium": (32**2, 96**2),
            "large": (96**2, float("inf")),
        }

    all_gt = []
    all_pred = []

    for img_gt, img_pred in zip(gt_annotations, predictions):
        all_gt.extend(img_gt)
        all_pred.extend(img_pred)

    for gt in all_gt:
        bbox = gt["bbox"]
        if isinstance(bbox, tuple):
            x, y, w, h = bbox
        else:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        gt["area"] = float(w * h)

    metrics = {"ap": {}, "ar": {}, "size": {"small": {}, "medium": {}, "large": {}}}
    ap_values = []

    for iou_threshold in iou_thresholds:
        ap = compute_ap_at_iou(all_gt, all_pred, iou_threshold)
        ap_values.append(ap)
        metrics["ap"][float(iou_threshold)] = float(ap)

    metrics["map"] = float(sum(ap_values) / len(ap_values) if ap_values else 0.0)

    ar_values = []
    ar_small_values = []
    ar_medium_values = []
    ar_large_values = []

    for iou_threshold in iou_thresholds:
        ar = compute_ar_at_iou(all_gt, all_pred, iou_threshold, max_dets)
        ar_values.append(ar)
        metrics["ar"][float(iou_threshold)] = float(ar)

        ar_small = compute_ar_at_iou(
            all_gt, all_pred, iou_threshold, max_dets, area_ranges["small"]
        )
        ar_small_values.append(ar_small)
        metrics["size"]["small"][float(iou_threshold)] = float(ar_small)

        ar_medium = compute_ar_at_iou(
            all_gt, all_pred, iou_threshold, max_dets, area_ranges["medium"]
        )
        ar_medium_values.append(ar_medium)
        metrics["size"]["medium"][float(iou_threshold)] = float(ar_medium)

        ar_large = compute_ar_at_iou(
            all_gt, all_pred, iou_threshold, max_dets, area_ranges["large"]
        )
        ar_large_values.append(ar_large)
        metrics["size"]["large"][float(iou_threshold)] = float(ar_large)

    metrics["ar"]["mean"] = float(sum(ar_values) / len(ar_values) if ar_values else 0.0)
    metrics["size"]["small"]["mean"] = float(
        sum(ar_small_values) / len(ar_small_values) if ar_small_values else 0.0
    )
    metrics["size"]["medium"]["mean"] = float(
        sum(ar_medium_values) / len(ar_medium_values) if ar_medium_values else 0.0
    )
    metrics["size"]["large"]["mean"] = float(
        sum(ar_large_values) / len(ar_large_values) if ar_large_values else 0.0
    )

    return metrics


def mean_average_precision(
    gt_annotations: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]],
    predictions: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]],
    iou_thresholds: Optional[List[float]] = None,
    max_dets: int = 100,
    area_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    Functional interface for computing mean average precision.

    This is a convenience function that creates a MeanAveragePrecision instance
    and calls it with the provided arguments.

    Args:
        gt_annotations: Ground truth annotations, either as a list of lists (one per image)
                       or a flat list of annotations
        predictions: Predicted annotations, either as a list of lists (one per image)
                    or a flat list of predictions
        iou_thresholds: IoU thresholds for evaluation
        max_dets: Maximum number of detections per image
        area_ranges: Area ranges for evaluation

    Returns:
        Dictionary containing AP and AR metrics

    Example:
        >>> from sane_coco.metrics import mean_average_precision
        >>> # Ground truth annotations per image
        >>> gt_annotations = [
        ...     [  # First image
        ...         {"category": "person", "bbox": [100, 100, 50, 100]},
        ...         {"category": "dog", "bbox": [200, 150, 80, 60]}
        ...     ],
        ...     [  # Second image
        ...         {"category": "person", "bbox": [300, 200, 40, 90]}
        ...     ]
        ... ]
        >>> # Predictions per image
        >>> predictions = [
        ...     [  # First image
        ...         {"category": "person", "bbox": [102, 98, 48, 102], "score": 0.9},
        ...         {"category": "dog", "bbox": [198, 152, 82, 58], "score": 0.8}
        ...     ],
        ...     [  # Second image
        ...         {"category": "person", "bbox": [305, 195, 38, 92], "score": 0.95}
        ...     ]
        ... ]
        >>> results = mean_average_precision(gt_annotations, predictions)
        >>> print(f"mAP: {results['ap']:.4f}, mAP@0.5: {results['ap_0.5']:.4f}")
    """
    # Handle flat lists by wrapping them
    if gt_annotations and isinstance(gt_annotations[0], dict):
        gt_annotations = [gt_annotations]
    if predictions and isinstance(predictions[0], dict):
        predictions = [predictions]

    metric = MeanAveragePrecision(
        iou_thresholds=iou_thresholds,
        max_dets=max_dets,
        area_ranges=area_ranges,
    )
    return metric(gt_annotations, predictions)
