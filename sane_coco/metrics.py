from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np

try:
    from .numba import calculate_iou_batch_numba as calculate_iou_batch
except ImportError:
    from .util import calculate_iou_batch


class MeanAveragePrecision:
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

        self.gt_annotations = []
        self.predictions = []

    def reset(self):
        self.gt_annotations = []
        self.predictions = []

    def update(self, gt_annotations, predictions):
        self.gt_annotations.append(gt_annotations)
        self.predictions.append(predictions)

    def forward(self, gt_annotations, predictions):
        self.reset()
        self.update(gt_annotations, predictions)
        return self.compute()

    def __call__(self, gt_annotations, predictions):
        return self.forward(gt_annotations, predictions)

    def compute(self) -> Dict[str, float]:
        return average_precision(
            self.gt_annotations,
            self.predictions,
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

    gt_boxes = [box for box in gt_boxes if box["category"] != ""]
    pred_boxes = [box for box in pred_boxes if box["category"] != ""]
    if not gt_boxes or not pred_boxes:
        return 0.0

    gt_boxes_array = np.array([box["bbox"] for box in gt_boxes])
    pred_boxes_array = np.array([box["bbox"] for box in pred_boxes])
    scores = np.array([box["score"] for box in pred_boxes])
    gt_cats = [box["category"] for box in gt_boxes]
    pred_cats = [box["category"] for box in pred_boxes]

    sort_idx = np.argsort(-scores)
    pred_boxes_array = pred_boxes_array[sort_idx]
    pred_cats = [pred_cats[i] for i in sort_idx]

    ious = calculate_iou_batch(pred_boxes_array, gt_boxes_array)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    for pred_idx, pred_box in enumerate(pred_boxes_array):
        max_iou = 0
        max_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes_array):
            if not gt_matched[gt_idx] and pred_cats[pred_idx] == gt_cats[gt_idx]:
                iou = ious[pred_idx, gt_idx]
                if iou > max_iou:
                    max_iou = iou
                    max_idx = gt_idx

        if max_iou >= iou_threshold:
            tp[pred_idx] = 1
            gt_matched[max_idx] = True
        else:
            fp[pred_idx] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    return float(
        np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    )


def compute_ar_at_iou(
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    iou_threshold: float,
    max_dets: int = 100,
    area_range: Optional[Tuple[float, float]] = None,
) -> float:
    if not gt_boxes or not pred_boxes:
        return 0.0

    if area_range:
        min_area, max_area = area_range
        gt_boxes = [box for box in gt_boxes if min_area <= box["area"] <= max_area]
        if not gt_boxes:
            return 0.0

    gt_boxes = [box for box in gt_boxes if box["category"] != ""]
    pred_boxes = [box for box in pred_boxes if box["category"] != ""]
    if not gt_boxes or not pred_boxes:
        return 0.0

    gt_boxes_array = np.array([box["bbox"] for box in gt_boxes])
    pred_boxes_array = np.array([box["bbox"] for box in pred_boxes[:max_dets]])
    gt_cats = [box["category"] for box in gt_boxes]
    pred_cats = [box["category"] for box in pred_boxes[:max_dets]]

    ious = calculate_iou_batch(pred_boxes_array, gt_boxes_array)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    for pred_idx, pred_box in enumerate(pred_boxes_array):
        max_iou = 0
        max_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes_array):
            if not gt_matched[gt_idx] and pred_cats[pred_idx] == gt_cats[gt_idx]:
                iou = ious[pred_idx, gt_idx]
                if iou > max_iou:
                    max_iou = iou
                    max_idx = gt_idx

        if max_iou >= iou_threshold and max_idx >= 0:
            gt_matched[max_idx] = True

    return float(np.sum(gt_matched) / len(gt_boxes))


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

    for img_gt_list, img_pred_list in zip(gt_annotations, predictions):
        for img_gt in img_gt_list:
            x, y, w, h = img_gt["bbox"]
            img_gt["area"] = float(w * h)
            all_gt.append(img_gt)
        all_pred.extend(img_pred_list)

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
    metric = MeanAveragePrecision(
        iou_thresholds=iou_thresholds,
        max_dets=max_dets,
        area_ranges=area_ranges,
    )
    return metric(gt_annotations, predictions)
