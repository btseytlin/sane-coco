from typing import List, Tuple
import numpy as np

try:
    from .numba import calculate_iou_batch_numba as calculate_iou_batch
except ImportError:
    from .util import calculate_iou_batch

from sane_coco.types import COCOBBox


def match_predictions_to_ground_truth(
    bboxes_true: list[COCOBBox],
    bboxes_pred: list[COCOBBox],
    scores_pred: list[float],
    iou_threshold: float,
) -> tuple[list[bool], list[bool], list[float]]:
    """
    Match within-image predictions to ground truth.

    Args:
        bboxes_true: List of ground truth bounding boxes for an image.
        bboxes_pred: List of predicted bounding boxes for an image.
        scores_pred: List of scores for the predicted bounding boxes.
        iou_threshold: IoU threshold for matching.

    Returns:
        tp: List of true positives.
        fp: List of false positives.
        scores: List of scores for the predicted bounding boxes.
    """
    if not bboxes_pred:
        return [], [], []

    tp, fp, scores = (
        np.zeros(len(bboxes_pred)),
        np.ones(len(bboxes_pred)),
        np.array(scores_pred),
    )

    if not bboxes_true:
        return tp.tolist(), fp.tolist(), scores.tolist()

    true_matched = np.zeros(len(bboxes_true), dtype=bool)

    argsort = np.argsort(scores_pred)[::-1]
    bboxes_pred = [bboxes_pred[i] for i in argsort]
    scores = np.array(scores_pred)[argsort]

    true_boxes = np.array(bboxes_true)
    pred_boxes = np.array(bboxes_pred)

    ious = calculate_iou_batch(pred_boxes, true_boxes)
    best_iou_idxs = np.argmax(ious, axis=1)
    max_ious = ious[np.arange(ious.shape[0]), best_iou_idxs]

    for pred_idx in range(len(bboxes_pred)):
        if (
            max_ious[pred_idx] >= iou_threshold
            and not true_matched[best_iou_idxs[pred_idx]]
        ):
            tp[pred_idx] = 1
            fp[pred_idx] = 0
            true_matched[best_iou_idxs[pred_idx]] = True
        else:
            tp[pred_idx] = 0
            fp[pred_idx] = 1

    return tp.tolist(), fp.tolist(), scores.tolist()
