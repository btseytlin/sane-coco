import numpy as np

try:
    from .numba import calculate_iou_batch_numba as calculate_iou_batch
except ImportError:
    from .util import calculate_iou_batch


def match_predictions_to_ground_truth(
    true_image_annotations: list[dict],
    pred_image_annotations: list[dict],
    iou_threshold: float,
) -> tuple[list[int], list[int], list[float]]:
    """In this version we assume that all the annotations are of the same category"""
    if not pred_image_annotations:
        return [], [], []

    # Pre-allocate memory for tp, fp, scores
    tp, fp, scores = (
        np.zeros(len(pred_image_annotations)),
        np.ones(len(pred_image_annotations)),
        np.array([b["score"] for b in pred_image_annotations]),
    )

    if not true_image_annotations:
        return tp.tolist(), fp.tolist(), scores.tolist()

    pred_image_annotations = sorted(
        pred_image_annotations, reverse=True, key=lambda x: x["score"]
    )
    scores = np.array([b["score"] for b in pred_image_annotations])
    true_matched = np.zeros(len(true_image_annotations), dtype=bool)

    true_boxes = np.array([b["bbox"] for b in true_image_annotations])
    pred_boxes = np.array([b["bbox"] for b in pred_image_annotations])

    ious = calculate_iou_batch(pred_boxes, true_boxes)
    best_iou_idxs = np.argmax(ious, axis=1)
    max_ious = ious[np.arange(ious.shape[0]), best_iou_idxs]

    for pred_idx, pred in enumerate(pred_image_annotations):
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
