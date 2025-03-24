from numba import njit
import numpy as np


@njit(cache=True, nopython=True, fastmath=True)
def calculate_iou_batch_numba(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    n, m = boxes1.shape[0], boxes2.shape[0]
    ious = np.zeros((n, m), dtype=np.float32)
    eps = np.finfo(np.float32).eps

    boxes1_coords = np.zeros((n, 4), dtype=np.float32)
    boxes2_coords = np.zeros((m, 4), dtype=np.float32)

    boxes1_coords[:, :2] = boxes1[:, :2]
    boxes1_coords[:, 2:] = boxes1[:, :2] + boxes1[:, 2:]

    boxes2_coords[:, :2] = boxes2[:, :2]
    boxes2_coords[:, 2:] = boxes2[:, :2] + boxes2[:, 2:]

    for i in range(n):
        x1, y1, x1_end, y1_end = boxes1_coords[i]
        area1 = (x1_end - x1) * (y1_end - y1)

        for j in range(m):
            x2, y2, x2_end, y2_end = boxes2_coords[j]

            if not (x1_end >= x2 and x2_end >= x1 and y1_end >= y2 and y2_end >= y1):
                continue

            area2 = (x2_end - x2) * (y2_end - y2)

            xi = max(x1, x2)
            yi = max(y1, y2)
            xi_end = min(x1_end, x2_end)
            yi_end = min(y1_end, y2_end)

            intersection = (xi_end - xi) * (yi_end - yi)
            union = area1 + area2 - intersection

            if union < eps:
                ious[i, j] = 1.0 if abs(area1 - area2) < eps else 0.0
            else:
                ious[i, j] = intersection / union

    return ious
