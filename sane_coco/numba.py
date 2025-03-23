from numba import njit
import numpy as np


@njit
def calculate_iou_batch_numba(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    n, m = boxes1.shape[0], boxes2.shape[0]
    ious = np.zeros((n, m), dtype=np.float32)
    eps = np.finfo(np.float32).eps

    for i in range(n):
        x1, y1, w1, h1 = boxes1[i].astype(np.float32)
        x1_end, y1_end = x1 + w1, y1 + h1
        area1 = w1 * h1

        for j in range(m):
            x2, y2, w2, h2 = boxes2[j].astype(np.float32)
            x2_end, y2_end = x2 + w2, y2 + h2
            area2 = w2 * h2

            xi = max(x1, x2)
            yi = max(y1, y2)
            xi_end = min(x1_end, x2_end)
            yi_end = min(y1_end, y2_end)

            if xi_end > xi and yi_end > yi:
                intersection = (xi_end - xi) * (yi_end - yi)
                union = area1 + area2 - intersection
                if union < eps:
                    ious[i, j] = 1.0 if abs(area1 - area2) < eps else 0.0
                else:
                    ious[i, j] = intersection / union

    return ious
