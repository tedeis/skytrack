"""Box overlap and assignment utilities."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) boxes in xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def buffered_iou_matrix(a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    """IoU after expanding every box by `scale` of its size on each side.

    A 10 px target that moves 12 px between frames has zero IoU with its own
    prediction. Buffering (C-BIoU, Yang et al. 2023) restores overlap for small,
    fast-moving objects without changing the matching for large ones much.
    """
    if scale <= 0:
        return iou_matrix(a, b)
    return iou_matrix(_expand(a, scale), _expand(b, scale))


def _expand(boxes: np.ndarray, scale: float) -> np.ndarray:
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    d = np.stack([-w, -h, w, h], axis=1) * scale
    return boxes + d


def linear_assignment(
    cost: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian matching that rejects pairs with cost > threshold.

    Returns (matches (K, 2), unmatched_rows, unmatched_cols).
    """
    if cost.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost.shape[0]),
            np.arange(cost.shape[1]),
        )
    padded = np.where(cost > threshold, threshold + 1e5, cost)
    rows, cols = linear_sum_assignment(padded)
    keep = cost[rows, cols] <= threshold
    matches = np.stack([rows[keep], cols[keep]], axis=1)
    unmatched_r = np.setdiff1d(np.arange(cost.shape[0]), matches[:, 0])
    unmatched_c = np.setdiff1d(np.arange(cost.shape[1]), matches[:, 1])
    return matches, unmatched_r, unmatched_c


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices sorted by score."""
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=int)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float
) -> np.ndarray:
    """Class-aware NMS via the coordinate-offset trick."""
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    offset = classes[:, None].astype(boxes.dtype) * (boxes.max() + 1)
    return nms(boxes + offset, scores, iou_thr)


def xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    return np.stack(
        [(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
         b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]],
        axis=-1,
    )  # fmt: skip


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    return np.stack(
        [b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2,
         b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2],
        axis=-1,
    )  # fmt: skip
