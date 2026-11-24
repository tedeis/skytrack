"""Multi-object tracking evaluation on VisDrone-MOT.

Reports CLEAR-MOT (MOTA, MOTP) and identity metrics (IDF1, ID switches).
MOTA is dominated by detection quality; IDF1 isolates how well identities are
kept over time, which is what the tracker itself controls. Both are reported
so detector and tracker changes can be judged separately.

Follows the VisDrone protocol where it matters for the numbers:
  * only pedestrian, car, van, truck and bus are scored
  * ground truth flagged score=0 and category-0 "ignored regions" are not
    scored, and predictions that fall mostly inside an ignored region are
    dropped rather than counted as false positives
Matching is class-agnostic at IoU >= 0.5 (the official toolkit additionally
reports per-class AP, which this module does not).
"""

from __future__ import annotations

from pathlib import Path

import motmetrics as mm
import numpy as np

from skytrack.data.visdrone import MOT_EVAL_CATEGORIES, load_mot_sequence
from skytrack.tracking.matching import iou_matrix

METRICS = [
    "mota",
    "idf1",
    "motp",
    "precision",
    "recall",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "mostly_tracked",
    "mostly_lost",
    "num_unique_objects",
]


def _xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = b.astype(np.float64).copy()
    out[:, 2] += out[:, 0]
    out[:, 3] += out[:, 1]
    return out


def _ioa(a: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Fraction of each box in `a` covered by its best-overlapping region."""
    if len(a) == 0 or len(regions) == 0:
        return np.zeros(len(a))
    iw = np.clip(
        np.minimum(a[:, None, 2], regions[None, :, 2])
        - np.maximum(a[:, None, 0], regions[None, :, 0]),
        0,
        None,
    )
    ih = np.clip(
        np.minimum(a[:, None, 3], regions[None, :, 3])
        - np.maximum(a[:, None, 1], regions[None, :, 1]),
        0,
        None,
    )
    area = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-9)
    return ((iw * ih) / area[:, None]).max(axis=1)


def accumulate_sequence(
    gt: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5
) -> mm.MOTAccumulator:
    """Build a motmetrics accumulator from VisDrone-format GT and predictions.

    Both arrays use rows `frame, id, x, y, w, h, score, category, ...` with
    1-based VisDrone category ids.
    """
    acc = mm.MOTAccumulator(auto_id=False)
    frames = np.union1d(gt[:, 0], pred[:, 0]) if len(pred) else np.unique(gt[:, 0])
    for f in frames.astype(int):
        g = gt[gt[:, 0] == f]
        p = pred[pred[:, 0] == f] if len(pred) else np.zeros((0, 10))

        ignore_mask = (g[:, 7] == 0) | (g[:, 6] == 0) | ~np.isin(g[:, 7], list(MOT_EVAL_CATEGORIES))
        scored = g[~ignore_mask]
        regions = _xywh_to_xyxy(g[g[:, 7] == 0][:, 2:6])

        p = p[np.isin(p[:, 7], list(MOT_EVAL_CATEGORIES))]
        p_xyxy = _xywh_to_xyxy(p[:, 2:6]) if len(p) else np.zeros((0, 4))
        keep = _ioa(p_xyxy, regions) < 0.5
        p, p_xyxy = p[keep], p_xyxy[keep]

        g_xyxy = _xywh_to_xyxy(scored[:, 2:6]) if len(scored) else np.zeros((0, 4))
        # Unscored GT (other classes, flagged boxes) also absorbs predictions:
        # a correct "van" box on an object we don't score is not an error.
        other = g[ignore_mask & (g[:, 7] != 0)]
        if len(other) and len(p):
            matched_other = iou_matrix(p_xyxy, _xywh_to_xyxy(other[:, 2:6])).max(axis=1) >= iou_thr
            p, p_xyxy = p[~matched_other], p_xyxy[~matched_other]

        dist = 1.0 - iou_matrix(g_xyxy, p_xyxy)
        dist[dist > 1.0 - iou_thr] = np.nan
        acc.update(scored[:, 1].astype(int).tolist(), p[:, 1].astype(int).tolist(), dist, frameid=f)
    return acc


def evaluate(gt_dir: Path, pred_dir: Path, iou_thr: float = 0.5) -> tuple[dict, str]:
    """Evaluate every `pred_dir/<seq>.txt` against `gt_dir/<seq>.txt`.

    Returns (overall metrics dict, printable per-sequence table).
    """
    accs, names = [], []
    for gt_path in sorted(Path(gt_dir).glob("*.txt")):
        pred_path = Path(pred_dir) / gt_path.name
        if not pred_path.exists():
            continue
        gt = load_mot_sequence(gt_path)
        pred = (
            np.loadtxt(pred_path, delimiter=",", ndmin=2)
            if pred_path.stat().st_size
            else np.zeros((0, 10))
        )
        accs.append(accumulate_sequence(gt, pred, iou_thr))
        names.append(gt_path.stem)
    if not accs:
        raise FileNotFoundError(f"no matching sequences between {gt_dir} and {pred_dir}")

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=METRICS, generate_overall=True)
    table = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    overall = {k: float(v) for k, v in summary.loc["OVERALL"].items()}
    return overall, table
