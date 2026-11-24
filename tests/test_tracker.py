from collections import defaultdict

import numpy as np
from conftest import synthetic_scene

from skytrack.tracking import ByteTracker, TrackerConfig
from skytrack.tracking.matching import iou_matrix


def _run(tracker: ByteTracker, scene, det_fn):
    """Return {gt_id: [track_ids over time]} by matching outputs to GT."""
    assigned = defaultdict(list)
    for f, (frame, gt) in enumerate(scene):
        dets = det_fn(f, gt)
        tracks = tracker.update(dets, frame)
        if len(tracks) and len(gt):
            iou = iou_matrix(gt[:, :4], tracks[:, :4])
            for gi, ti in zip(*np.nonzero(iou > 0.3), strict=True):
                assigned[int(gt[gi, 4])].append(int(tracks[ti, 4]))
    return assigned


def _id_switches(assigned) -> int:
    return sum(sum(a != b for a, b in zip(ids, ids[1:])) for ids in assigned.values())


def _perfect(_, gt, score=0.9, rng=None):
    noise = rng.normal(0, 0.7, (len(gt), 4)) if rng is not None else 0
    return np.c_[gt[:, :4] + noise, np.full(len(gt), score), np.zeros(len(gt))]


def test_stable_ids_on_clean_detections():
    rng = np.random.default_rng(1)
    t = ByteTracker(TrackerConfig(use_cmc=False))
    assigned = _run(t, synthetic_scene(), lambda f, gt: _perfect(f, gt, rng=rng))
    assert len(assigned) == 6
    assert _id_switches(assigned) == 0
    assert all(len(set(ids)) == 1 for ids in assigned.values())


def test_low_score_detections_keep_tracks_alive():
    """During frames 15-25 every detection is weak (score 0.3). BYTE should
    keep matching them; a tracker that discards them loses the targets."""

    def det_fn(f, gt):
        return _perfect(f, gt, score=0.3 if 15 <= f < 25 else 0.9)

    with_byte = _run(ByteTracker(TrackerConfig(use_cmc=False)), synthetic_scene(), det_fn)
    hits = sum(len(v) for v in with_byte.values())
    without = _run(
        ByteTracker(TrackerConfig(use_cmc=False, low_thresh=0.5)), synthetic_scene(), det_fn
    )
    assert hits > sum(len(v) for v in without.values())
    assert _id_switches(with_byte) == 0


def test_track_survives_missed_detections():
    def det_fn(f, gt):
        keep = gt if not 10 <= f < 16 else gt[gt[:, 4] != 0]  # object 0 missing 6 frames
        return _perfect(f, keep)

    assigned = _run(ByteTracker(TrackerConfig(use_cmc=False)), synthetic_scene(), det_fn)
    assert len(set(assigned[0])) == 1


def test_camera_motion_compensation_under_fast_pan():
    """A 14 px/frame pan moves 12 px targets farther than their own size
    between frames. Detections are perfect; only CMC can explain the jump."""
    pan = (14.0, 0.0)

    def run(use_cmc):
        t = ByteTracker(TrackerConfig(use_cmc=use_cmc, buffer_scale=0.0))
        return _run(t, synthetic_scene(n_frames=30, pan=pan, seed=3), _perfect)

    with_cmc = run(True)
    without = run(False)
    assert _id_switches(with_cmc) == 0
    assert all(len(set(ids)) == 1 for ids in with_cmc.values())
    # Without CMC every prediction lands a full box-width behind the target,
    # so tracks are lost after the first frame and coverage collapses.
    coverage = lambda a: sum(len(ids) for ids in a.values())  # noqa: E731
    assert coverage(with_cmc) > 3 * coverage(without)


def test_class_aware_matching_blocks_cross_class():
    t = ByteTracker(TrackerConfig(use_cmc=False))
    box = np.array([[100, 100, 120, 120, 0.9, 0]])
    t.update(box)
    out = t.update(box)
    tid = out[0, 4]
    out = t.update(np.array([[100, 100, 120, 120, 0.9, 3]]))  # same place, other class
    assert tid not in out[:, 4]


def test_empty_frames():
    t = ByteTracker()
    assert t.update(np.zeros((0, 6))).shape == (0, 7)
    assert t.update(np.zeros((0, 6)), np.zeros((64, 64, 3), np.uint8)).shape == (0, 7)
