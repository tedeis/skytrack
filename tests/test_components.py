import numpy as np
import pytest

from skytrack.data import degradations as D
from skytrack.detection import tile_origins
from skytrack.tracking.kalman import KalmanFilter, warp_states
from skytrack.tracking.matching import (
    batched_nms,
    buffered_iou_matrix,
    cxcywh_to_xyxy,
    iou_matrix,
    linear_assignment,
    nms,
    xyxy_to_cxcywh,
)

# ---------------------------------------------------------------- matching


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_buffered_iou_recovers_overlap_for_small_shift():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[12, 0, 22, 10]], float)
    assert iou_matrix(a, b)[0, 0] == 0
    assert buffered_iou_matrix(a, b, 0.3)[0, 0] > 0


def test_linear_assignment_respects_threshold():
    cost = np.array([[0.1, 0.9], [0.95, 0.99]])
    m, ur, uc = linear_assignment(cost, 0.5)
    assert m.tolist() == [[0, 0]]
    assert ur.tolist() == [1] and uc.tolist() == [1]


def test_nms_and_batched_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    scores = np.array([0.9, 0.8, 0.7])
    assert sorted(nms(boxes, scores, 0.5).tolist()) == [0, 2]
    # Different classes are never suppressed against each other.
    assert sorted(batched_nms(boxes, scores, np.array([0, 1, 0]), 0.5).tolist()) == [0, 1, 2]


def test_box_format_roundtrip():
    b = np.array([[3.0, 4.0, 13.0, 24.0]])
    np.testing.assert_allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(b)), b)


def test_tile_origins_cover_image():
    starts = tile_origins(1500, 640, 0.2)
    assert starts[0] == 0 and starts[-1] + 640 == 1500
    assert all(b - a <= 640 for a, b in zip(starts, starts[1:]))
    assert tile_origins(500, 640, 0.2) == [0]


# ------------------------------------------------------------------ kalman


def test_kalman_learns_constant_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 20.0, 20.0]))
    for t in range(1, 30):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([100 + 3.0 * t, 100 - 2.0 * t, 20, 20]))
    np.testing.assert_allclose(mean[4:6], [3.0, -2.0], atol=0.1)
    pred, _ = kf.predict(mean, cov)
    np.testing.assert_allclose(pred[:2], [100 + 3 * 30, 100 - 2 * 30], atol=0.5)


def test_multi_predict_matches_single():
    kf = KalmanFilter()
    m, c = kf.initiate(np.array([10.0, 20.0, 8.0, 6.0]))
    m[4:] = [1, 2, 0.1, 0]
    m1, c1 = kf.predict(m, c)
    mb, cb = kf.multi_predict(m[None], c[None])
    np.testing.assert_allclose(mb[0], m1)
    np.testing.assert_allclose(cb[0], c1)


def test_warp_states_translation():
    means = np.array([[50.0, 60.0, 10, 10, 1, 0, 0, 0]])
    covs = np.eye(8)[None]
    affine = np.array([[1, 0, 5.0], [0, 1, -3.0]])
    m, c = warp_states(means, covs, affine)
    np.testing.assert_allclose(m[0, :4], [55, 57, 10, 10])
    np.testing.assert_allclose(c[0], np.eye(8))


# ------------------------------------------------------------ degradations


@pytest.mark.parametrize("name", list(D.DEGRADATIONS))
def test_degradations_preserve_shape_and_grow_with_severity(name):
    rng = np.random.default_rng(0)
    img = (rng.random((96, 128, 3)) * 255).astype(np.uint8)
    img[30:60, 40:90] = 255
    errs = []
    for s in (1, 3, 5):
        out = D.apply(name, img, s, seed=0)
        assert out.shape == img.shape and out.dtype == np.uint8
        errs.append(np.mean((out.astype(float) - img) ** 2))
    assert errs[0] < errs[2], f"{name}: severity 5 should differ more than severity 1"


def test_degradation_rejects_bad_severity():
    with pytest.raises(ValueError):
        D.apply("jpeg", np.zeros((8, 8, 3), np.uint8), 6)
