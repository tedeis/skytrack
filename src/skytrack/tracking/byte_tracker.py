"""BYTE-style multi-object tracker tuned for aerial video.

Implements the two-stage association from ByteTrack (Zhang et al., ECCV 2022):
high-confidence detections are matched first, and the low-confidence leftovers
are then used to keep existing tracks alive through occlusion and blur instead
of being thrown away.

Aerial-specific additions, each switchable for ablation:
  * camera motion compensation (BoT-SORT style) before association
  * buffered IoU (C-BIoU) so small, fast targets still overlap their prediction
  * class-consistent matching so a car track cannot absorb a pedestrian box
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from skytrack.tracking.cmc import CameraMotionCompensator
from skytrack.tracking.kalman import KalmanFilter, warp_states
from skytrack.tracking.matching import (
    buffered_iou_matrix,
    cxcywh_to_xyxy,
    iou_matrix,
    linear_assignment,
    xyxy_to_cxcywh,
)


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5  # detections above this enter the first association
    low_thresh: float = 0.1  # detections below this are discarded outright
    new_track_thresh: float = 0.6  # unmatched high detections above this spawn tracks
    match_thresh: float = 0.8  # max cost (1 - IoU) in the first association
    second_match_thresh: float = 0.5  # max cost in the low-score association
    unconfirmed_match_thresh: float = 0.7
    max_age: int = 30  # frames a lost track is kept before deletion
    buffer_scale: float = 0.3  # C-BIoU expansion; 0 disables
    use_cmc: bool = True
    class_aware: bool = True
    fuse_score: bool = True  # weight IoU by detection confidence


class TrackState(Enum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


class Track:
    def __init__(self, mean, cov, score: float, cls: int, frame: int):
        self.mean, self.cov = mean, cov
        self.score = score
        self.cls = cls
        self.state = TrackState.TENTATIVE
        self.track_id = -1
        self.start_frame = frame
        self.last_frame = frame
        self.hits = 1

    @property
    def xyxy(self) -> np.ndarray:
        return cxcywh_to_xyxy(self.mean[:4])

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, cls={self.cls}, state={self.state.name})"


class ByteTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self.kf = KalmanFilter()
        self.cmc = CameraMotionCompensator() if self.cfg.use_cmc else None
        self.reset()

    def reset(self) -> None:
        self.tracks: list[Track] = []
        self.frame_id = 0
        self._next_id = 1
        if self.cmc is not None:
            self.cmc.reset()

    def update(self, detections: np.ndarray, frame: np.ndarray | None = None) -> np.ndarray:
        """Advance one frame.

        Args:
            detections: (N, 6) array of [x1, y1, x2, y2, score, class].
            frame: BGR image, needed only for camera motion compensation.

        Returns:
            (M, 7) array of [x1, y1, x2, y2, track_id, score, class] for
            confirmed tracks that were matched this frame.
        """
        self.frame_id += 1
        cfg = self.cfg
        dets = np.asarray(detections, dtype=np.float64).reshape(-1, 6)
        dets = dets[dets[:, 4] >= cfg.low_thresh]
        high = dets[dets[:, 4] >= cfg.high_thresh]
        low = dets[dets[:, 4] < cfg.high_thresh]

        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]
        self._predict(frame, dets[:, :4])

        confirmed = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self.tracks if t.state == TrackState.TENTATIVE]

        # 1) Confirmed + lost tracks vs high-score detections.
        cost = self._cost(confirmed, high, cfg.buffer_scale, cfg.fuse_score)
        matches, um_tracks, um_high = linear_assignment(cost, cfg.match_thresh)
        for ti, di in matches:
            self._apply(confirmed[ti], high[di])

        # 2) Tracks still active last frame vs low-score detections. Lost
        #    tracks are excluded: a weak box is not enough evidence to revive one.
        remaining = [confirmed[i] for i in um_tracks if confirmed[i].state == TrackState.CONFIRMED]
        cost = self._cost(remaining, low, 0.0, False)
        matches, um_rem, _ = linear_assignment(cost, cfg.second_match_thresh)
        for ti, di in matches:
            self._apply(remaining[ti], low[di])
        for i in um_rem:
            remaining[i].state = TrackState.LOST
        for i in um_tracks:
            t = confirmed[i]
            if t.state == TrackState.LOST and self.frame_id - t.last_frame > cfg.max_age:
                t.state = TrackState.REMOVED

        # 3) Tentative tracks need a second consecutive high-score match.
        high_left = high[um_high]
        cost = self._cost(tentative, high_left, cfg.buffer_scale, cfg.fuse_score)
        matches, um_tent, um_det = linear_assignment(cost, cfg.unconfirmed_match_thresh)
        for ti, di in matches:
            self._apply(tentative[ti], high_left[di])
        for i in um_tent:
            tentative[i].state = TrackState.REMOVED

        # 4) Spawn new tracks from confident leftovers.
        for d in high_left[um_det]:
            if d[4] < cfg.new_track_thresh:
                continue
            mean, cov = self.kf.initiate(xyxy_to_cxcywh(d[:4]))
            t = Track(mean, cov, float(d[4]), int(d[5]), self.frame_id)
            if self.frame_id == 1:
                self._confirm(t)
            self.tracks.append(t)

        out = [
            np.r_[t.xyxy, t.track_id, t.score, t.cls]
            for t in self.tracks
            if t.state == TrackState.CONFIRMED and t.last_frame == self.frame_id
        ]
        return np.array(out).reshape(-1, 7)

    # ------------------------------------------------------------------ helpers

    def _predict(self, frame: np.ndarray | None, det_boxes: np.ndarray) -> None:
        if not self.tracks:
            if self.cmc is not None and frame is not None:
                self.cmc.estimate(frame, det_boxes)
            return
        means = np.stack([t.mean for t in self.tracks])
        covs = np.stack([t.cov for t in self.tracks])
        # Lost tracks keep coasting but with velocity damped on size so boxes
        # don't balloon or collapse while unobserved.
        for i, t in enumerate(self.tracks):
            if t.state != TrackState.CONFIRMED:
                means[i, 6:8] = 0
        means, covs = self.kf.multi_predict(means, covs)
        if self.cmc is not None and frame is not None:
            affine = self.cmc.estimate(frame, det_boxes)
            means, covs = warp_states(means, covs, affine)
        for t, m, c in zip(self.tracks, means, covs, strict=True):
            m[2:4] = np.maximum(m[2:4], 1.0)
            t.mean, t.cov = m, c

    def _cost(self, tracks: list[Track], dets: np.ndarray, buffer: float, fuse: bool) -> np.ndarray:
        if not tracks or len(dets) == 0:
            return np.ones((len(tracks), len(dets)))
        tboxes = np.stack([t.xyxy for t in tracks])
        iou = (
            buffered_iou_matrix(tboxes, dets[:, :4], buffer)
            if buffer > 0
            else iou_matrix(tboxes, dets[:, :4])
        )
        if fuse:
            iou = iou * dets[None, :, 4]
        cost = 1.0 - iou
        if self.cfg.class_aware:
            tcls = np.array([t.cls for t in tracks])
            cost[tcls[:, None] != dets[None, :, 5].astype(int)] = 1.0
        return cost

    def _apply(self, track: Track, det: np.ndarray) -> None:
        track.mean, track.cov = self.kf.update(track.mean, track.cov, xyxy_to_cxcywh(det[:4]))
        track.score = float(det[4])
        track.last_frame = self.frame_id
        track.hits += 1
        if track.state == TrackState.TENTATIVE:
            self._confirm(track)
        else:
            track.state = TrackState.CONFIRMED

    def _confirm(self, track: Track) -> None:
        track.state = TrackState.CONFIRMED
        track.track_id = self._next_id
        self._next_id += 1
