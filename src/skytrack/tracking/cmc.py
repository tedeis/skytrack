"""Camera motion compensation (CMC) for a moving UAV.

On a drone, most apparent motion in the image is the platform, not the
targets. Without compensation a constant-velocity Kalman filter predicts boxes
where the object *was* in image coordinates, and IoU matching collapses during
yaw or translation. We estimate a frame-to-frame similarity transform from
sparse optical flow on background corners and warp every track's state by it.
"""

from __future__ import annotations

import cv2
import numpy as np


class CameraMotionCompensator:
    def __init__(self, downscale: int = 2, max_corners: int = 1000):
        self.downscale = downscale
        self.max_corners = max_corners
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_pts = None

    def estimate(self, frame: np.ndarray, det_boxes: np.ndarray | None = None) -> np.ndarray:
        """Return the 2x3 affine mapping previous-frame pixels to this frame."""
        identity = np.eye(2, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        s = self.downscale
        if s > 1:
            gray = cv2.resize(gray, (gray.shape[1] // s, gray.shape[0] // s))

        # Mask out detections so corners come from the static background.
        mask = np.full(gray.shape, 255, np.uint8)
        if det_boxes is not None:
            for x0, y0, x1, y1 in (det_boxes / s).astype(int):
                mask[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = 0

        pts = cv2.goodFeaturesToTrack(
            gray, self.max_corners, qualityLevel=0.01, minDistance=8, mask=mask, blockSize=3
        )

        affine = identity
        if self._prev_gray is not None and self._prev_pts is not None and len(self._prev_pts) > 8:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._prev_pts, None)
            good = status.ravel() == 1
            if good.sum() > 8:
                m, _ = cv2.estimateAffinePartial2D(
                    self._prev_pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                if m is not None:
                    affine = m.copy()
                    affine[:, 2] *= s

        self._prev_gray = gray
        self._prev_pts = pts
        return affine
