"""Constant-velocity Kalman filter over box center and size.

State x = [cx, cy, w, h, vcx, vcy, vw, vh]. Measurements z = [cx, cy, w, h].

Process and measurement noise scale with box size, so a 12 px pedestrian and
a 200 px bus get comparable relative uncertainty. Tracking width and height
directly (rather than SORT's area + aspect ratio) behaves better for the tiny,
near-square boxes typical of aerial footage.
"""

from __future__ import annotations

import numpy as np

STD_POS = 1.0 / 20
STD_VEL = 1.0 / 160


class KalmanFilter:
    ndim = 4

    def __init__(self, dt: float = 1.0):
        n = self.ndim
        self.F = np.eye(2 * n)
        self.F[:n, n:] = dt * np.eye(n)
        self.H = np.eye(n, 2 * n)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(self.ndim)]
        w, h = z[2], z[3]
        std = [
            2 * STD_POS * w, 2 * STD_POS * h, 2 * STD_POS * w, 2 * STD_POS * h,
            10 * STD_VEL * w, 10 * STD_VEL * h, 10 * STD_VEL * w, 10 * STD_VEL * h,
        ]  # fmt: skip
        return mean, np.diag(np.square(std))

    def _process_noise(self, mean: np.ndarray) -> np.ndarray:
        w, h = mean[2], mean[3]
        std = [
            STD_POS * w, STD_POS * h, STD_POS * w, STD_POS * h,
            STD_VEL * w, STD_VEL * h, STD_VEL * w, STD_VEL * h,
        ]  # fmt: skip
        return np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q = self._process_noise(mean)
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def multi_predict(self, means: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized predict for (N, 8) means and (N, 8, 8) covariances."""
        if len(means) == 0:
            return means, covs
        w, h = means[:, 2], means[:, 3]
        std = np.stack(
            [STD_POS * w, STD_POS * h, STD_POS * w, STD_POS * h,
             STD_VEL * w, STD_VEL * h, STD_VEL * w, STD_VEL * h],
            axis=1,
        )  # fmt: skip
        Q = np.zeros_like(covs)
        idx = np.arange(8)
        Q[:, idx, idx] = std**2
        means = means @ self.F.T
        covs = self.F @ covs @ self.F.T + Q
        return means, covs

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, h = mean[2], mean[3]
        R = np.diag(np.square([STD_POS * w, STD_POS * h, STD_POS * w, STD_POS * h]))
        return self.H @ mean, self.H @ cov @ self.H.T + R

    def update(
        self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        # Solve rather than invert: K = P H^T S^-1.
        K = np.linalg.solve(proj_cov, (cov @ self.H.T).T).T
        mean = mean + K @ (z - proj_mean)
        cov = cov - K @ proj_cov @ K.T
        return mean, cov


def warp_states(
    means: np.ndarray, covs: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 2x3 affine camera-motion transform to track states in place of
    re-detection. Positions get rotation+translation; velocities and sizes get
    the linear part only."""
    if len(means) == 0:
        return means, covs
    R = affine[:, :2]
    t = affine[:, 2]
    R8 = np.kron(np.eye(4), R)
    means = means @ R8.T
    means[:, :2] += t
    covs = R8 @ covs @ R8.T
    return means, covs
