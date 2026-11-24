"""Sensor and environment degradations for robustness evaluation.

Each function maps a uint8 BGR image and a severity in 1..5 to a degraded
image of the same shape. Severities are tuned so that level 1 is barely visible
and level 5 is a realistic worst case for a small UAV camera (heavy motion
blur from gimbal jitter, dusk lighting with sensor gain noise, haze, a
bandwidth-starved video link).
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

Degradation = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def _check(severity: int) -> int:
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be in 1..5, got {severity}")
    return severity - 1


def gaussian_noise(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    sigma = [4, 8, 14, 22, 32][_check(severity)]
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    sigma = [0.8, 1.4, 2.0, 3.0, 4.0][_check(severity)]
    return cv2.GaussianBlur(img, (0, 0), sigma)


def motion_blur(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    length = [5, 9, 15, 21, 29][_check(severity)]
    angle = rng.uniform(0, 180)
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def low_light(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Darken with a gamma curve, then add the shot + read noise that a
    sensor's analog gain amplifies in low light."""
    s = _check(severity)
    gain = [0.6, 0.45, 0.32, 0.22, 0.15][s]
    gamma = [1.2, 1.4, 1.6, 1.8, 2.0][s]
    x = img.astype(np.float32) / 255.0
    x = gain * np.power(x, gamma)
    # Shot noise std grows with sqrt(signal); amplified more as gain drops.
    shot = (
        rng.normal(0, 1, x.shape) * np.sqrt(np.maximum(x, 0)) * [0.01, 0.015, 0.02, 0.03, 0.04][s]
    )
    read = rng.normal(0, [0.004, 0.006, 0.009, 0.012, 0.016][s], x.shape)
    return np.clip((x + shot + read) * 255, 0, 255).astype(np.uint8)


def haze(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Atmospheric scattering model I = J*t + A*(1-t) with a vertical depth
    gradient: the top of an oblique aerial frame is farther away."""
    s = _check(severity)
    beta = [0.4, 0.7, 1.0, 1.4, 1.9][s]
    h = img.shape[0]
    depth = np.linspace(1.0, 0.3, h, dtype=np.float32)[:, None, None]
    t = np.exp(-beta * depth)
    airlight = 220.0
    out = img.astype(np.float32) * t + airlight * (1 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    quality = [60, 40, 25, 15, 8][_check(severity)]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def downsample(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate a higher altitude / lower-resolution sensor."""
    factor = [1.5, 2, 2.5, 3, 4][_check(severity)]
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w / factor), int(h / factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


DEGRADATIONS: dict[str, Degradation] = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "low_light": low_light,
    "haze": haze,
    "jpeg": jpeg,
    "downsample": downsample,
}


def apply(name: str, img: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    return DEGRADATIONS[name](img, severity, np.random.default_rng(seed))
