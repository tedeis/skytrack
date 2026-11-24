import cv2
import numpy as np


def textured_background(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Smooth random texture with enough corners for optical flow."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)


def synthetic_scene(
    n_frames: int = 40,
    n_objects: int = 6,
    size: int = 12,
    pan: tuple[float, float] = (0.0, 0.0),
    seed: int = 0,
    H: int = 360,
    W: int = 640,
):
    """Small objects moving linearly in world coordinates, viewed by a camera
    that pans by `pan` px/frame. Yields (frame, gt) with gt rows
    [x1, y1, x2, y2, obj_id]."""
    rng = np.random.default_rng(seed)
    pad = 60
    pan = np.asarray(pan, dtype=float)
    extra = np.ceil(np.abs(pan) * n_frames).astype(int)
    world = textured_background(H + 2 * pad + extra[1], W + 2 * pad + extra[0], seed)
    cam0 = pad + np.where(pan < 0, extra, 0)
    start = cam0 + rng.uniform([40, 40], [W - 40, H - 40], (n_objects, 2))
    vel = rng.uniform(-2.5, 2.5, (n_objects, 2))
    for f in range(n_frames):
        cx0, cy0 = (cam0 + pan * f).astype(int)
        frame = world[cy0 : cy0 + H, cx0 : cx0 + W].copy()
        gt = []
        for i in range(n_objects):
            cx, cy = start[i] + vel[i] * f - (cx0, cy0)
            x1, y1 = int(cx - size / 2), int(cy - size / 2)
            if 0 <= x1 and 0 <= y1 and x1 + size < W and y1 + size < H:
                cv2.rectangle(frame, (x1, y1), (x1 + size, y1 + size), (255, 255, 255), -1)
                gt.append([x1, y1, x1 + size, y1 + size, i])
        yield frame, np.array(gt, dtype=float).reshape(-1, 5)
