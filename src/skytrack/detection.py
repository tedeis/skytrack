"""Detector wrapper with optional sliced (tiled) inference.

VisDrone frames are ~1360x765 to 2000x1500 with a median object side under
20 px. Resizing the whole frame to a 640 px network input shrinks those objects
to a few pixels. Sliced inference runs the network on overlapping native-
resolution crops plus one full-frame pass, then merges with NMS. It trades
latency for small-object recall; `benchmark` quantifies that trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skytrack.tracking.matching import batched_nms


@dataclass
class DetectorConfig:
    weights: str = "yolo11s.pt"
    imgsz: int = 640
    conf: float = 0.05  # keep low scores: the tracker's second association uses them
    iou: float = 0.6
    device: str | None = None  # "cuda:0", "mps", "cpu"; None = auto
    tile: bool = False
    tile_size: int = 640
    tile_overlap: float = 0.2
    merge_iou: float = 0.5
    half: bool = False


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    """Start offsets that cover [0, length) with tiles of size `tile`."""
    if length <= tile:
        return [0]
    stride = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


class Detector:
    def __init__(self, config: DetectorConfig):
        from ultralytics import YOLO

        self.cfg = config
        self.model = YOLO(config.weights, task="detect")
        self.names = self.model.names

    def _predict(self, images: list[np.ndarray], imgsz: int) -> list[np.ndarray]:
        results = self.model.predict(
            images,
            imgsz=imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            half=self.cfg.half,
            verbose=False,
        )
        out = []
        for r in results:
            b = r.boxes
            out.append(
                np.concatenate(
                    [
                        b.xyxy.cpu().numpy(),
                        b.conf.cpu().numpy()[:, None],
                        b.cls.cpu().numpy()[:, None],
                    ],
                    axis=1,
                )
                if len(b)
                else np.zeros((0, 6))
            )
        return out

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return (N, 6) [x1, y1, x2, y2, score, class] in image pixels."""
        if not self.cfg.tile:
            return self._predict([image], self.cfg.imgsz)[0]

        H, W = image.shape[:2]
        ts = self.cfg.tile_size
        crops, offsets = [], []
        for y in tile_origins(H, ts, self.cfg.tile_overlap):
            for x in tile_origins(W, ts, self.cfg.tile_overlap):
                crops.append(image[y : y + ts, x : x + ts])
                offsets.append((x, y))

        parts = [self._predict([image], self.cfg.imgsz)[0]]  # full frame for large objects
        for dets, (x, y) in zip(self._predict(crops, ts), offsets, strict=True):
            dets = dets.copy()
            dets[:, [0, 2]] += x
            dets[:, [1, 3]] += y
            parts.append(dets)

        dets = np.concatenate(parts)
        keep = batched_nms(dets[:, :4], dets[:, 4], dets[:, 5], self.cfg.merge_iou)
        return dets[keep]


def export_onnx(weights: str | Path, imgsz: int = 640, half: bool = False) -> Path:
    from ultralytics import YOLO

    kwargs = {"half": True} if half else {}
    path = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, simplify=True, **kwargs)
    return Path(path)
