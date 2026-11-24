"""Detector + tracker over a video file or a folder of frames."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from skytrack.tracking import ByteTracker

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def iter_frames(source: Path) -> Iterator[np.ndarray]:
    source = Path(source)
    if source.is_dir():
        for p in sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            frame = cv2.imread(str(p))
            if frame is not None:
                yield frame
        return
    cap = cv2.VideoCapture(str(source))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def draw_tracks(
    frame: np.ndarray, tracks: np.ndarray, names: dict | list | None = None
) -> np.ndarray:
    out = frame.copy()
    for x1, y1, x2, y2, tid, _score, cls in tracks:
        c = _color(int(tid))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        label = f"{int(tid)}"
        if names is not None:
            label += f" {names[int(cls)]}"
        cv2.putText(
            out, label, (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA
        )
    return out


@dataclass
class RunStats:
    frames: int = 0
    detect_ms: list[float] = field(default_factory=list)
    track_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        d, t = np.array(self.detect_ms or [0]), np.array(self.track_ms or [0])
        total = d + t
        return {
            "frames": self.frames,
            "detect_ms_p50": float(np.percentile(d, 50)),
            "track_ms_p50": float(np.percentile(t, 50)),
            "track_ms_p95": float(np.percentile(t, 95)),
            "end_to_end_fps": float(1000 / max(total.mean(), 1e-9)),
        }


def run_sequence(
    source: Path,
    detect: Callable[[np.ndarray], np.ndarray],
    tracker: ByteTracker,
    mot_out: Path | None = None,
    video_out: Path | None = None,
    names: dict | list | None = None,
    fps: float = 25.0,
) -> RunStats:
    """Track one sequence. Writes VisDrone/MOT-format rows
    `frame,id,x,y,w,h,score,category(1-based),-1,-1` to `mot_out`."""
    tracker.reset()
    stats = RunStats()
    writer = None
    rows = []
    for frame_idx, frame in enumerate(iter_frames(source), start=1):
        t0 = time.perf_counter()
        dets = detect(frame)
        t1 = time.perf_counter()
        tracks = tracker.update(dets, frame)
        t2 = time.perf_counter()
        stats.frames += 1
        stats.detect_ms.append((t1 - t0) * 1000)
        stats.track_ms.append((t2 - t1) * 1000)

        for x1, y1, x2, y2, tid, score, cls in tracks:
            rows.append(
                f"{frame_idx},{int(tid)},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},"
                f"{score:.3f},{int(cls) + 1},-1,-1"
            )
        if video_out is not None:
            if writer is None:
                h, w = frame.shape[:2]
                video_out.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(draw_tracks(frame, tracks, names))

    if writer is not None:
        writer.release()
    if mot_out is not None:
        mot_out.parent.mkdir(parents=True, exist_ok=True)
        mot_out.write_text("\n".join(rows) + ("\n" if rows else ""))
    return stats
