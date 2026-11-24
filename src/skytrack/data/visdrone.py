"""VisDrone annotation parsing and conversion to YOLO format.

VisDrone-DET rows:  x, y, w, h, score, category, truncation, occlusion
VisDrone-MOT rows:  frame, id, x, y, w, h, score, category, truncation, occlusion

Category 0 marks "ignored regions" (crowds, far-away clutter the annotators
skipped) and category 11 is "others". Neither is a trainable class. Naively
dropping the ignored regions teaches the detector that the objects inside them
are background, so `convert_split` can paint them out of the image instead.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Official VisDrone categories 1..10, re-indexed to 0..9.
CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
IGNORED_CATEGORY = 0
OTHERS_CATEGORY = 11

# Classes scored by the official VisDrone MOT toolkit.
MOT_EVAL_CATEGORIES = {1, 4, 5, 6, 9}  # pedestrian, car, van, truck, bus


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    category: int
    truncation: int = 0
    occlusion: int = 0


def parse_det_annotation(path: Path) -> tuple[list[Box], list[Box]]:
    """Return (objects, ignored_regions) from a VisDrone-DET annotation file."""
    objects, ignored = [], []
    for line in Path(path).read_text().splitlines():
        parts = [p for p in line.strip().split(",") if p != ""]
        if len(parts) < 6:
            continue
        x, y, w, h, score, cat = (int(float(v)) for v in parts[:6])
        trunc = int(parts[6]) if len(parts) > 6 else 0
        occ = int(parts[7]) if len(parts) > 7 else 0
        box = Box(x, y, w, h, cat, trunc, occ)
        # score == 0 means "ignore this box" in the DET ground truth.
        if cat == IGNORED_CATEGORY or score == 0:
            ignored.append(box)
        elif cat != OTHERS_CATEGORY:
            objects.append(box)
    return objects, ignored


def mask_regions(image: np.ndarray, regions: list[Box]) -> np.ndarray:
    """Fill ignored regions with the image's mean color."""
    if not regions:
        return image
    out = image.copy()
    fill = image.reshape(-1, image.shape[-1]).mean(axis=0).astype(image.dtype)
    H, W = image.shape[:2]
    for r in regions:
        x0, y0 = max(0, r.x), max(0, r.y)
        x1, y1 = min(W, r.x + r.w), min(H, r.y + r.h)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = fill
    return out


def to_yolo_lines(
    objects: list[Box], img_w: int, img_h: int, min_size: int = 2, max_occlusion: int = 2
) -> list[str]:
    """Clip boxes to the image and emit normalized `cls cx cy w h` lines.

    Boxes smaller than `min_size` px on either side after clipping carry no
    learnable signal and are dropped. `max_occlusion` uses VisDrone's 0/1/2
    scale (none / partial / heavy); the default keeps everything.
    """
    lines = []
    for b in objects:
        if b.occlusion > max_occlusion:
            continue
        x0, y0 = max(0, b.x), max(0, b.y)
        x1, y1 = min(img_w, b.x + b.w), min(img_h, b.y + b.h)
        w, h = x1 - x0, y1 - y0
        if w < min_size or h < min_size:
            continue
        cls = b.category - 1
        cx, cy = (x0 + w / 2) / img_w, (y0 + h / 2) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}")
    return lines


@dataclass
class ConversionStats:
    images: int = 0
    boxes_in: int = 0
    boxes_out: int = 0
    ignored_regions: int = 0
    class_counts: Counter = field(default_factory=Counter)
    box_areas: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        areas = np.sqrt(np.array(self.box_areas)) if self.box_areas else np.zeros(1)
        return {
            "images": self.images,
            "boxes_in": self.boxes_in,
            "boxes_out": self.boxes_out,
            "dropped": self.boxes_in - self.boxes_out,
            "ignored_regions": self.ignored_regions,
            "class_counts": {CLASS_NAMES[k]: v for k, v in sorted(self.class_counts.items())},
            # COCO size buckets use sqrt(area): small < 32, medium < 96.
            "frac_small": float((areas < 32).mean()),
            "frac_medium": float(((areas >= 32) & (areas < 96)).mean()),
            "frac_large": float((areas >= 96).mean()),
            "median_box_side_px": float(np.median(areas)),
        }


def convert_split(
    src: Path,
    dst: Path,
    mask_ignored: bool = True,
    min_size: int = 2,
    max_occlusion: int = 2,
) -> dict:
    """Convert one VisDrone-DET split (`images/`, `annotations/`) to YOLO layout.

    Writes `dst/images/*.jpg`, `dst/labels/*.txt` and `dst/stats.json`.
    """
    src, dst = Path(src), Path(dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    stats = ConversionStats()

    for img_path in sorted((src / "images").glob("*.jpg")):
        ann_path = src / "annotations" / f"{img_path.stem}.txt"
        objects, ignored = parse_det_annotation(ann_path) if ann_path.exists() else ([], [])
        out_img = dst / "images" / img_path.name

        if mask_ignored and ignored:
            image = cv2.imread(str(img_path))
            h, w = image.shape[:2]
            cv2.imwrite(str(out_img), mask_regions(image, ignored), [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            h, w = _image_size(img_path)
            if out_img.resolve() != img_path.resolve():
                shutil.copy2(img_path, out_img)

        lines = to_yolo_lines(objects, w, h, min_size, max_occlusion)
        (dst / "labels" / f"{img_path.stem}.txt").write_text("\n".join(lines))

        stats.images += 1
        stats.boxes_in += len(objects)
        stats.boxes_out += len(lines)
        stats.ignored_regions += len(ignored)
        for ln in lines:
            cls, _, _, bw, bh = ln.split()
            stats.class_counts[int(cls)] += 1
            stats.box_areas.append(float(bw) * w * float(bh) * h)

    summary = stats.summary()
    (dst / "stats.json").write_text(json.dumps(summary, indent=2))
    return summary


def _image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img.shape[1], img.shape[0]


def write_dataset_yaml(root: Path, out: Path) -> None:
    """Write an Ultralytics dataset YAML pointing at converted splits."""
    lines = [f"path: {Path(root).resolve()}", "train: train/images", "val: val/images", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    Path(out).write_text("\n".join(lines) + "\n")


def load_mot_sequence(ann_path: Path) -> np.ndarray:
    """Load a VisDrone-MOT annotation file as an (N, 10) int array."""
    data = np.loadtxt(ann_path, delimiter=",", dtype=np.int64, ndmin=2)
    return data[:, :10]
