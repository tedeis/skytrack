"""Detection robustness sweep: mAP under each degradation and severity."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import cv2

from skytrack.data import degradations as D
from skytrack.data.visdrone import CLASS_NAMES


def make_degraded_split(
    src: Path, dst: Path, name: str | None, severity: int, images: list[Path], seed: int = 0
) -> Path:
    """Write degraded copies of `images` (and their labels) under `dst`.

    `name=None` writes clean copies, giving a baseline on the same subset.
    """
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if name is not None:
            img = D.apply(name, img, severity, seed=seed + i)
        cv2.imwrite(str(dst / "images" / img_path.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        label = src / "labels" / f"{img_path.stem}.txt"
        if label.exists():
            shutil.copy2(label, dst / "labels" / label.name)
    return dst


def _write_yaml(split_dir: Path, path: Path) -> None:
    lines = [f"path: {split_dir.resolve()}", "train: images", "val: images", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    path.write_text("\n".join(lines) + "\n")


def run(
    weights: str,
    val_split: Path,
    work_dir: Path,
    out_csv: Path,
    names: list[str] | None = None,
    severities: tuple[int, ...] = (1, 3, 5),
    n_images: int = 200,
    imgsz: int = 640,
    device: str | None = None,
    seed: int = 0,
) -> list[dict]:
    """Evaluate `weights` on a fixed random subset of the val split under each
    degradation. Results are appended to `out_csv` as they finish."""
    from ultralytics import YOLO

    model = YOLO(weights)
    names = names or list(D.DEGRADATIONS)
    all_images = sorted((Path(val_split) / "images").glob("*.jpg"))
    images = random.Random(seed).sample(all_images, min(n_images, len(all_images)))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["degradation", "severity", "map50", "map50_95", "precision", "recall"]
    rows = []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, sev in [(None, 0)] + [(n, s) for n in names for s in severities]:
            tag = f"{name}_s{sev}" if name else "clean"
            split = make_degraded_split(Path(val_split), work_dir / tag, name, sev, images, seed)
            yaml_path = work_dir / f"{tag}.yaml"
            _write_yaml(split, yaml_path)
            m = model.val(
                data=str(yaml_path), imgsz=imgsz, device=device, plots=False, verbose=False
            ).box
            row = {
                "degradation": name or "clean",
                "severity": sev,
                "map50": round(float(m.map50), 4),
                "map50_95": round(float(m.map), 4),
                "precision": round(float(m.mp), 4),
                "recall": round(float(m.mr), 4),
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)
            shutil.rmtree(split)
    return rows


def plot(csv_path: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    with Path(csv_path).open() as f:
        rows = list(csv.DictReader(f))
    clean = next(float(r["map50"]) for r in rows if r["degradation"] == "clean")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name in dict.fromkeys(r["degradation"] for r in rows if r["degradation"] != "clean"):
        pts = sorted(
            (int(r["severity"]), float(r["map50"])) for r in rows if r["degradation"] == name
        )
        ax.plot([0] + [p[0] for p in pts], [clean] + [p[1] for p in pts], marker="o", label=name)
    ax.set_xlabel("Severity (0 = clean)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Detection robustness on VisDrone-DET val")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
