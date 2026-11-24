"""Command-line entry point: `skytrack <command> --help`."""

from __future__ import annotations

import argparse
import itertools
import json
import time
import urllib.request
import zipfile
from dataclasses import fields
from pathlib import Path

import numpy as np

DET_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{split}.zip"
)

# Tracker ablations: each row turns off one component of the full tracker.
ABLATIONS = {
    "full": {},
    "no_cmc": {"use_cmc": False},
    "no_buffered_iou": {"buffer_scale": 0.0},
    "no_low_score_assoc": {"low_thresh": 0.5},
    "class_agnostic": {"class_aware": False},
    "plain_bytetrack": {"use_cmc": False, "buffer_scale": 0.0, "class_aware": False},
}


def cmd_download(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        zip_path = out / f"VisDrone2019-DET-{split}.zip"
        if not zip_path.exists():
            print(f"downloading {split} ...")
            urllib.request.urlretrieve(DET_URL.format(split=split), zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
        print(f"extracted {split} -> {out}")


def cmd_prepare(a: argparse.Namespace) -> None:
    from skytrack.data.visdrone import convert_split, write_dataset_yaml

    src, out = Path(a.src), Path(a.out)
    for split in ("train", "val"):
        summary = convert_split(
            src / f"VisDrone2019-DET-{split}",
            out / split,
            mask_ignored=not a.no_mask,
            min_size=a.min_size,
        )
        print(f"[{split}] " + json.dumps(summary, indent=2))
    write_dataset_yaml(out, out / "visdrone.yaml")
    print(f"wrote {out / 'visdrone.yaml'}")


def cmd_train(a: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        project=a.project,
        name=a.name,
        # Aerial imagery has no canonical "up": vertical flips are valid.
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        # Small objects: keep scale jitter moderate so they don't vanish.
        scale=0.5,
        patience=30,
        seed=0,
        deterministic=True,
        plots=True,
    )


def _detector(a: argparse.Namespace):
    from skytrack.detection import Detector, DetectorConfig

    return Detector(
        DetectorConfig(weights=a.weights, imgsz=a.imgsz, device=a.device, tile=a.tile, conf=a.conf)
    )


def _sequences(source: Path) -> list[Path]:
    """A video, a folder of frames, or a folder of frame folders."""
    if source.is_file():
        return [source]
    subdirs = sorted(p for p in source.iterdir() if p.is_dir())
    return subdirs or [source]


def _tracker_config(a: argparse.Namespace):
    from skytrack.tracking import TrackerConfig

    overrides = {}
    if getattr(a, "no_cmc", False):
        overrides["use_cmc"] = False
    return TrackerConfig(**overrides)


def cmd_track(a: argparse.Namespace) -> None:
    from skytrack.pipeline import run_sequence
    from skytrack.tracking import ByteTracker

    det = _detector(a)
    tracker = ByteTracker(_tracker_config(a))
    out = Path(a.out)
    for seq in _sequences(Path(a.source)):
        stats = run_sequence(
            seq,
            det,
            tracker,
            mot_out=out / f"{seq.stem}.txt",
            video_out=out / f"{seq.stem}.mp4" if a.video else None,
            names=det.names,
        )
        print(f"{seq.name}: " + json.dumps(stats.summary()))


def _cache_detections(a: argparse.Namespace, seqs: list[Path], cache: Path) -> None:
    from skytrack.pipeline import iter_frames

    det = None
    cache.mkdir(parents=True, exist_ok=True)
    for seq in seqs:
        path = cache / f"{seq.stem}.npz"
        if path.exists():
            continue
        det = det or _detector(a)
        frames = [det(f) for f in iter_frames(seq)]
        np.savez_compressed(path, *frames)
        print(f"cached detections for {seq.name} ({len(frames)} frames)")


def cmd_ablate(a: argparse.Namespace) -> None:
    """Run each tracker variant on identical cached detections so differences
    come from the tracker alone."""
    from skytrack.eval.mot import evaluate
    from skytrack.pipeline import iter_frames
    from skytrack.tracking import ByteTracker, TrackerConfig

    seqs = _sequences(Path(a.source))
    cache = Path(a.out) / "detections"
    _cache_detections(a, seqs, cache)

    results = {}
    for variant, overrides in ABLATIONS.items():
        tracker = ByteTracker(TrackerConfig(**overrides))
        pred_dir = Path(a.out) / variant
        pred_dir.mkdir(parents=True, exist_ok=True)
        track_ms = []
        for seq in seqs:
            dets = np.load(cache / f"{seq.stem}.npz")
            tracker.reset()
            rows = []
            frames = iter_frames(seq) if tracker.cfg.use_cmc else itertools.repeat(None)
            for i, frame in enumerate(frames):
                if i >= len(dets.files):
                    break
                t0 = time.perf_counter()
                tracks = tracker.update(dets[f"arr_{i}"], frame)
                track_ms.append((time.perf_counter() - t0) * 1000)
                for x1, y1, x2, y2, tid, s, c in tracks:
                    rows.append(
                        f"{i + 1},{int(tid)},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},"
                        f"{s:.3f},{int(c) + 1},-1,-1"
                    )
            (pred_dir / f"{seq.stem}.txt").write_text("\n".join(rows) + "\n")
        overall, _ = evaluate(Path(a.gt), pred_dir)
        overall["track_ms_mean"] = float(np.mean(track_ms))
        results[variant] = overall
        print(
            f"{variant:>20}: MOTA {overall['mota']:.3f}  IDF1 {overall['idf1']:.3f}  "
            f"IDsw {int(overall['num_switches'])}  {overall['track_ms_mean']:.2f} ms/frame"
        )
    (Path(a.out) / "ablation.json").write_text(json.dumps(results, indent=2))


def cmd_eval_mot(a: argparse.Namespace) -> None:
    from skytrack.eval.mot import evaluate

    overall, table = evaluate(Path(a.gt), Path(a.pred))
    print(table)
    if a.json:
        Path(a.json).write_text(json.dumps(overall, indent=2))


def cmd_robustness(a: argparse.Namespace) -> None:
    from skytrack.eval import robustness

    out = Path(a.out)
    csv_path = out / "robustness.csv"
    rows = robustness.run(
        a.weights,
        Path(a.val),
        out / "work",
        csv_path,
        names=a.degradations,
        severities=tuple(a.severities),
        n_images=a.n_images,
        imgsz=a.imgsz,
        device=a.device,
    )
    robustness.plot(csv_path, out / "robustness.png")
    for r in rows:
        print(r)


def cmd_benchmark(a: argparse.Namespace) -> None:
    """Latency of detector variants on real frames (warm-up excluded)."""
    import cv2

    from skytrack.detection import Detector, DetectorConfig

    frames = [cv2.imread(str(p)) for p in sorted(Path(a.images).glob("*.jpg"))[: a.n]]
    if not frames:
        raise SystemExit(f"no .jpg files in {a.images}")
    results = []
    for weights in a.weights:
        for tile in (False, True):
            det = Detector(
                DetectorConfig(weights=weights, imgsz=a.imgsz, device=a.device, tile=tile)
            )
            for f in frames[:3]:
                det(f)
            ms = []
            for f in frames:
                t0 = time.perf_counter()
                det(f)
                ms.append((time.perf_counter() - t0) * 1000)
            r = {
                "weights": Path(weights).name,
                "tiled": tile,
                "p50_ms": round(float(np.percentile(ms, 50)), 1),
                "p95_ms": round(float(np.percentile(ms, 95)), 1),
                "fps": round(1000 / float(np.mean(ms)), 1),
            }
            results.append(r)
            print(r)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))


def cmd_export(a: argparse.Namespace) -> None:
    from skytrack.detection import export_onnx

    print(export_onnx(a.weights, a.imgsz, a.half))


def _add_detector_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--tile", action="store_true", help="sliced inference for small objects")


def main(argv: list[str] | None = None) -> None:
    from skytrack.data.degradations import DEGRADATIONS
    from skytrack.tracking import TrackerConfig

    parser = argparse.ArgumentParser(prog="skytrack", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download VisDrone-DET")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("prepare", help="convert VisDrone-DET to YOLO format")
    p.add_argument("--src", default="data/raw")
    p.add_argument("--out", default="data/visdrone")
    p.add_argument("--no-mask", action="store_true", help="don't paint out ignored regions")
    p.add_argument("--min-size", type=int, default=2)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="fine-tune a YOLO detector")
    p.add_argument("--data", default="data/visdrone/visdrone.yaml")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="visdrone")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("track", help="detect + track a video or frame folders")
    _add_detector_args(p)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="runs/track")
    p.add_argument("--video", action="store_true", help="also write annotated mp4")
    p.add_argument("--no-cmc", action="store_true")
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("ablate", help="tracker ablation study on VisDrone-MOT")
    _add_detector_args(p)
    p.add_argument("--source", required=True, help="VisDrone-MOT sequences/ folder")
    p.add_argument("--gt", required=True, help="VisDrone-MOT annotations/ folder")
    p.add_argument("--out", default="runs/ablate")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("eval-mot", help="score tracker output against VisDrone-MOT GT")
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--json")
    p.set_defaults(func=cmd_eval_mot)

    p = sub.add_parser("robustness", help="mAP under sensor/environment degradations")
    p.add_argument("--weights", required=True)
    p.add_argument("--val", default="data/visdrone/val")
    p.add_argument("--out", default="runs/robustness")
    p.add_argument("--degradations", nargs="+", choices=list(DEGRADATIONS))
    p.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--n-images", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_robustness)

    p = sub.add_parser("benchmark", help="detector latency: PyTorch vs ONNX, tiled vs not")
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--images", default="data/visdrone/val/images")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--json")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("export", help="export weights to ONNX")
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    assert {f.name for f in fields(TrackerConfig)} >= {k for v in ABLATIONS.values() for k in v}
    args.func(args)


if __name__ == "__main__":
    main()
