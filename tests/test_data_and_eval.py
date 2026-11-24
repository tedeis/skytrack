import json

import cv2
import numpy as np

from skytrack.data.visdrone import convert_split, parse_det_annotation, to_yolo_lines
from skytrack.eval.mot import accumulate_sequence, evaluate

ANNOTATION = """\
10,20,30,40,1,4,0,0
-5,0,20,10,1,1,0,1
200,100,50,50,0,0,0,0
5,5,1,1,1,4,0,0
60,60,20,20,1,11,0,0
70,70,10,10,0,4,0,0
"""


def test_parse_separates_ignored_and_others(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text(ANNOTATION)
    objects, ignored = parse_det_annotation(p)
    # "others" (11) dropped entirely; score==0 rows become ignored regions.
    assert [o.category for o in objects] == [4, 1, 4]
    assert len(ignored) == 2


def test_yolo_lines_clip_and_filter(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text(ANNOTATION)
    objects, _ = parse_det_annotation(p)
    lines = to_yolo_lines(objects, 320, 240, min_size=2)
    assert len(lines) == 2  # 1x1 box removed
    cls, cx, cy, w, h = map(float, lines[1].split())
    assert cls == 0  # pedestrian
    # x=-5..15 clipped to 0..15
    assert np.isclose(w * 320, 15, atol=1e-3) and np.isclose(cx * 320, 7.5, atol=1e-3)


def test_convert_split_masks_ignored_regions(tmp_path):
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "annotations").mkdir()
    img = np.full((240, 320, 3), 0, np.uint8)
    img[100:150, 200:250] = 255  # bright patch inside the ignored region
    cv2.imwrite(str(src / "images" / "f.jpg"), img)
    (src / "annotations" / "f.txt").write_text(ANNOTATION)

    stats = convert_split(src, tmp_path / "out", mask_ignored=True)
    out = cv2.imread(str(tmp_path / "out" / "images" / "f.jpg"))
    assert out[110:140, 210:240].mean() < 100
    assert stats["boxes_out"] == 2
    assert json.loads((tmp_path / "out" / "stats.json").read_text())["images"] == 1
    assert len((tmp_path / "out" / "labels" / "f.txt").read_text().splitlines()) == 2


def _gt_rows():
    # frame, id, x, y, w, h, score, cat, trunc, occ
    rows = []
    for f in range(1, 11):
        rows.append([f, 1, 10 + f, 10, 20, 20, 1, 4, 0, 0])
        rows.append([f, 2, 100, 100 + f, 20, 20, 1, 1, 0, 0])
    rows.append([1, 99, 300, 300, 50, 50, 0, 0, 0, 0])  # ignored region
    return np.array(rows, float)


def test_perfect_tracking_scores_one(tmp_path):
    gt = _gt_rows()
    pred = gt[gt[:, 7] != 0].copy()
    pred[:, 1] += 100  # IDs need not match numerically
    acc = accumulate_sequence(gt, pred)
    import motmetrics as mm

    s = mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches"])
    assert s["mota"].iloc[0] == 1.0 and s["idf1"].iloc[0] == 1.0
    assert s["num_switches"].iloc[0] == 0


def test_id_swap_and_ignored_region(tmp_path):
    gt = _gt_rows()
    pred = gt[gt[:, 7] != 0].copy()
    pred[(pred[:, 0] > 5) & (pred[:, 1] == 1), 1] = 7  # identity changes mid-sequence
    # A false positive inside the ignored region must not be penalized.
    pred = np.vstack([pred, [1, 50, 305, 305, 30, 30, 0.9, 4, -1, -1]])
    (tmp_path / "gt").mkdir()
    (tmp_path / "pred").mkdir()
    np.savetxt(tmp_path / "gt" / "s.txt", gt, delimiter=",", fmt="%d")
    np.savetxt(tmp_path / "pred" / "s.txt", pred, delimiter=",", fmt="%.1f")
    overall, table = evaluate(tmp_path / "gt", tmp_path / "pred")
    assert overall["num_switches"] == 1
    assert overall["num_false_positives"] == 0
    assert overall["idf1"] < 1.0
    assert "OVERALL" in table
