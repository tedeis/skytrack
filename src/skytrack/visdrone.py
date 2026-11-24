import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

palette = plt.get_cmap("tab10").colors          # 10 RGB tuples with values 0–1
colors = {cat: tuple(int(c * 255) for c in reversed(palette[cat - 1]))
    for cat in range(1, 11)}              # reversed() turns RGB into BGR
CATEGORY_NAMES = {
    0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others",
}

def load_annotations(ann_dir, img_dir):
    frames = []
    image_dir = Path(img_dir)
    annotations_dir = Path(ann_dir)
    for annotation_file in annotations_dir.glob("*.txt"):
        img_path = image_dir / f"{annotation_file.stem}.jpg"
        img_w, img_h = Image.open(img_path).size
        ann = pd.read_csv(annotation_file, header=None, usecols=range(8), names = ["x", "y", "w", "h", "score", "category", "truncation", "occlusion"])
        ann["image"] = annotation_file.stem
        ann["size"] = np.sqrt(ann["w"] * ann["h"])
        ann["img_w"] = img_w
        ann["img_h"] = img_h
        ann["size_640"] = ann["size"] * 640 / ann[["img_w", "img_h"]].max(axis=1)
        frames.append(ann)
    df = pd.concat(frames, ignore_index=True)
    df["name"] = df["category"].map(CATEGORY_NAMES)
    return df
    
def draw_boxes(img_data, annotation):
    vis = img_data.copy() #draw on a copy so we don't accidentally save or train on images with boxes painted onto it
    colors[0] = (128, 128, 128)                     # ignored regions: gray
    colors[11] = (255, 255, 255)                    # others: white
    for row in annotation:
        x, y, w, h, score, cat = row[:6]
        vis = cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), colors[int(cat)], thickness=2)
        
    return vis

def convert_split(src_dir, dst_dir):
    """Convert one VisDrone split to YOLO format: painted images + label files."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir) 
    src_img_dir = src_dir / "images"    
    src_ann_dir = src_dir / "annotations"
    out_img_dir = dst_dir / "images"
    out_lbl_dir = dst_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    df = load_annotations(src_ann_dir, src_img_dir)
    counts = {"images": df["image"].nunique(), "raw_rows": len(df)} 

    # Rule 5 rows (painted later), taken before filtering removes them
    ignored = df[df["category"] == 0]

    # Rule 1: real objects only
    clean_df = df[(df["score"] == 1) & (df["category"].between(1, 10))].copy()
    counts["rule1_removed"] = len(df) - len(clean_df)

    # Rule 3: clip boxes to the image (work with corners, then rebuild w and h)
    x1 = clean_df["x"].clip(lower=0)
    y1 = clean_df["y"].clip(lower=0)
    x2 = (clean_df["x"] + clean_df["w"]).clip(upper=clean_df["img_w"])
    y2 = (clean_df["y"] + clean_df["h"]).clip(upper=clean_df["img_h"])
    counts["rule3_clipped"] = int((
        (x1 != clean_df["x"]) | (y1 != clean_df["y"]) |
        (x2 != clean_df["x"] + clean_df["w"]) | (y2 != clean_df["y"] + clean_df["h"])
    ).sum())
    clean_df["x"], clean_df["y"] = x1, y1
    clean_df["w"], clean_df["h"] = x2 - x1, y2 - y1

    # Rule 4: drop boxes under 2 px (after clipping, so cut-off boxes get caught)
    before = len(clean_df)
    clean_df = clean_df[(clean_df["w"] >= 2) & (clean_df["h"] >= 2)].copy()
    counts["rule4_removed"] = before - len(clean_df)

    # Rule 2 + YOLO conversion
    clean_df["cls"] = clean_df["category"] - 1
    clean_df["x_center"] = (clean_df["x"] + clean_df["w"] / 2) / clean_df["img_w"]
    clean_df["y_center"] = (clean_df["y"] + clean_df["h"] / 2) / clean_df["img_h"]
    clean_df["width"] = clean_df["w"] / clean_df["img_w"]
    clean_df["height"] = clean_df["h"] / clean_df["img_h"]
    yolo_cols = ["cls", "x_center", "y_center", "width", "height"]

    # Per image: paint, save the image, write the labels
    for stem in df["image"].unique():
        img_path = src_img_dir / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)

        img_ignored = ignored[ignored["image"] == stem]
        img_kept = clean_df[clean_df["image"] == stem]

        # Rule 5: paint ignored regions gray
        for row in img_ignored.itertuples():
            x, y = max(int(row.x), 0), max(int(row.y), 0)
            x2, y2 = int(row.x + row.w), int(row.y + row.h)
            img[y:y2, x:x2] = 114

        cv2.imwrite(str(out_img_dir / f"{stem}.jpg"), img)
        img_kept[yolo_cols].to_csv(out_lbl_dir / f"{stem}.txt", sep=" ",
                                   header=False, index=False, float_format="%.6f")

    counts["boxes_kept"] = len(clean_df)
    return counts
