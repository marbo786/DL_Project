"""Stage private inputs for official TruFor and collect its map and document outputs."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from research.manifest import file_sha256, resolve_image
from research.patch_data import tile_boxes
from research.receipt_model import SIDE

TRUFOR_COMMIT = "ae54475df6f41a491d7615100feb19263dec13f7"
SPLITS = ("calibration", "validation", "test", "shifted_test")


def input_name(row):
    return sha256(row["sample_id"].encode()).hexdigest() + ".png"


def rows_for(args):
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == args.split]
    if not rows:
        raise ValueError("Empty split")
    return rows


def stage(args):
    rows = rows_for(args)
    if args.smoke:
        rows = [next(row for row in rows if row["label"] == label) for label in ("genuine", "manipulated")]
    args.output.mkdir(parents=True)
    for row in rows:
        source = resolve_image(args.root.resolve(), row["image_path"])
        with Image.open(source) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(args.output / input_name(row))


def read_map(path, shape):
    with np.load(path, allow_pickle=False) as prediction:
        score = float(prediction["score"].item())
        heatmap = prediction["map"]
        if (heatmap.shape != shape or not np.isfinite(heatmap).all()
                or (heatmap < 0).any() or (heatmap > 1).any() or not 0 <= score <= 1):
            raise ValueError(f"Invalid official score or raster in {path.name}")
        return score, heatmap


def collect(args):
    rows = rows_for(args)
    if args.smoke:
        rows = [next(row for row in rows if row["label"] == label) for label in ("genuine", "manipulated")]
    data = []
    for row in rows:
        label = row["label"] == "manipulated"
        source = resolve_image(args.root.resolve(), row["image_path"])
        with Image.open(source) as image:
            width, height = ImageOps.exif_transpose(image).size
        path = args.predictions / (input_name(row) + ".npz")
        score, heatmap = read_map(path, (height, width))
        boxes = [(x, y, min(end_x, width), min(end_y, height))
                 for x, y, end_x, end_y in tile_boxes((width, height), SIDE, SIDE)]
        x, y, end_x, end_y = max(boxes, key=lambda box: heatmap[box[1]:box[3], box[0]:box[2]].mean())
        iou, hit, pixel_iou = "", "", ""
        if label:
            with Image.open(resolve_image(args.root.resolve(), row["mask_path"])) as source:
                mask = np.asarray(ImageOps.exif_transpose(source).convert("L")) > 0
            if mask.shape != heatmap.shape or not mask.any():
                raise ValueError("Invalid evaluation mask")
            intersection = mask[y:end_y, x:end_x].sum()
            union = mask.sum() + (end_x - x) * (end_y - y) - intersection
            iou, hit = float(intersection / union), int(intersection > 0)
            thresholded = heatmap >= .5
            pixel_iou = float((thresholded & mask).sum() / (thresholded | mask).sum())
        data.append((row["sample_id"], row["source_group"], row["source_dataset"], int(label), score,
                     iou, hit, f"{x} {y} {end_x} {end_y}", "", "clean", pixel_iou))
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_id", "source_group", "corpus", "label", "logit", "iou", "hit",
                         "bbox", "milliseconds", "condition", "native_mask_iou_at_0_5"))
        writer.writerows(data)
    sidecar = args.output.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({"upstream_commit": TRUFOR_COMMIT, "checkpoint_sha256": file_sha256(args.weights),
                                  "manifest_sha256": file_sha256(args.manifest), "smoke": args.smoke,
                                  "split": args.split, "scores_sha256": file_sha256(args.output)}, indent=2) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    staging = commands.add_parser("stage")
    collecting = commands.add_parser("collect")
    for command in (staging, collecting):
        for name in ("manifest", "root", "output"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--split", choices=SPLITS, required=True)
        command.add_argument("--smoke", action="store_true")
    collecting.add_argument("--predictions", type=Path, required=True)
    collecting.add_argument("--weights", type=Path, required=True)
    args = cli.parse_args()
    {"stage": stage, "collect": collect}[args.action](args)
