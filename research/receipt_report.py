"""Generate figures and inspect a fixed set of development errors locally."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from research.manifest import resolve_image


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def diagnostic_plots(metrics, output):
    fig, ax = plt.subplots(figsize=(5.7, 4.4))
    ax.plot([0, 1], [0, 1], color=".6", linestyle="--", label="Perfect calibration")
    for key, name, marker in (("reliability", "Fitted model", "o"),
                              ("constant_reliability", "Constant prevalence", "x")):
        bins = metrics[key]["bins"]
        ax.plot([b["mean_probability"] for b in bins], [b["observed_fraction"] for b in bins],
                marker=marker, label=name)
    ax.set(xlabel="Mean predicted edit probability", ylabel="Observed edit fraction",
           xlim=(0, 1), ylim=(0, 1), title="Development reliability")
    ax.legend()
    save(fig, output / "reliability.png")

    fig, ax = plt.subplots(figsize=(5.7, 4.4))
    for key, name, marker in (("risk_coverage", "Fitted model", "."), ("constant_risk_coverage", "Constant prevalence", "x")):
        points = [row for row in metrics[key] if row["accepted_error"] is not None]
        ax.plot([row["coverage"] for row in points], [row["accepted_error"] for row in points],
                marker=marker, label=name)
        if key == "risk_coverage":
            smallest = min(points, key=lambda row: row["coverage"])
            count = round(smallest["coverage"] * metrics["documents"])
            if count < 5:
                ax.annotate(f"{count} of {metrics['documents']} accepted",
                            (smallest["coverage"], smallest["accepted_error"]), xytext=(18, 20),
                            textcoords="offset points", arrowprops={"arrowstyle": "-", "color": ".4"})
    ax.set(xlabel="Fraction accepted", ylabel="Error among accepted documents", xlim=(0, 1.02),
           ylim=(0, 1), title="Development risk and coverage")
    ax.legend()
    save(fig, output / "risk-coverage.png")


def inspect_examples(metrics, scores, manifest, root, output):
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
    evidence = []
    groups = set()
    candidates = [(row, "high_scoring_authentic") for row in metrics["highest_scoring_authentic"]]
    candidates += [(row, "low_scoring_edit") for row in metrics["lowest_scoring_edits"]]
    with scores.open(newline="", encoding="utf-8") as stream:
        ranked = sorted(csv.DictReader(stream), key=lambda row: float(row["logit"]))
    threshold = metrics["operating_points"]["0.05"]["logit_threshold"]
    candidates += [(row, "localized_and_flagged_edit") for row in ranked[::-1]
                   if row["hit"] == "1" and float(row["logit"]) > threshold][:3]
    candidates += [(row, "low_scoring_authentic") for row in ranked if row["label"] == "0"][:2]
    for score, kind in candidates:
        row = rows[score["sample_id"]]
        if row["source_group"] in groups:
            continue
        groups.add(row["source_group"])
        release = {"sample_id": score["sample_id"], "kind": kind, "corpus": score["corpus"],
                   "score": float(score["logit"]), "tile": score["bbox"]}
        box = tuple(map(int, score["bbox"].split()))
        with Image.open(resolve_image(root, row["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("L")
            pixels = np.asarray(image.crop(box), dtype=np.float32)
            release["tile_ink_fraction_below_gray_96"] = float((pixels < 96).mean())
            release["tile_edge_strength"] = float(np.abs(np.diff(pixels, axis=0)).mean() +
                                                   np.abs(np.diff(pixels, axis=1)).mean())
            release["tile_center_y_fraction"] = (box[1] + box[3]) / (2 * image.height)
        if row["mask_path"]:
            with Image.open(resolve_image(root, row["mask_path"])) as source:
                mask = ImageOps.exif_transpose(source).convert("L")
                release["mask_fraction"] = mask.histogram()[255] / (mask.width * mask.height)
                release["tile_mask_overlap_pixels"] = mask.crop(box).histogram()[255]
                release["mask_bbox"] = mask.getbbox()
        evidence.append(release)
    (output / "case-review.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def learning_curve(study, seed, output):
    folders = [study / f"full-{seed}-fraction-{value}" for value in ("0.25", "0.5")] + [study / f"full-{seed}"]
    if not all((folder / "metrics.json").exists() for folder in folders):
        return
    fit = [read(folder / "fit.json") for folder in folders]
    metrics = [read(folder / "metrics.json") for folder in folders]
    values = [report["auroc"] for report in metrics]
    limits = np.array([report["source_bootstrap_95"]["auroc"] for report in metrics])
    fig, ax = plt.subplots(figsize=(5.7, 4.4))
    groups = [report["trained_groups"] for report in fit]
    ax.plot(groups, values, marker="o", color="#214e87")
    ax.vlines(groups, limits[:, 0], limits[:, 1], color="#214e87")
    ax.axhline(.5, linestyle="--", color=".6")
    ax.set(xlabel="Fitting source groups", ylabel="Development AUROC", title=f"RGB full fine-tuning, seed {seed}")
    save(fig, output / "learning-curve.png")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "root", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = read(args.run / "metrics.json")
    diagnostic_plots(report, args.output)
    inspect_examples(report, args.run / "validation.csv", args.manifest, args.root.resolve(), args.output)
    learning_curve(args.run.parent, read(args.run / "fit.json")["seed"], args.output)
