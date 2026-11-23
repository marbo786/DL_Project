"""Inspect training pairs and instrument the existing ReceiptNet training loop."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from api.network import ReceiptNet
from api.signal import SIZE
from research.data import ManifestDocuments
from research.train import cache, collect, measures, probabilities, train


def select_pairs(dataset, count, seed):
    sources = defaultdict(dict)
    for index, row in enumerate(dataset.rows):
        label = row["label"]
        key = (row["source_dataset"], row["source_document_id"])
        group = sources[key]
        if label not in {"genuine", "manipulated"} or label in group:
            raise ValueError("Expected exactly one authentic and one edited row per source.")
        group[label] = index

    by_corpus = defaultdict(list)
    for (corpus, source), rows in sorted(sources.items()):
        if set(rows) != {"genuine", "manipulated"}:
            raise ValueError(f"Incomplete pair for {source}")
        good, edit = rows["genuine"], rows["manipulated"]
        if dataset.rows[good]["image_path"] != dataset.rows[edit]["original_image_path"]:
            raise ValueError(f"Authentic counterpart mismatch for {source}")
        by_corpus[corpus].append((good, edit))
    if count > len(sources):
        raise ValueError(f"Requested {count} pairs but only {len(sources)} are available.")

    rng = random.Random(seed)
    for pairs in by_corpus.values():
        rng.shuffle(pairs)
    ordered = []
    while len(ordered) < count:
        ordered.extend(pairs.pop() for pairs in by_corpus.values() if pairs)
    return ordered[:count]


def change_stats(diff, foreground):
    inside, outside = diff[foreground], diff[~foreground]
    return {
        "foreground_fraction": float(foreground.mean()),
        "changed_inside": int(np.count_nonzero(inside)),
        "changed_outside": int(np.count_nonzero(outside)),
        "mean_delta_inside": float(inside.mean()) if inside.size else None,
        "mean_delta_outside": float(outside.mean()) if outside.size else None,
        "total_delta": float(diff.sum()),
        "inside_delta_fraction": float(inside.sum() / diff.sum()) if diff.any() else None,
    }


def inspect_pair(dataset, pair, preview):
    good, edit = pair
    original, original_mask, genuine = dataset.load(good)
    edited, mask, manipulated = dataset.load(edit)
    if genuine or not manipulated or original_mask.getbbox() or original.size != edited.size:
        raise ValueError("Invalid original/edit label, mask, or oriented dimensions.")
    native_mask = np.asarray(mask)
    if set(np.unique(native_mask)) != {0, 255}:
        raise ValueError("Expected nonempty binary edit mask with some background.")
    difference = ImageChops.difference(original, edited)
    native_diff = np.asarray(difference).max(axis=2)
    original_tensor, _, _ = dataset[good]
    edited_tensor, resized_mask, _ = dataset[edit]
    resized_diff = (edited_tensor - original_tensor).abs().squeeze(0).numpy() * 255
    pooled = F.max_pool2d(resized_mask, 4, stride=4)
    file_metadata = {}
    for role, index in (("authentic", good), ("edited", edit)):
        path = dataset.root / dataset.rows[index]["image_path"]
        with Image.open(path) as image:
            file_metadata[role] = {
                "format": image.format, "mode": image.mode, "bytes": path.stat().st_size,
                "exif_orientation": image.getexif().get(274, 1),
            }
    result = {
        "corpus": dataset.rows[edit]["source_dataset"],
        "dimensions": list(edited.size),
        "file_metadata": file_metadata,
        "native": change_stats(native_diff, native_mask > 0),
        "resized": change_stats(resized_diff, resized_mask[0].numpy() > 0),
        "pooled_foreground_fraction": pooled.mean().item(),
    }
    if preview is not None:
        overlay = Image.composite(Image.blend(edited, Image.new("RGB", edited.size, "red"), .45), edited, mask)
        resized = Image.fromarray(np.rint((edited_tensor[0].numpy() + .5) * 255).astype(np.uint8))
        panels = [
            ("Authentic", original), ("Edited", edited), ("RGB absolute difference (4x)", difference.point(lambda x: min(x * 4, 255))),
            ("Original mask", mask), ("Edited + mask", overlay), ("112 x 160 input", resized),
            ("112 x 160 mask", Image.fromarray((resized_mask[0].numpy() * 255).astype(np.uint8))),
        ]
        page = Image.new("RGB", (4 * 220, 2 * 280), "#20242b")
        draw = ImageDraw.Draw(page)
        for index, (title, image) in enumerate(panels):
            x, y = index % 4 * 220, index // 4 * 280
            image = image.copy()
            image.thumbnail((204, 244))
            draw.text((x + 8, y + 5), title, fill="white")
            page.paste(image, (x + (220 - image.width) // 2, y + 25))
        page.save(preview)
    return result


def distribution(values):
    return dict(zip(("min", "p25", "median", "p75", "max"), np.quantile(values, [0, .25, .5, .75, 1]).tolist()))


def score_diagnostics(outputs, temperature=1., bias=0.):
    scores, grid, masks, labels = outputs
    probability = probabilities(scores, temperature, bias)
    return {
        "accuracy": float(np.mean((probability >= .5) == labels)),
        "mean_probability": float(probability.mean()),
        "prevalence": float(labels.mean()),
        "target_positive_fraction_on_edits": float(np.mean(masks[labels == 1] > 0)),
        "predicted_positive_fraction_by_class": {
            label: float(np.mean(grid[labels == value] >= 0))
            for label, value in (("genuine", 0), ("manipulated", 1))
        },
        "median_within_image_grid_logit_std": float(np.median(grid.std(axis=(1, 2)))),
        "by_class": {
            label: {
                "logits": distribution(scores[labels == value]),
                "probabilities": distribution(probability[labels == value]),
            }
            for label, value in (("genuine", 0), ("manipulated", 1))
        },
        **measures(outputs, temperature=temperature, bias=bias),
    }


class TrainingTrace:
    def __init__(self, path):
        self.path = path
        self.batches = []

    def batch(self, model, class_loss, pixel_loss):
        gradient = torch.stack([parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None])
        self.batches.append((class_loss.item(), pixel_loss.item(), gradient.norm().item()))

    def epoch(self, epoch, outputs, metrics):
        class_loss, pixel_loss, gradient = np.mean(self.batches, axis=0).tolist()
        row = {
            "epoch": epoch, "train_loss": metrics["train_loss"], "classification_loss_mean_per_batch": class_loss,
            "localization_loss_mean_per_batch": pixel_loss, "gradient_l2_mean_per_batch": gradient,
            "gradient_l2_by_batch": distribution(np.asarray(self.batches)[:, 2]),
            **score_diagnostics(outputs),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.batches.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "overfit", "calibration"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--previews", type=int, default=0, help="Save selected receipt images locally when authorized.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    overfit = parser.add_argument_group("overfit controls")
    overfit.add_argument("--resolution", type=int, choices=(1, 2), default=1)
    overfit.add_argument("--class-weight", type=float, default=1.)
    overfit.add_argument("--map-weight", type=float, default=.65)
    overfit.add_argument("--learning-rate", type=float, default=.003)
    args = parser.parse_args()
    if min(args.pairs, args.epochs, args.threads) < 1 or args.previews < 0:
        parser.error("use positive counts and a nonnegative number of previews")
    if args.mode == "calibration" and args.checkpoint is None:
        parser.error("calibration requires the existing benchmark --checkpoint")
    settings = (args.resolution, args.class_weight, args.map_weight, args.learning_rate)
    if args.mode != "overfit" and settings != (1, 1., .65, .003):
        parser.error("training controls are only available in overfit mode")
    weights = (args.class_weight, args.map_weight)
    if not all(np.isfinite(settings)) or min(weights) < 0 or not any(weights) or args.learning_rate <= 0:
        parser.error("use finite nonnegative loss weights, at least one active loss, and a positive learning rate")
    torch.set_num_threads(args.threads)
    size = tuple(axis * args.resolution for axis in SIZE)
    dataset = ManifestDocuments(args.manifest, args.asset_root, "validation" if args.mode == "calibration" else "train", size=size)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "calibration":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("scope") != "manifest":
            raise ValueError("Use the existing manifest benchmark checkpoint.")
        model = ReceiptNet(**checkpoint["config"]).eval()
        model.load_state_dict(checkpoint["weights"])
        outputs = collect(model, cache(dataset))
        result = {
            "scope": "validation", "example_count": len(dataset),
            "temperature": checkpoint["temperature"], "bias": checkpoint.get("bias", 0.),
            "raw": score_diagnostics(outputs),
            "calibrated": score_diagnostics(outputs, checkpoint["temperature"], checkpoint.get("bias", 0.)),
        }
    else:
        pairs = select_pairs(dataset, args.pairs, args.seed)
        result = {"scope": "train", "seed": args.seed, "pair_count": len(pairs), "input_size": list(size)}
        if args.mode == "inspect":
            result["pairs"] = [
                inspect_pair(dataset, pair, args.output / f"pair-{index + 1:03}.png" if index < args.previews else None)
                for index, pair in enumerate(pairs)
            ]
        else:
            batch = cache(Subset(dataset, [index for pair in pairs for index in pair]))
            result["epochs"] = args.epochs
            result["scope"] = "train-only memorization"
            result["objective"] = {
                "classification_weight": args.class_weight,
                "localization_weight": args.map_weight,
                "learning_rate": args.learning_rate,
            }
            train("local", {"width": 8}, batch, batch, args.epochs, args.seed,
                  observer=TrainingTrace(args.output / "epochs.jsonl"), class_weight=args.class_weight,
                  map_weight=args.map_weight, learning_rate=args.learning_rate)
    (args.output / "diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
