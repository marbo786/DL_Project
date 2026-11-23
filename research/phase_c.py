"""Phase C controlled off-mask training and image-only receipt evaluation."""

import argparse
from hashlib import sha256
from math import ceil, comb
from pathlib import Path
import json
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from research.data import ManifestDocuments
from research.diagnose import select_pairs
from research.hard_negatives import BackgroundTiles
from research.patch_data import check_source_splits
from research.patch_probe import PatchNet, document_scores, metrics, pool, quantiles, write_history


SIDE, STRIDE, THRESHOLD, BATCH = 112, 84, .005, 8
POOLS = ("max", "top3", "mean", "top5_percent")


def pooling(logits):
    count = max(1, ceil(.05 * len(logits)))
    return [*pool(logits), float(np.partition(logits, -count)[-count:].mean())]


def localization(scores, overlaps, areas, mask_area, ranks=(1, 3, 5)):
    order = np.argsort(-scores, kind="stable")
    total = len(order)
    clean = int(np.sum(overlaps == 0))
    result = {}
    for rank in ranks:
        count = min(rank, total)
        result[f"hit_at_{rank}"] = float(np.any(overlaps[order[:count]] > 0))
        result[f"uniform_hit_at_{rank}"] = 1 - comb(clean, count) / comb(total, count)
    first = order[0]
    intersection = overlaps[first]
    result["top_tile_iou"] = float(intersection / (areas[first] + mask_area - intersection))
    result["best_tile_coverage"] = float(overlaps.max() / mask_area)
    return result


def paired_intervals(values, reference=None, draws=1000):
    rng = np.random.default_rng(42)
    samples = rng.integers(len(values), size=(draws, len(values)))

    def auc(scores, positions):
        pairs = scores[positions]
        difference = pairs[:, 1, None] - pairs[None, :, 0]
        return ((difference > 0).mean() + .5 * (difference == 0).mean())

    estimates = [auc(values, indices) - (auc(reference, indices) if reference is not None else 0)
                 for indices in samples]
    return np.quantile(estimates, [.025, .975]).tolist()


def load_rgb(documents, row):
    with Image.open(documents.root / documents.rows[row]["image_path"]) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def crop_thumb(image, box):
    crop = image.crop((box[0], box[1], min(box[2], image.width), min(box[3], image.height)))
    return ImageOps.contain(crop, (104, 104))


def save_gallery(cases, directory):
    width, height = 660, 156
    for category, candidates in cases.items():
        examples = sorted(candidates, key=lambda row: -row[0])[:6]
        if not examples:
            continue
        sheet = Image.new("RGB", (width, height * len(examples)), "#20232b")
        draw = ImageDraw.Draw(sheet)
        for row, (score, prediction, reference, overlap) in enumerate(examples):
            top = height * row
            title = f"{category} | logit {score:.2f} | selected tile intersection {overlap} px"
            draw.text((10, top + 8), title, fill="white")
            draw.text((10, top + 27), "Top-scoring tile", fill="white")
            sheet.paste(prediction, (10, top + 44))
            if reference is not None:
                draw.text((132, top + 27), "Best mask-overlap tile", fill="white")
                sheet.paste(reference, (132, top + 44))
        sheet.save(directory / f"{category}.png")


def evaluate(model, documents, count, device, gallery=None, spatial=False):
    values, durations, counts, details, authentic_exceedance = [], [], [], [], []
    cases = {"ranked_edits": [], "authentic_top_tiles": [], "missed_edits": []}
    model.eval()
    for position, pair in enumerate(select_pairs(documents, count, 42), 1):
        scores = []
        for row in pair:
            start = perf_counter()
            image = load_rgb(documents, row)
            boxes, logits = document_scores(model, image, SIDE, STRIDE, "rgb", device, 32)
            scores.append(pooling(logits))
            durations.append((perf_counter() - start) * 1000)
            if row == pair[0]:
                authentic_exceedance.append(int(np.count_nonzero(logits >= 0)))
                if gallery is not None:
                    best = int(logits.argmax())
                    cases["authentic_top_tiles"].append((float(logits[best]), crop_thumb(image, boxes[best]), None, 0))
        values.append(scores)
        counts.append(len(boxes))
        if spatial:
            _, mask, _ = documents.load(pair[1])
            overlaps = np.array([mask.crop(box).histogram()[255] for box in boxes])
            areas = [(min(x1, mask.width) - x0) * (min(y1, mask.height) - y0) for x0, y0, x1, y1 in boxes]
            result = localization(logits, overlaps, np.asarray(areas), mask.histogram()[255])
            details.append(result)
            if gallery is not None:
                best, target = int(logits.argmax()), int(overlaps.argmax())
                category = "ranked_edits" if result["hit_at_1"] else "missed_edits"
                cases[category].append((float(logits[best]), crop_thumb(image, boxes[best]),
                                        crop_thumb(image, boxes[target]), int(overlaps[best])))
        if position % 32 == 0 or position == count:
            print(f"tiled {position}/{count} pairs", flush=True)
    if gallery is not None:
        save_gallery(cases, gallery)
    return np.asarray(values), np.asarray(counts), np.asarray(authentic_exceedance), durations, details


def collate(samples):
    images, labels = zip(*samples)
    return torch.cat(images), torch.cat(labels)


def fit(model, optimizer, dataset, start_epoch, end_epoch, seed, device, output, history):
    updates = 0
    model.train()
    for epoch in range(start_epoch, end_epoch + 1):
        loss_sum = 0.
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed + epoch))
        for images, labels in loader:
            labels = labels.to(device)
            logits = model(images.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            positive = labels == 1
            loss = .5 * (loss[positive].mean() + loss[~positive].mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        updates += len(loader)
        history.append({"epoch": epoch, "loss": loss_sum / len(loader), "updates": len(loader)})
        write_history(output / "epochs.jsonl", history)
        print(f"epoch {epoch}: class-balanced BCE {loss_sum / len(loader):.5f}", flush=True)
    return updates


def cache_training(dataset, limit_bytes=160 * 1024 ** 2):
    n = sum(sum(row[key] for key in ("positive", "matched", "random", "hard")) for row in dataset.counts)
    return list(dataset) if n * 3 * SIDE ** 2 <= limit_bytes else dataset


def aggregate(train, validation):
    learned = make_pipeline(StandardScaler(), LogisticRegression())
    train3 = train[:, :, :3].reshape(-1, 3)
    validation3 = validation[:, :, :3].reshape(-1, 3)
    learned.fit(train3, np.tile([0, 1], len(train)))
    return learned.decision_function(train3).reshape(-1, 2), learned.decision_function(validation3).reshape(-1, 2)


def report_metrics(training, validation, counts, authentic, elapsed, details, reference):
    train_pool, val_pool = aggregate(training, validation)
    report = {}
    for name, train_values, val_values in (
        *[(name, training[:, :, i], validation[:, :, i]) for i, name in enumerate(POOLS)],
        ("trained_pooling", train_pool, val_pool),
    ):
        threshold = float(np.quantile(train_values[:, 0], .90))
        report[name] = {**metrics(val_values), "auroc_ci95": paired_intervals(val_values),
                        "training_genuine_p90_threshold": threshold,
                        "at_training_genuine_p90": metrics(val_values - threshold)}
    if reference is not None:
        report["max_delta_ci95"] = paired_intervals(validation[:, :, 0], reference[:, :, 0])
    summary = {name: float(np.mean([row[name] for row in details])) for name in details[0]}
    hit_draws = np.random.default_rng(42).integers(len(details), size=(1000, len(details)))
    hits = np.array([row["hit_at_1"] for row in details])
    summary["hit_at_1_ci95"] = np.quantile(hits[hit_draws].mean(axis=1), [.025, .975]).tolist()
    summary["best_tile_coverage_quantiles"] = quantiles([row["best_tile_coverage"] for row in details])
    bins = {}
    for key, chosen in (("below_144", counts < 144), ("at_least_144", counts >= 144)):
        if chosen.any():
            bins[key] = {"pairs": int(chosen.sum()), "genuine_tiles_above_zero_mean": float(authentic[chosen].mean()),
                         "genuine_tile_positive_fraction": float(authentic[chosen].sum() / counts[chosen].sum()),
                         "max_fpr": metrics(validation[chosen, :, 0])["false_positive_rate"],
                         "top5_percent_fpr": metrics(validation[chosen, :, 3])["false_positive_rate"]}
    return {"pools": report, "localization": summary, "tile_count_bins": bins,
            "tiles_per_receipt_quantiles": quantiles(counts), "latency_median_ms": float(np.median(elapsed)),
            "latency_p95_ms": float(np.quantile(elapsed, .95))}


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = ManifestDocuments(args.manifest, args.asset_root, "train")
    validation = ManifestDocuments(args.manifest, args.asset_root, "validation")
    check_source_splits(training, validation)
    digest = sha256(args.manifest.read_bytes()).hexdigest()
    if digest != "3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80":
        raise ValueError("The manifest does not match Phase B")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model = PatchNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    history, summary = [], {}
    start = perf_counter()
    if args.mode == "baseline":
        saved = torch.load(args.baseline, map_location=device, weights_only=True)
        if saved["config"]["manifest_sha256"] != digest or saved["config"]["seed"] != 42:
            raise ValueError("Expected the verified Phase B fixed-tile checkpoint")
        model.load_state_dict(saved["best"]["weights"])
        summary["historical_checkpoint"] = str(args.baseline)
        summary["selected_epoch"] = saved["best"]["epoch"]
    else:
        if args.mode == "random":
            first = BackgroundTiles(training, 128, args.seed, SIDE, STRIDE, THRESHOLD, 2 * args.ratio)
            dataset = cache_training(first)
            source_counts = first.counts
            fit(model, optimizer, dataset, 1, 3, args.seed, device, args.output, history)
            torch.save({"weights": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
                        "seed": args.seed, "ratio": args.ratio}, args.output / "epoch3.tmp")
            (args.output / "epoch3.tmp").replace(args.output / "epoch3.pt")
            fit(model, optimizer, dataset, 4, 6, args.seed, device, args.output, history)
        else:
            saved = torch.load(args.initial, map_location=device, weights_only=True)
            if saved["seed"] != args.seed or saved["ratio"] != args.ratio or len(saved["history"]) != 3:
                raise ValueError("Mining requires a matching epoch-three random checkpoint")
            model.load_state_dict(saved["weights"])
            optimizer.load_state_dict(saved["optimizer"])
            history = saved["history"]
            mined = BackgroundTiles(training, 128, args.seed, SIDE, STRIDE, THRESHOLD, args.ratio, args.ratio,
                                    model=model, device=device)
            fit(model, optimizer, cache_training(mined), 4, 6, args.seed, device, args.output, history)
            summary["mining"] = mined.mining
            source_counts = mined.counts
        summary["patch_counts"] = {name: sum(row[name] for row in source_counts)
                                   for name in ("positive", "matched", "random", "hard")}
        (args.output / "training_sources.json").write_text(json.dumps(source_counts, indent=2), encoding="utf-8")
        summary["optimizer_updates"] = sum(row["updates"] for row in history)
        summary["selected_epoch"] = 6
        torch.save(model.state_dict(), args.output / "final.pt")
    summary["training_and_mining_seconds"] = perf_counter() - start
    train_values, *_ = evaluate(model, training, 128, device)
    values, counts, authentic, timings, details = evaluate(model, validation, args.val_pairs, device,
                                                        args.output if args.gallery else None, spatial=True)
    np.save(args.output / "validation_scores.npy", values)
    reference = np.load(args.reference) if args.reference is not None else None
    summary.update(report_metrics(train_values, values, counts, authentic, timings, details, reference))
    summary.update(mode=args.mode, seed=args.seed, ratio=args.ratio, validation_pairs=args.val_pairs,
                   elapsed_seconds=perf_counter() - start)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "random", "hard"))
    for name in ("manifest", "asset-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline", "initial", "reference"):
        parser.add_argument("--" + name, type=Path)
    for name, default in (("seed", 42), ("threads", 5), ("ratio", 1), ("val-pairs", 64)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--gallery", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.ratio, args.val_pairs) < 1 or args.ratio > 4 or args.val_pairs > 228:
        parser.error("thread count, ratio (1–4), and validation pair count (1–228) must be in range")
    if (args.mode == "baseline" and args.baseline is None) or (args.mode == "hard" and args.initial is None):
        parser.error("provide the corresponding baseline or initialization checkpoint")
    run(args)
