"""Measure local receipt evidence on the existing development partitions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from api.network import block
from research.data import ManifestDocuments
from research.patch_data import MatchedCrops, check_source_splits, rgb_crop, tile_boxes
from research.diagnose import select_pairs


class PatchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(block(6, 8), block(8, 16), block(16, 24), block(24, 32))
        self.head = nn.Linear(64, 1)

    def forward(self, image):
        image = image.float().div(255).sub(.5)
        smooth = F.avg_pool2d(image, 3, stride=1, padding=1, count_include_pad=False)
        features = self.layers(torch.cat((image, 3 * (image - smooth)), dim=1))
        return self.head(torch.cat((features.mean((2, 3)), features.amax((2, 3))), dim=1)).flatten()


def quantiles(values):
    return np.quantile(values, [0, .25, .5, .75, 1]).tolist() if len(values) else []


def metrics(scores):
    paired = np.asarray(scores).reshape(-1, 2)
    logits = paired.ravel()
    labels = np.tile([0, 1], len(paired))
    verdicts = logits >= 0
    probability = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
    return {
        "pairs": len(paired), "accuracy": accuracy_score(labels, verdicts),
        "auroc": roc_auc_score(labels, logits), "average_precision": average_precision_score(labels, logits),
        "macro_f1": f1_score(labels, verdicts, average="macro"),
        "recall": recall_score(labels, verdicts), "false_positive_rate": float(verdicts[labels == 0].mean()),
        "bce": float(np.mean(np.logaddexp(0, logits) - labels * logits)),
        "paired_win_rate": float(np.mean((paired[:, 1] > paired[:, 0]) + .5 * (paired[:, 1] == paired[:, 0]))),
        "genuine_probability_quantiles": quantiles(probability[::2]),
        "edited_probability_quantiles": quantiles(probability[1::2]),
    }


def texture_statistics(batch):
    image = batch.float().div(255)
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()
    vertical = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs()
    return torch.cat((image.mean((2, 3)), image.std((2, 3)), horizontal.mean((2, 3)),
                      vertical.mean((2, 3)), horizontal.std((2, 3)), vertical.std((2, 3))), dim=1)


def bounded_cache(dataset, mib):
    if len(dataset) * 2 * 3 * dataset.side ** 2 <= mib * 1024 ** 2:
        return torch.stack(list(dataset))
    return dataset


def patch_pass(model, data, batch, device, optimizer=None, seed=42):
    model.train(optimizer is not None)
    loader = DataLoader(data, batch_size=batch, shuffle=optimizer is not None,
                        generator=torch.Generator().manual_seed(seed))
    scores = []
    with torch.set_grad_enabled(optimizer is not None):
        for pairs in loader:
            inputs = pairs.flatten(0, 1).to(device)
            logits = model(inputs)
            if optimizer is not None:
                labels = torch.tensor([0., 1.], device=device).repeat(len(pairs))
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            scores.extend(logits.detach().cpu().tolist())
    return metrics(scores)


def write_history(destination, history):
    pending = destination.with_suffix(".tmp")
    pending.write_text("".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")
    pending.replace(destination)


def shortcut_probe(training, validation, batch):
    def features(dataset):
        return torch.cat([texture_statistics(p.flatten(0, 1)) for p in DataLoader(dataset, batch_size=batch)]).numpy()

    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(features(training), np.tile([0, 1], len(training)))
    return metrics(model.decision_function(features(validation)))


@torch.inference_mode()
def document_scores(model, image, side, stride, transform, device, batch):
    boxes = tile_boxes(image.size, side, stride)
    logits = []
    for start in range(0, len(boxes), batch):
        images = torch.stack([rgb_crop(image, box, transform) for box in boxes[start:start + batch]])
        logits.extend(model(images.to(device)).cpu().tolist())
    return boxes, np.asarray(logits)


def pool(logits):
    count = min(3, len(logits))
    return [float(logits.max()), float(np.sort(logits)[-count:].mean()), float(logits.mean())]


def stratify(scores, factors, split):
    array = np.asarray(factors)
    return {name: {"pairs": int(group.sum()), "max_auroc": metrics(scores[group, :, 0])["auroc"]}
            for name, group in (("below", array < split), ("at_or_above", array >= split)) if group.any()}


def tiled_pass(model, documents, count, sample_seed, config, device):
    output, timings, hit, random_hit, iou, counts, coverage = [], [], [], [], [], [], []
    model.eval()
    for position, pair in enumerate(select_pairs(documents, count, sample_seed), start=1):
        document = []
        for index in pair:
            start = perf_counter()
            # Decode only the receipt in the inference timer. Masks are loaded below for scoring.
            path = documents.root / documents.rows[index]["image_path"]
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            boxes, logits = document_scores(model, image, config["side"], config["stride"],
                                              config["transform"], device, config["tile_batch"])
            document.append(pool(logits))
            timings.append(1000 * (perf_counter() - start))
        _, mask, _ = documents.load(pair[1])
        box = boxes[logits.argmax()]
        overlaps = [mask.crop(candidate).histogram()[255] for candidate in boxes]
        intersection = overlaps[logits.argmax()]
        foreground = mask.histogram()[255]
        area = (min(box[2], mask.width) - box[0]) * (min(box[3], mask.height) - box[1])
        hit.append(intersection > 0)
        random_hit.append(np.mean(np.asarray(overlaps) > 0))
        iou.append(intersection / (foreground + area - intersection))
        counts.append(len(boxes))
        coverage.append(max(overlaps) / foreground)
        output.append(document)
        if position % 16 == 0 or position == count:
            print(f"tiled receipts: {position}/{count} pairs", flush=True)
    scores = np.asarray(output)
    median_count = int(np.median(counts))
    return scores, {
        "latency_ms_quantiles": quantiles(timings), "latency_ms_p95": float(np.quantile(timings, .95)),
        "tiles_per_edited_document": quantiles(counts), "top_tile_mask_hit_rate": float(np.mean(hit)),
        "uniform_tile_mask_hit_rate": float(np.mean(random_hit)),
        "top_tile_mask_iou": float(np.mean(iou)), "best_tile_mask_coverage": quantiles(coverage),
        "by_tile_count": {"split": median_count, **stratify(scores, counts, median_count)},
        "by_coverage": {"split": .5, **stratify(scores, coverage, .5)},
    }


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_docs = ManifestDocuments(args.manifest, args.asset_root, "train")
    val_docs = ManifestDocuments(args.manifest, args.asset_root, "validation")
    check_source_splits(train_docs, val_docs)
    config = {"manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(), "train_pairs": args.train_pairs,
              "val_pairs": args.val_pairs, "seed": args.seed, "sample_seed": 42, "side": args.side,
              "stride": args.stride, "transform": args.transform, "batch": args.batch,
              "threshold": args.threshold, "tile_batch": args.tile_batch, "device": str(device), "threads": args.threads}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    checkpoint = args.output / "checkpoint.pt"
    start = perf_counter()
    training = MatchedCrops(train_docs, args.train_pairs, 42, args.side, args.stride, args.threshold, args.transform)
    validation = MatchedCrops(val_docs, args.val_pairs, 42, args.side, transform=args.transform)
    audit = {split: {name: quantiles(values) for name, values in dataset.audit.items()}
             for split, dataset in (("training", training), ("validation_oracle", validation))}
    training = bounded_cache(training, args.cache_mib)
    validation = bounded_cache(validation, args.cache_mib)
    model = PatchNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    epoch_start, history, best = 0, [], None
    if args.resume:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved["config"] != config:
            raise ValueError("Resume requires the same experiment configuration")
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best = saved["history"], saved["best"]
        epoch_start = len(history)
        if args.epochs < epoch_start:
            raise ValueError("Epoch budget is smaller than the saved run")

    for epoch in range(epoch_start + 1, args.epochs + 1):
        train_result = patch_pass(model, training, args.batch, device, optimizer, args.seed + epoch)
        val_result = patch_pass(model, validation, args.batch, device)
        history.append({"epoch": epoch, "training_online": train_result, "validation_oracle": val_result})
        if best is None or val_result["auroc"] > best["auroc"]:
            best = {"epoch": epoch, "auroc": val_result["auroc"],
                    "weights": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
        pending = checkpoint.with_suffix(".tmp")
        torch.save({"config": config, "weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "history": history, "best": best}, pending)
        pending.replace(checkpoint)
        write_history(args.output / "epochs.jsonl", history)
        print(f"epoch {epoch}: oracle AUROC {val_result['auroc']:.4f}", flush=True)

    write_history(args.output / "epochs.jsonl", history)
    model.load_state_dict(best["weights"])
    report = {"config": config, "parameters": sum(p.numel() for p in model.parameters()), "audit_quantiles": audit,
              "best_epoch": best["epoch"], "training": patch_pass(model, training, args.batch, device),
              "validation_oracle": history[best["epoch"] - 1]["validation_oracle"],
              "training_and_oracle_seconds": perf_counter() - start}
    (args.output / "oracle.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.shortcuts:
        report["shortcut_baseline"] = shortcut_probe(training, validation, args.batch)
        report["oracle_preprocessing_stress"] = {}
        for variant in ("gray", "jpeg"):
            altered = MatchedCrops(val_docs, args.val_pairs, 42, args.side, transform=variant)
            report["oracle_preprocessing_stress"][variant] = patch_pass(model, altered, args.batch, device)
    if args.tiled:
        tiling = {**config, "stride": args.tile_stride or args.stride}
        values, details = tiled_pass(model, val_docs, args.tile_pairs or args.val_pairs, 42, tiling, device)
        report["validation_tiled"] = {**details, **{name: metrics(values[:, :, i])
                                      for i, name in enumerate(("max", "top3", "mean"))}}
        report["validation_tiled"]["stride"] = tiling["stride"]
        if args.threshold is not None:
            train_values, _ = tiled_pass(model, train_docs, args.train_pairs, 42, tiling, device)
            aggregation = make_pipeline(StandardScaler(), LogisticRegression())
            aggregation.fit(train_values.reshape(-1, 3), np.tile([0, 1], len(train_values)))
            fitted = aggregation.decision_function(values.reshape(-1, 3))
            report["validation_tiled"]["trained_pooling"] = metrics(fitted)
    report["elapsed_seconds"] = perf_counter() - start
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "asset-root", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option, default in (("seed", 42), ("epochs", 6), ("threads", 4), ("train-pairs", 128), ("val-pairs", 228),
                            ("side", 224), ("stride", 112), ("batch", 8), ("tile-batch", 32), ("cache-mib", 160)):
        parser.add_argument("--" + option, type=int, default=default)
    parser.add_argument("--transform", choices=("rgb", "gray", "jpeg"), default="rgb")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--tile-pairs", type=int, default=0)
    parser.add_argument("--tile-stride", type=int, default=0)
    for option in ("resume", "shortcuts", "tiled"):
        parser.add_argument("--" + option, action="store_true")
    options = parser.parse_args()
    if min(options.epochs, options.threads, options.train_pairs, options.val_pairs, options.batch, options.tile_batch) < 1:
        parser.error("counts must be positive")
    if not 1 <= options.stride <= options.side or options.side < 32 or options.cache_mib < 0:
        parser.error("side must be at least 32 and stride must be positive and at most side")
    if not 0 <= options.tile_stride <= options.side or not 0 <= options.tile_pairs <= options.val_pairs:
        parser.error("tile stride and validation pair cap must fit the experiment settings")
    if options.threshold is not None and not 0 < options.threshold <= 1:
        parser.error("threshold must be a positive mask fraction")
    run(options)
