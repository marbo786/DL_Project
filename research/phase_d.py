"""Frozen document-level MIL and matched MobileNetV3 feature controls."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

from research.data import ManifestDocuments
from research.diagnose import select_pairs
from research.mil_bags import build_bags, make_encoder
from research.mil_heads import BagAttention, spatial_groups
from research.patch_data import check_source_splits
from research.patch_probe import metrics, quantiles
from research.phase_c import POOLS, localization, paired_intervals, pooling

EXPECTED_MANIFEST = "3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80"


def invoke(head, bag, device, grouped):
    features = bag.features.to(device)
    if grouped:
        groups = spatial_groups(bag.boxes, bag.size).to(device)
        return head.grouped(features, groups)
    return head(features)


def fit(pairs, dimension, seed, epochs, device, grouped=False, shuffled=False):
    torch.manual_seed(seed)
    head = BagAttention(dimension).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=.001)
    rng = random.Random(seed)
    flips = [rng.randrange(2) if shuffled else 0 for _ in pairs]
    curve = []
    for epoch in range(epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        epoch_loss = 0.
        for start in range(0, len(order), 8):
            opt.zero_grad(set_to_none=True)
            losses = []
            for index in order[start:start + 8]:
                pair = pairs[index]
                logits = torch.stack([invoke(head, bag, device, grouped)[0] for bag in pair])
                labels = logits.new_tensor([bag.label ^ flips[index] for bag in pair])
                losses.append(F.binary_cross_entropy_with_logits(logits, labels))
            loss = torch.stack(losses).mean()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(losses)
        curve.append({"epoch": epoch + 1, "loss": epoch_loss / len(pairs)})
    return head.eval(), curve


@torch.inference_mode()
def head_scores(head, pairs, device, grouped):
    document, evidence, timing = [], [], []
    for pair in pairs:
        scores, ranks, elapsed = [], [], []
        for bag in pair:
            start = perf_counter()
            score, weights = invoke(head, bag, device, grouped)
            scores.append(float(score))
            ranks.append(weights.cpu().numpy())
            elapsed.append(bag.seconds + perf_counter() - start)
        document.append(scores)
        evidence.append(ranks[1])
        timing.extend(elapsed)
    return np.asarray(document), evidence, timing


def score_features(pairs, kind):
    rows = []
    for pair in pairs:
        for bag in pair:
            count = np.log1p(len(bag.boxes))
            if kind == "count":
                rows.append([count])
            else:
                rows.append([*pooling(bag.logits), *np.quantile(bag.logits, [.75, .90, .95]), count])
    return np.asarray(rows)


def mask_targets(documents, count, bags):
    targets = []
    for pair, (_, edited) in zip(select_pairs(documents, count, 42), bags):
        _, mask, _ = documents.load(pair[1])
        overlaps = np.array([mask.crop(tuple(box)).histogram()[255] for box in edited.boxes])
        x0, y0, x1, y1 = edited.boxes.T
        areas = (np.minimum(x1, mask.width) - x0) * (np.minimum(y1, mask.height) - y0)
        targets.append((overlaps, areas, mask.histogram()[255]))
    return targets


def evaluate(train, val, bags, reference, evidence=None, targets=None, timings=None):
    result = metrics(val)
    result["balanced_accuracy"] = .5 * (result["recall"] + 1 - result["false_positive_rate"])
    result["auroc_ci95"] = paired_intervals(val)
    if reference is not None:
        result["delta_auroc"] = result["auroc"] - metrics(reference)["auroc"]
        result["delta_ci95"] = paired_intervals(val, reference)
    result["operating_points"] = {}
    for fpr in (.05, .10, .20):
        cutoff = np.quantile(train[:, 0], 1 - fpr)
        at_cutoff = metrics(val - cutoff)
        result["operating_points"][str(fpr)] = {"recall": at_cutoff["recall"], "fpr": at_cutoff["false_positive_rate"]}
    for name, population in (("tile_count", [len(pair[0].boxes) for pair in bags]),
                             ("height_px", [pair[0].size[1] for pair in bags])):
        values = np.asarray(population)
        median = np.median(values)
        groups = {"split": float(median)}
        for group, chosen in (("below", values < median), ("at_or_above", values >= median)):
            if chosen.any():
                subset = metrics(val[chosen])
                groups[group] = {"pairs": int(chosen.sum()), "auroc": subset["auroc"]}
        result[name] = groups
    for corpus in sorted({pair[0].corpus for pair in bags}):
        chosen = [pair[0].corpus == corpus for pair in bags]
        subset = metrics(val[chosen])
        result.setdefault("by_corpus", {})[corpus] = {"pairs": subset["pairs"], "auroc": subset["auroc"]}
    gaps = val[:, 1] - val[:, 0]
    result["source_score_delta_quantiles"] = quantiles(gaps)
    result["wrong_way_pairs"] = int((gaps < -1e-6).sum())
    if evidence is not None:
        rows = [localization(score, *target) for score, target in zip(evidence, targets)]
        result["localization"] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        result["localization"]["lift_at_1"] = result["localization"]["hit_at_1"] - result["localization"]["uniform_hit_at_1"]
    if timings is not None:
        result["median_ms"], result["p95_ms"] = (1000 * np.quantile(timings, [.5, .95])).tolist()
    return result


def source_errors(scores, source_rows, limit=5):
    failures = {}
    for method, values in scores.items():
        delta = values[:, 1] - values[:, 0]
        wrong = np.argsort(delta)
        orders = {
            "wrong_way_pairs": wrong[delta[wrong] < -1e-6][:limit],
            "high_scoring_authentic": np.argsort(-values[:, 0])[:limit],
            "low_scoring_edited": np.argsort(values[:, 1])[:limit],
        }
        failures[method] = {
            name: [{**source_rows[index], "genuine_logit": float(values[index, 0]),
                    "edited_logit": float(values[index, 1])} for index in order]
            for name, order in orders.items()
        }
    return failures


def run(args):
    started = perf_counter()
    digest = sha256(args.manifest.read_bytes()).hexdigest()
    if digest != EXPECTED_MANIFEST:
        raise ValueError("Manifest does not match the pinned development split")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = ManifestDocuments(args.manifest, args.asset_root, "train")
    validation = ManifestDocuments(args.manifest, args.asset_root, "validation")
    check_source_splits(training, validation)
    model, identity = make_encoder(args.encoder, args.checkpoint, device, 42)
    train_bags = build_bags(model, training, 128, identity, digest, args.cache, device)
    val_bags = build_bags(model, validation, args.val_pairs, identity, digest, args.cache, device)
    cache_wall_seconds = perf_counter() - started
    reference = np.load(args.reference)[:, :, 0] if args.reference is not None else None
    results = {}
    raw = {}
    targets = mask_targets(validation, args.val_pairs, val_bags)
    if args.encoder == "patchnet":
        train_fixed = np.asarray([[pooling(bag.logits) for bag in pair] for pair in train_bags])
        fixed_times = []
        val_fixed = []
        for pair in val_bags:
            row = []
            for bag in pair:
                start = perf_counter()
                row.append(pooling(bag.logits))
                fixed_times.append(bag.seconds + perf_counter() - start)
            val_fixed.append(row)
        val_fixed = np.asarray(val_fixed)
        if reference is None:
            reference = val_fixed[:, :, 0]
        for i, name in enumerate(POOLS):
            results[name] = evaluate(train_fixed[:, :, i], val_fixed[:, :, i], val_bags, reference,
                                     [bag.logits for _, bag in val_bags], targets, fixed_times)
            raw[name] = val_fixed[:, :, i]
    for name in (("count", "summary") if args.encoder == "patchnet" else ("count",)):
        train_x, val_x = (score_features(bags, name) for bags in (train_bags, val_bags))
        classifier = make_pipeline(StandardScaler(), LogisticRegression())
        classifier.fit(train_x, np.tile([0, 1], len(train_bags)))
        train_scores = classifier.decision_function(train_x).reshape(-1, 2)
        val_scores = classifier.decision_function(val_x).reshape(-1, 2)
        results[name] = evaluate(train_scores, val_scores, val_bags, reference)
        raw[name] = val_scores
    methods = ("attention", "spatial", "shuffled_labels") if args.encoder == "patchnet" else ("attention", "spatial")
    for name in methods:
        for seed in args.seeds:
            start = perf_counter()
            head, curve = fit(train_bags, identity["feature_dim"], seed, args.epochs, device,
                              name == "spatial", name == "shuffled_labels")
            seconds = perf_counter() - start
            train_scores, _, _ = head_scores(head, train_bags, device, name == "spatial")
            val_scores, evidence, timing = head_scores(head, val_bags, device, name == "spatial")
            key = f"{name}-{seed}"
            result = evaluate(train_scores, val_scores, val_bags, reference, evidence, targets, timing)
            result.update(train_auroc=metrics(train_scores)["auroc"], training_seconds=seconds, curve=curve,
                          head_parameters=sum(parameter.numel() for parameter in head.parameters()),
                          attention_effective_tiles=quantiles([1 / np.square(weights).sum() for weights in evidence]))
            results[key] = result
            raw[key] = val_scores
            torch.save(head.state_dict(), args.output / f"{key}.pt")
            print(f"{key}: AUROC {result['auroc']:.4f}, attention hit@3 {result['localization']['hit_at_3']:.4f}", flush=True)
    summary = {**identity, "manifest_sha256": digest, "train_pairs": len(train_bags),
               "validation_pairs": len(val_bags), "seeds": args.seeds, "epochs": args.epochs, "device": str(device),
               "cache_seconds": cache_wall_seconds, "elapsed_seconds": perf_counter() - started,
               "cache_bytes": sum(bag.path.stat().st_size for pairs in (train_bags, val_bags) for pair in pairs for bag in pair),
               "validation_tile_counts": quantiles([len(bag.boxes) for pair in val_bags for bag in pair]),
               "results": results}
    np.savez(args.output / "scores.npz", **raw)
    sources = [{"source": pair[0].source, "corpus": pair[0].corpus, "height_px": int(pair[0].size[1]),
                "tile_count": len(pair[0].boxes)} for pair in val_bags]
    (args.output / "source_errors.json").write_text(json.dumps(source_errors(raw, sources), indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "asset-root", "checkpoint", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Phase C paired score array on the identical 64-pair screen")
    parser.add_argument("--encoder", choices=("patchnet", "mobile-random", "mobile-pretrained"), default="patchnet")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--val-pairs", type=int, choices=(64, 228), default=64)
    args = parser.parse_args()
    if min(args.epochs, args.threads) < 1 or (args.reference is not None and args.val_pairs != 64):
        parser.error("positive budgets required; Phase C reference is only valid for the 64-pair screen")
    run(args)
