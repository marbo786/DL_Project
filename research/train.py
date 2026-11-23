"""Train, compare, calibrate, and export ForgeryLens CPU models."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from api.network import ReceiptNet
from research.data import SYNTHETIC_SPLITS, ManifestDocuments, Specimens, features, perturb, specimen
from research.manifest import audit

SEEDS = (17, 42, 71)
ABLATIONS = {
    "grayscale_only": {"residual": False},
    "average_pooling": {"topk": False},
    "no_global_context": {"context": False},
    "classification_only": {},
}
STRESSES = ("blur", "low_light", "jpeg", "occlusion")


def cache(dataset):
    images, masks, labels = zip(*(dataset[index] for index in range(len(dataset))))
    if len(set(labels)) != 2:
        raise ValueError("each dataset partition requires genuine and manipulated examples")
    return TensorDataset(torch.stack(images), torch.stack(masks), torch.tensor(labels))


@torch.inference_mode()
def collect(model, dataset):
    scores, predictions, targets, labels = [], [], [], []
    for image, mask, label in DataLoader(dataset, batch_size=32):
        score, grid = model(image)
        scores.append(score)
        predictions.append(grid[:, 0])
        targets.append(F.max_pool2d(mask, 4, stride=4)[:, 0])
        labels.append(label)
    return tuple(torch.cat(items).numpy() for items in (scores, predictions, targets, labels))


def probabilities(logits, temperature=1.0, bias=0.0):
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature + bias, -60.0, 60.0)))


def calibration(scores, labels):
    candidates = []
    target = labels.mean()
    for temperature in np.geomspace(.05, 8.0, 72):
        scaled = scores / temperature
        low, high = -float(scaled.max()) - 30, -float(scaled.min()) + 30
        for _ in range(24):
            bias = (low + high) / 2
            if probabilities(scores, temperature, bias).mean() < target:
                low = bias
            else:
                high = bias
        shifted = scaled + bias
        loss = np.mean(np.logaddexp(0, shifted) - labels * shifted)
        candidates.append((float(loss), float(temperature), float(bias)))
    _, temperature, bias = min(candidates)
    return temperature, bias

def reliability(probability, actual, bins=10):
    bins_out = []
    index = np.minimum((probability * bins).astype(int), bins - 1)
    for bin_id in range(bins):
        there = index == bin_id
        if there.any():
            bins_out.append({
                "bin": bin_id, "count": int(there.sum()),
                "mean_probability": float(probability[there].mean()),
                "observed_fraction": float(actual[there].mean()),
            })
    ece = sum(row["count"] * abs(row["mean_probability"] - row["observed_fraction"]) for row in bins_out) / len(actual)
    return ece, bins_out


def risk(probability, labels, confidence):
    accepted = np.maximum(probability, 1 - probability) >= confidence
    return {
        "confidence_threshold": confidence, "coverage": float(accepted.mean()),
        "error_rate": float(np.mean((probability[accepted] >= 0.5) != labels[accepted])) if accepted.any() else None,
    }


def measures(outputs, temperature=1.0, cutoff=.5, localized=True, bias=0.0):
    scores, grid, masks, labels = outputs
    probability = probabilities(scores, temperature, bias)
    prediction = probability >= .5
    ece, bins = reliability(probability, labels)
    result = {
        "examples": len(labels),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "manipulated_recall": float(recall_score(labels, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pr_auc": float(average_precision_score(labels, probability)),
        "ece": ece, "reliability": bins,
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
        "selective": risk(probability, labels, cutoff),
    }
    edited = labels == 1
    if localized and edited.any():
        region = grid[edited] >= 0
        target = masks[edited] > 0
        intersection = (region & target).sum(axis=(1, 2))
        union = (region | target).sum(axis=(1, 2))
        result["mean_mask_iou"] = float(np.mean(intersection / np.maximum(union, 1)))
    return result


def summary_metrics(outputs, **options):
    metrics = measures(outputs, **options)
    return {key: value for key, value in metrics.items() if key not in ("reliability", "confusion_matrix", "selective")}


def train(variant, config, train_data, val_data, epochs, seed, teacher=None, observer=None, *,
          class_weight=1., map_weight=None, learning_rate=.003, positive_weight=18., batch_size=32):
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    network = ReceiptNet(**config)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=learning_rate * (2 / 15))
    checkpoint, best, curve = None, -float("inf"), []
    if map_weight is None:
        map_weight = 0 if variant in ("baseline", "classification_only") else .65
    foreground_weight = torch.tensor(positive_weight)
    for epoch in range(epochs):
        network.train()
        total = 0.0
        for image, mask, labels in loader:
            score, grid = network(image)
            target = F.max_pool2d(mask, 4, stride=4)
            class_loss = F.binary_cross_entropy_with_logits(score, labels)
            loss = class_weight * class_loss if class_weight else None
            pixel_loss = None
            if map_weight or observer is not None:
                pixel_loss = F.binary_cross_entropy_with_logits(grid, target, pos_weight=foreground_weight)
            if map_weight:
                weighted_map_loss = map_weight * pixel_loss
                loss = weighted_map_loss if loss is None else loss + weighted_map_loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_score, teacher_map = teacher(image)
                loss = loss + .25 * F.smooth_l1_loss(score, teacher_score) + .15 * F.smooth_l1_loss(grid, teacher_map)
            if observer is not None and class_weight and map_weight:
                observer.objectives(network, class_loss, pixel_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if observer is not None:
                observer.batch(network, class_loss, pixel_loss)
            optimizer.step()
            total += loss.item() * len(labels)
        scheduler.step()
        network.eval()
        outputs = collect(network, val_data)
        val = summary_metrics(outputs, localized=bool(map_weight))
        rank = val["roc_auc"] + .25 * val.get("mean_mask_iou", 0) if class_weight else val["mean_mask_iou"]
        if rank > best:
            best = rank
            checkpoint = {key: value.clone() for key, value in network.state_dict().items()}
        curve.append({"epoch": epoch + 1, "train_loss": total / len(train_data), **val})
        if observer is not None:
            observer.epoch(epoch + 1, outputs, curve[-1])
    network.load_state_dict(checkpoint)
    return network, curve


def timing(model, image):
    sample = image[:1].contiguous()
    with torch.inference_mode():
        for _ in range(12):
            model(sample)
        elapsed = []
        for _ in range(60):
            start = perf_counter()
            model(sample)
            elapsed.append((perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(elapsed)), "p95_ms": float(np.percentile(elapsed, 95))}


def save_checkpoint(path, net, config, scope, temperature, cutoff, specimens, fmt="fp32", bias=0.0):
    state = net.state_dict()
    if fmt == "fp16_storage":
        state = {key: value.half() if value.is_floating_point() else value for key, value in state.items()}
    torch.save({
        "weights": state, "config": config, "temperature": temperature, "bias": bias, "confidence_threshold": cutoff,
        "version": "forgerylens-2-" + scope, "scope": scope, "specimen_digests": specimens, "format": fmt,
    }, path)


def artifact(output, name, payload):
    (output / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def analyze_failures(outputs, temperature, bias):
    score, grid, mask, labels = outputs
    probability = probabilities(score, temperature, bias)
    hard = np.argsort(np.where(labels, probability, 1 - probability))
    failures = []
    for index in hard:
        expected = int(labels[index])
        predicted = int(probability[index] >= 0.5)
        if expected == predicted:
            continue
        region, target = grid[index] >= 0, mask[index] > 0
        overlap = int((region & target).sum()) / max(int((region | target).sum()), 1)
        failures.append({
            "shifted_test_index": int(index), "expected": expected, "predicted": predicted,
            "manipulation_probability": float(probability[index]), "mask_iou": overlap,
        })
        if len(failures) == 5:
            break
    return failures


def run(args):
    torch.set_num_threads(args.threads)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    scope = "synthetic" if args.manifest is None else "manifest"
    if args.manifest:
        if not args.asset_root:
            raise ValueError("--asset-root is required with --manifest")
        print("Auditing manifest and image assets", flush=True)
        errors = audit(args.manifest, args.asset_root)
        if errors:
            raise ValueError("\n".join(errors[:20]))
    dataset = (
        (lambda split, stress=None: ManifestDocuments(args.manifest, args.asset_root, split, stress))
        if args.manifest else
        (lambda split, stress=None: Specimens(
            SYNTHETIC_SPLITS[split][0], args.train_sources if split == "train" else SYNTHETIC_SPLITS[split][1],
            shift=split == "shifted_test", stress=stress,
        ))
    )
    print("Preparing training and validation tensors", flush=True)
    train_set = dataset("train")
    train_data = cache(train_set)
    val_data = cache(dataset("validation"))
    base_config = {"width": 8}
    variants = {"baseline": {"local": False, "residual": False}, "local": {}, **ABLATIONS}
    trials, checkpoints = {}, {}
    for name, options in variants.items():
        config = {**base_config, **options}
        trials[name] = []
        for seed in SEEDS:
            model, curve = train(name, config, train_data, val_data, args.epochs, seed)
            metrics = summary_metrics(collect(model, val_data), localized=name not in ("baseline", "classification_only"))
            trials[name].append({"seed": seed, "validation": metrics, "curve": curve})
            rank = metrics["roc_auc"] + .25 * metrics.get("mean_mask_iou", 0)
            if name not in checkpoints or rank > checkpoints[name][0]:
                checkpoints[name] = (rank, model, config, seed)
            print(name, seed, f"validation ROC-AUC {metrics['roc_auc']:.3f}", flush=True)
    contenders = ("local", "grayscale_only", "average_pooling", "no_global_context")
    teacher_name = max(contenders, key=lambda name: checkpoints[name][0])
    _, teacher, config, seed = checkpoints[teacher_name]
    student_config = {**config, "width": 4}
    student, student_curve = train("student", student_config, train_data, val_data, args.epochs, SEEDS[0], teacher)
    fp16 = ReceiptNet(**config).eval()
    fp16.load_state_dict({key: value.half() if value.is_floating_point() else value for key, value in teacher.state_dict().items()})

    compressed = {"baseline": (*checkpoints["baseline"][1:3], "fp32"),
                  "local": (teacher, config, "fp32"),
                  "fp16_storage": (fp16, config, "fp16_storage"),
                  "distilled_student": (student, student_config, "fp32")}
    model_rows = {}
    specimen_images = {}
    if scope == "synthetic" and args.web_assets:
        args.web_assets.mkdir(parents=True, exist_ok=True)
        demo_source = SYNTHETIC_SPLITS["test"][0]
        clean, _, _ = specimen(demo_source, 0)
        edited, demo_mask, _ = specimen(demo_source, 1)
        left, top, right, bottom = demo_mask.getbbox()
        w, h = edited.size
        region = {"x": left / w, "y": top / h, "width": (right - left) / w, "height": (bottom - top) / h}
        artifact(args.web_assets, "specimen", {"run_id": "synthetic-specimens", "edit_region": region})
        for name, image in {"original": clean, "edited": edited, "blurred": perturb(clean, "blur")}.items():
            image.save(args.web_assets / f"{name}.png")
            specimen_images[sha256(features(image).numpy().tobytes()).hexdigest()] = name != "blurred"

    for name, (model, architecture, fmt) in compressed.items():
        val = collect(model, val_data)
        temperature, bias = calibration(val[0], val[3])
        probability = probabilities(val[0], temperature, bias)
        candidates = [risk(probability, val[3], float(level)) for level in np.arange(.50, .97, .025)]
        feasible = [row for row in candidates if row["coverage"] > 0 and row["error_rate"] <= .10]
        cutoff = max(feasible, key=lambda row: row["coverage"])["confidence_threshold"] if feasible else .99
        dest = output / f"{name}.pt"
        save_checkpoint(dest, model, architecture, scope, temperature, cutoff, specimen_images, fmt, bias=bias)
        model_rows[name] = {
            "params": sum(param.numel() for param in model.parameters()),
            "checkpoint_bytes": dest.stat().st_size, "timing": timing(model, val_data.tensors[0]),
            "validation": summary_metrics(val, temperature=temperature, localized=name != "baseline", bias=bias),
            "temperature": temperature, "bias": bias, "confidence_threshold": cutoff,
        }

    # Freeze the validation-selected model and thresholds before reading either test split.
    variant = max(("local", "distilled_student"), key=lambda name: (
        model_rows[name]["validation"]["macro_f1"], model_rows[name]["validation"]["mean_mask_iou"],
    ))
    selected, architecture, fmt = compressed[variant]
    temp, cutoff, bias = (model_rows[variant][key] for key in ("temperature", "confidence_threshold", "bias"))
    save_checkpoint(output / "model.pt", selected, architecture, scope, temp, cutoff, specimen_images, fmt, bias=bias)
    print("Evaluating the frozen models on test and shifted test", flush=True)
    test_data = cache(dataset("test"))
    shifted_data = cache(dataset("shifted_test"))
    for name, (model, _, _) in compressed.items():
        row = model_rows[name]
        row["test"] = summary_metrics(collect(model, test_data), temperature=row["temperature"], localized=name != "baseline", bias=row["bias"])
        row["shifted_test"] = summary_metrics(collect(model, shifted_data), temperature=row["temperature"], localized=name != "baseline", bias=row["bias"])
    for name, row in model_rows.items():
        row["pareto_dominated"] = any(
            peer["timing"]["median_ms"] <= row["timing"]["median_ms"]
            and peer["validation"]["macro_f1"] >= row["validation"]["macro_f1"]
            and (peer["timing"]["median_ms"] < row["timing"]["median_ms"]
                 or peer["validation"]["macro_f1"] > row["validation"]["macro_f1"])
            for other, peer in model_rows.items() if other != name
        )

    clean = collect(selected, test_data)
    shifted = collect(selected, shifted_data)
    train_sources = args.train_sources
    if args.manifest:
        train_sources = len({row["source_document_id"] for row in train_set.rows})
    run_id = f"{scope}-cpu-{args.epochs}ep-{train_sources}src-v2"
    meta = {"run_id": run_id, "scope": scope, "epochs": args.epochs, "seeds": list(SEEDS), "selected_model": variant}
    if args.manifest:
        meta["manifest_sha256"] = sha256(args.manifest.read_bytes()).hexdigest()
    artifact(output, "summary", {**meta, "training_examples": len(train_data), "validation_examples": len(val_data),
                                "models": model_rows, "teacher_architecture": teacher_name, "local_selected_seed": seed, "student_curve": student_curve})
    comparison = {}
    for name, variant_trials in trials.items():
        keys = variant_trials[0]["validation"]
        comparison[name] = {key: {"mean": float(np.mean([trial["validation"][key] for trial in variant_trials])),
                                  "std": float(np.std([trial["validation"][key] for trial in variant_trials], ddof=1))}
                            for key in keys if key != "examples"}
    artifact(output, "ablations", {**meta, "validation_aggregate": comparison, "runs": trials})
    print("Evaluating blur, low light, JPEG compression, and occlusion", flush=True)
    artifact(output, "robustness", {**meta, "clean": measures(clean, temp, cutoff, bias=bias),
        "stress": {stress: measures(collect(selected, cache(dataset("test", stress))), temp, cutoff, bias=bias) for stress in STRESSES},
        "shifted": measures(shifted, temp, cutoff, bias=bias), "shifted_failure_cases": analyze_failures(shifted, temp, bias)})
    validation = collect(selected, val_data)
    artifact(output, "calibration", {**meta, "temperature": temp, "bias": bias, "confidence_threshold": cutoff,
        "coverage_curve": [risk(probabilities(validation[0], temp, bias), validation[3], float(level)) for level in np.linspace(.5, .99, 25)],
        "test_before": measures(clean), "test_after": measures(clean, temp, cutoff, bias=bias),
        "shifted_after": measures(shifted, temp, cutoff, bias=bias)})
    print(json.dumps({"selected": variant, "scope": scope, "test": model_rows[variant]["test"],
                      "shifted": model_rows[variant]["shifted_test"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("release"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--web-assets", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-sources", type=int, default=112)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 2 <= args.epochs <= 100 or not 32 <= args.train_sources < 9000 or args.threads < 1:
        parser.error("use 2–100 epochs, 32–8999 train sources, and at least one thread")
    run(args)
