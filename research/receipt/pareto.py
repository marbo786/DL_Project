"""Benchmark frozen receipt detectors at a common development review budget."""

import argparse
import csv
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import torch

from research.data import ManifestDocuments
from research.manifest import file_sha256, resolve_image
from research.receipt.demo import FrozenReceiptDemo, LOCK_PATH, POLICY_VERSION, RUN_PATH, route_receipt
from research.receipt.efficiency import load_image, quantize, ranked_paths
from research.receipt.metrics import frozen_thresholds
from research.receipt.model import inference_tiles, load_detector, predict
from research.receipt.report import plt, save


BUDGET = .10
TILE_BATCH = 64
MODEL_LABELS = {
    "residual-fp32": "Residual FP32", "residual-int8": "Residual INT8",
    "student": "PatchNet hard labels", "distilled": "PatchNet distilled",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def score_at_budget(rows, k):
    ranked = sorted(rows, key=lambda row: (-row["logit"], row["sample_id"]))
    positives = sum(row["edited"] for row in rows)
    found = sum(row["edited"] for row in ranked[:k])
    boundary = ranked[k - 1]["logit"]
    return {"edits_found": found, "edits_missed": positives - found, "false_flags": k - found,
            "recall": found / positives, "fpr": (k - found) / (len(rows) - positives),
            "precision": found / k, "boundary_score_ties": sum(row["logit"] == boundary for row in ranked)}


def dominated_by(results, cost):
    return {
        name: [peer for peer, other in results.items()
               if other[cost] <= result[cost] and other["recall"] >= result["recall"]
               and (other[cost] < result[cost] or other["recall"] > result["recall"])]
        for name, result in results.items()
    }


def timed_score(model, policy, path):
    start = perf_counter()
    logit, box = predict(model, load_image(path), TILE_BATCH)
    if not isfinite(logit):
        raise ValueError("Receipt detector emitted a non-finite logit")
    probability = float(expit(policy["slope"] * logit + policy["intercept"]))
    outcome, advice = route_receipt(logit, probability, policy["review_cutoff"])
    json.dumps({"logit": logit, "probability": probability, "confidence": max(probability, 1 - probability),
                "decision": outcome, "advice": advice, "tile": box})
    return logit, (perf_counter() - start) * 1000


def plot_frontier(results, dominance, cost, xlabel, output):
    figure, axis = plt.subplots(figsize=(7.5, 4.25))
    figure.patch.set_facecolor("#fbfcfe")
    axis.set_facecolor("#fbfcfe")
    axis.set_xscale("log")
    axis.margins(x=.20)
    axis.set(axisbelow=True, xlabel=xlabel, ylabel="Edited receipts found (%)", ylim=(0, 25),
             title="Development recall at 40 reviews per 400 receipts")
    axis.grid(axis="y", color="#dbe3eb")
    offsets = {"residual-fp32": (-10, 30), "residual-int8": (-10, -21),
               "student": (10, 12), "distilled": (10, -20)}
    for name, item in results.items():
        point = (item[cost], 100 * item["recall"])
        color = "#576779" if dominance[name] else "#087f73"
        dx, dy = offsets[name]
        axis.scatter(*point, s=75, edgecolor=color, facecolor="white" if dominance[name] else color, zorder=3)
        axis.annotate(MODEL_LABELS[name], point, xytext=(dx, dy), textcoords="offset points",
                      ha="right" if dx < 0 else "left", color="#14283f", fontsize=9,
                      arrowprops={"arrowstyle": "-", "color": color, "shrinkA": 0, "shrinkB": 5})
    figure.text(.5, .01, "Filled markers indicate the measured Pareto frontier", ha="center", color="#465969")
    figure.subplots_adjust(bottom=.18)
    save(figure, output)


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = args.root.resolve()
    frozen = FrozenReceiptDemo()
    manifest_digest = file_sha256(args.manifest)
    lock = read_json(LOCK_PATH)
    if manifest_digest != lock["manifest_sha256"]:
        raise ValueError("Use the manifest from the frozen receipt study")
    calibration = ManifestDocuments(args.manifest, root, "calibration")
    development = ManifestDocuments(args.manifest, root, "validation")
    calibration_groups = {row["source_group"] for row in calibration.rows}
    if calibration_groups.intersection(row["source_group"] for row in development.rows):
        raise ValueError("Calibration and development sources overlap")
    paths = {"residual-fp32": RUN_PATH, "student": args.students / "student", "distilled": args.students / "distilled"}
    models = {"residual-fp32": frozen.model}
    policies = {}
    fingerprints = {}
    for name, folder in paths.items():
        fit = read_json(folder / "fit.json")
        metrics = read_json(folder / "metrics.json")
        if fit["training_manifest_sha256"] != manifest_digest or fit["seed"] != lock["seed"]:
            raise ValueError(f"Incompatible training provenance for {name}")
        calibration_sha = file_sha256(folder / "calibration.csv")
        if calibration_sha != metrics["calibration_sha256"]:
            raise ValueError(f"Saved calibrator and calibration scores disagree for {name}")
        if name != "residual-fp32":
            models[name] = load_detector(folder)
        policies[name] = {**metrics["calibration"], "review_cutoff": metrics["operating_points"]["0.05"]["logit_threshold"]}
        fingerprints[name] = {"checkpoint_sha256": fit["checkpoint_sha256"],
                              "bytes": (folder / "model.pt").stat().st_size,
                              "calibration_sha256": calibration_sha, "fit_sha256": file_sha256(folder / "fit.json")}
    if (policies["residual-fp32"]["review_cutoff"] != frozen.review_cutoff
        or policies["residual-fp32"]["slope"] != frozen.slope
        or policies["residual-fp32"]["intercept"] != frozen.intercept):
        raise ValueError("FP32 timing policy differs from the frozen demo")

    args.output.mkdir(parents=True)
    observers = ranked_paths(calibration, 16)
    batches = (inference_tiles(load_image(resolve_image(root, row["image_path"])), "residual")[1][::4]
               for row in observers)
    int8, operators = quantize(frozen.model, batches, "onednn")
    models["residual-int8"] = int8
    int8_path = args.output / "residual-int8.pt"
    torch.save(int8.state_dict(), int8_path)
    calibrated = []
    for row in calibration.rows:
        logit, _ = predict(int8, load_image(resolve_image(root, row["image_path"])), TILE_BATCH)
        calibrated.append({"sample_id": row["sample_id"], "edited": row["label"] == "manipulated", "logit": logit})
    calib_path = args.output / "int8-calibration.json"
    write_json(calib_path, calibrated)
    logits = np.array([row["logit"] for row in calibrated])
    labels = np.array([row["edited"] for row in calibrated])
    calibrator = LogisticRegression(C=1.).fit(logits[:, None], labels)
    policies["residual-int8"] = {"slope": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0]),
                                 "prevalence": float(labels.mean()), "review_cutoff": frozen_thresholds(labels, logits)["0.05"]}
    fingerprints["residual-int8"] = {"checkpoint_sha256": file_sha256(int8_path), "bytes": int8_path.stat().st_size,
                                    "calibration_sha256": file_sha256(calib_path),
                                    "parent_checkpoint_sha256": frozen.checkpoint_sha256}

    schedule = ranked_paths(development, len(development.rows))
    names = tuple(MODEL_LABELS)
    budget_count = ceil(BUDGET * len(schedule))
    benchmark_lock = args.output / "lock.json"
    write_json(benchmark_lock, {
        "run_id": args.output.name, "manifest_sha256": manifest_digest, "receipt_lock_sha256": file_sha256(LOCK_PATH),
        "scope": "development only", "document_order": [row["sample_id"] for row in schedule],
        "warmup_documents_per_model": 8, "passes": 1, "model_order": list(names), "order_rule": "rotate by document index",
        "document_batch": 1, "tile_batch": TILE_BATCH, "torch_threads": 2, "torch_interop_threads": 1,
        "review_budget_fraction": BUDGET, "review_budget_documents": budget_count,
        "ranking": "raw logit descending, sample ID ascending on ties", "response_policy": POLICY_VERSION,
        "calibration_observer_documents": [row["sample_id"] for row in observers],
        "quantized_engine": "onednn", "quantized_modules": operators, "policy": policies, "checkpoints": fingerprints,
        "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "processor": args.cpu_label or platform.processor(),
        "timing_boundary": "file decode through JSON response, without network or UI",
    })
    for name in names:
        for row in schedule[:8]:
            timed_score(models[name], policies[name], resolve_image(root, row["image_path"]))

    rows = {name: [] for name in names}
    timings_path = args.output / "scores-and-timings.csv"
    with timings_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "sample_id", "edited", "logit", "milliseconds"))
        writer.writeheader()
        for index, row in enumerate(schedule):
            path = resolve_image(root, row["image_path"])
            start_at = index % len(names)
            for name in names[start_at:] + names[:start_at]:
                logit, milliseconds = timed_score(models[name], policies[name], path)
                measured = {"model": name, "sample_id": row["sample_id"], "edited": row["label"] == "manipulated",
                            "logit": logit, "milliseconds": milliseconds}
                rows[name].append(measured)
                writer.writerow(measured)
            if (index + 1) % 100 == 0:
                stream.flush()
                print(f"Benchmarked {index + 1}/{len(schedule)} common documents", flush=True)

    results = {}
    for name, measured in rows.items():
        median, p95 = np.quantile([item["milliseconds"] for item in measured], (.5, .95))
        results[name] = {"checkpoint_mib": fingerprints[name]["bytes"] / (1024 ** 2),
                         "median_ms": float(median), "p95_ms": float(p95), **score_at_budget(measured, budget_count)}
    dominance = {cost: dominated_by(results, cost) for cost in ("checkpoint_mib", "p95_ms")}
    write_json(args.output / "summary.json", {
        "run_id": args.output.name, "lock_sha256": file_sha256(benchmark_lock),
        "timings_sha256": file_sha256(timings_path), "documents": len(schedule),
        "source_groups": len({row["source_group"] for row in schedule}), "review_budget_documents": budget_count,
        "target_median_ms": 100, "results": results, "dominated_by": dominance,
    })
    plot_frontier(results, dominance["checkpoint_mib"], "checkpoint_mib", "Checkpoint size (MiB, log scale)",
                  args.output / "recall-vs-size.png")
    plot_frontier(results, dominance["p95_ms"], "p95_ms", "End-to-end p95 latency (ms, log scale)",
                  args.output / "recall-vs-p95.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("manifest", "root", "students", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--cpu-label")
    run(parser.parse_args())
