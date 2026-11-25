"""Repeat paired, end-to-end development timings on a fixed receipt panel."""
import argparse
import csv
from itertools import cycle, islice
from pathlib import Path

import numpy as np
import torch

from research.data import ManifestDocuments
from research.manifest import file_sha256
from research.receipt.ablations import read_json, write_json, write_rows
from research.receipt.efficiency import ranked_paths
from research.receipt.model import load_detector
from research.receipt.pareto import timed_score


def run(manifest, assets, artifacts, output):
    previous = artifacts / "receipt-pareto-v1"
    lock = read_json(previous / "lock.json")
    if file_sha256(manifest) != lock["manifest_sha256"]:
        raise ValueError("Development manifest changed")
    roles = {"residual-fp32": artifacts / "receipt-study-v1/residual-42",
             "student": artifacts / "receipt-distill-v1/student",
             "distilled": artifacts / "receipt-distill-v1/distilled"}
    models = {name: load_detector(folder) for name, folder in roles.items()}
    with (previous / "scores-and-timings.csv").open(newline="") as stream:
        saved = {(r["model"], r["sample_id"]): float(r["logit"]) for r in csv.DictReader(stream)}
    docs = ranked_paths(ManifestDocuments(manifest, assets, "validation"), 40)
    variants = [(name, 2) for name in roles] + [("residual-fp32", 4)]
    output.mkdir(parents=True)
    write_json(output / "lock.json", {"parent": file_sha256(previous / "lock.json"), "passes": 3,
               "variants": variants, "ids": [r["sample_id"] for r in docs],
               "scope": "development timing sensitivity; common saved policy and logits"})
    torch.set_num_interop_threads(1)
    for name, threads in variants:
        torch.set_num_threads(threads)
        for row in docs[:4]:
            timed_score(models[name], lock["policy"][name], assets / row["image_path"])
    rows = []
    for repeat in range(3):
        for index, doc in enumerate(docs):
            start = (index + repeat) % len(variants)
            for name, threads in islice(cycle(variants), start, start + len(variants)):
                torch.set_num_threads(threads)
                score, ms = timed_score(models[name], lock["policy"][name], assets / doc["image_path"])
                if abs(score - saved[name, doc["sample_id"]]) > 1e-4:
                    raise ValueError("Logit differs from the original benchmark")
                rows.append({"pass": repeat + 1, "sample": doc["sample_id"], "group": doc["source_group"],
                             "variant": f"{name}-{threads}", "ms": ms})
    write_rows(output / "timings.csv", rows)
    summary = {}
    timings = {}
    for name, threads in variants:
        key = f"{name}-{threads}"
        matrix = np.array([r["ms"] for r in rows if r["variant"] == key]).reshape(3, len(docs))
        timings[key] = matrix
        summary[key] = {"median_ms_by_pass": np.median(matrix, axis=1).tolist(),
                        "p95_ms_by_pass": np.quantile(matrix, .95, axis=1).tolist()}
    diff = np.median(timings["distilled-2"] - timings["student-2"], axis=0)
    groups = np.array([r["source_group"] for r in docs])
    members = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    rng = np.random.default_rng(42)
    draws = [np.median(diff[np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])])
             for _ in range(2000)]
    summary["distilled_minus_hard_label_ms"] = {"paired_median": float(np.median(diff)),
             "source_bootstrap_95": np.quantile(draws, [.025,.975]).tolist()}
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "assets", "artifacts", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.assets, args.artifacts, args.output)
