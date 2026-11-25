"""Run the fixed MobileNet receipt matrix on the historical development protocol."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from research.data import ManifestDocuments
from research.manifest import file_sha256
from research.receipt.metrics import evaluate
from research.receipt.model import CONDITIONS, MODES, score, train, training_data
from research.receipt.protocol import verify_groups


def run_stress(base, checkpoint, folder, sources):
    folder.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        destination = folder / f"{condition}.csv"
        report = folder / f"{condition}.json"
        if not destination.exists():
            temporary = destination.with_suffix(".in-progress.csv")
            score(SimpleNamespace(**base, checkpoint=checkpoint, split="validation", output=temporary,
                                  condition=condition, limit_sources=sources))
            temporary.rename(destination)
        if not report.exists():
            evaluate(checkpoint / "calibration.csv", destination, report)
        print(f"Measured stress condition: {condition}", flush=True)


def run(args):
    torch.set_num_threads(args.threads)
    verify_groups(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    base = dict(manifest=args.manifest, root=args.root, weights=args.weights, epochs=args.epochs,
                batch_size=args.batch_size, threads=args.threads)
    train_documents = ManifestDocuments(args.manifest, args.root, "train")
    runs = [(mode, seed, 1.) for seed in args.seeds for mode in args.modes]
    if args.curve:
        runs += [("full", args.seeds[0], fraction) for fraction in (.25, .50)]
    prepared = {}
    stats = defaultdict(list)
    for mode, seed, fraction in runs:
        name = f"{mode}-{seed}" + (f"-fraction-{fraction:g}" if fraction < 1 else "")
        folder = args.output / name
        config = SimpleNamespace(**base, mode=mode, seed=seed, fraction=fraction, output=folder)
        print(name, flush=True)
        if folder.exists():
            recorded = json.loads((folder / "fit.json").read_text(encoding="utf-8"))
            expected = {"mode": mode, "seed": seed, "fraction": fraction, "batch_size": args.batch_size,
                        "training_manifest_sha256": file_sha256(args.manifest),
                        "initial_weights_sha256": file_sha256(args.weights)}
            if any(recorded[key] != value for key, value in expected.items()) or len(recorded["losses"]) != args.epochs:
                raise ValueError(f"Run configuration changed for {name}")
        else:
            key = ("whole" if mode == "whole" else "patch", fraction)
            if key not in prepared:
                prepared[key] = training_data(train_documents, fraction, mode)
            train(config, prepared[key])
        for split in ("calibration", "validation"):
            destination = folder / f"{split}.csv"
            if not destination.exists():
                temporary = destination.with_suffix(".in-progress.csv")
                score(SimpleNamespace(**base, checkpoint=folder, split=split, output=temporary,
                                      condition="clean", limit_sources=None))
                temporary.rename(destination)
        result = folder / "metrics.json"
        if not result.exists():
            evaluate(folder / "calibration.csv", folder / "validation.csv", result)
        report = json.loads(result.read_text(encoding="utf-8"))
        stats[mode if fraction == 1 else f"full-{fraction:g}"].append(
            {"seed": seed, "auroc": report["auroc"], "ap": report["ap"], "run": name})
        print(f"{name} development AUROC={report['auroc']:.4f} AP={report['ap']:.4f}", flush=True)

    summary = {}
    for mode, entries in stats.items():
        values = np.array([[entry["auroc"], entry["ap"]] for entry in entries])
        summary[mode] = {"seeds": entries, "mean_auroc": float(values[:, 0].mean()),
                         "mean_ap": float(values[:, 1].mean()),
                         "seed_sd_auroc": float(values[:, 0].std(ddof=1)) if len(entries) > 1 else None}
    with (args.output / "matrix.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    reference_mode = "full"
    if reference_mode in args.modes:
        for seed in args.seeds:
            reference = args.output / f"{reference_mode}-{seed}" / "validation.csv"
            for mode in sorted(set(args.modes) - {reference_mode}):
                folder = args.output / f"{mode}-{seed}"
                result = folder / "paired-vs-full.json"
                if not result.exists():
                    evaluate(folder / "calibration.csv", folder / "validation.csv", result, compare=reference)
    if args.stress_sources:
        selected_mode = max(args.modes, key=lambda mode: (summary[mode]["mean_auroc"], summary[mode]["mean_ap"]))
        chosen = args.output / f"{selected_mode}-{args.seeds[0]}"
        run_stress(base, chosen, chosen / "stress", args.stress_sources)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "root", "weights", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--epochs", type=int, default=3)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--batch-size", type=int, default=64)
    cli.add_argument("--seeds", type=int, nargs="+", default=[42, 17, 71])
    cli.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    cli.add_argument("--curve", action="store_true")
    cli.add_argument("--stress-sources", type=int, default=0)
    cli.add_argument("--stress-checkpoint", type=Path, help="run only the stress sweep for an existing scored checkpoint")
    args = cli.parse_args()
    if args.stress_checkpoint:
        if args.stress_sources < 1:
            cli.error("--stress-checkpoint requires a positive --stress-sources")
        verify_groups(args.manifest)
        torch.set_num_threads(args.threads)
        run_stress(dict(manifest=args.manifest, root=args.root, batch_size=args.batch_size),
                   args.stress_checkpoint, args.output, args.stress_sources)
    else:
        run(args)
