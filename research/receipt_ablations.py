"""Run paired development-only ablations against the selected receipt recipe."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace

import torch

from research import receipt_model, train as legacy
from research.data import ManifestDocuments
from research.manifest import file_sha256
from research.receipt_metrics import evaluate
from research.receipt_protocol import verify_groups


SEEDS = (17, 42, 71)
SUBSET_GROUPS = {"calibration": 60, "validation": 80}
MOBILE_CHANGES = {
    "tail-three": {"trainable_tail": 3},
    "unweighted-bce": {"weighted_bce": False},
    "symmetric-brightness": {"brightness_gains": (.85, 1.15, .95)},
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    temporary = path.with_suffix(".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.rename(path)


def development_manifest(source):
    rows = read_rows(source)
    selected = {role: set(sorted({row["source_group"] for row in rows if row["split"] == role},
                                key=lambda group: sha256(f"receipt-ablation-v1:{role}:{group}".encode()).digest())[:limit])
                for role, limit in SUBSET_GROUPS.items()}
    return [row for row in rows if row["split"] == "train"
            or row["source_group"] in selected.get(row["split"], set())]


def saved_checkpoint(study, mode, seed, manifest_hash, weights_hash):
    path = study / f"{mode}-{seed}"
    fit = read_json(path / "fit.json")
    expected = {"mode": mode, "seed": seed, "fraction": 1., "batch_size": 64, "threads": 4,
                "training_examples": 1755, "trained_groups": 556,
                "training_manifest_sha256": manifest_hash, "initial_weights_sha256": weights_hash}
    if any(fit[k] != v for k, v in expected.items()) or len(fit["losses"]) != 3:
        raise ValueError(f"Baseline recipe differs for {path.name}")
    if file_sha256(path / "model.pt") != fit["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash differs for {path.name}")
    return path


def reuse_scores(source, folder, audiences):
    report = read_json(source / "metrics.json")
    folder.mkdir(exist_ok=True)
    for role, ids in audiences.items():
        path = source / f"{role}.csv"
        field = "evaluation_sha256" if role == "validation" else "calibration_sha256"
        if file_sha256(path) != report[field]:
            raise ValueError(f"Saved score digest differs for {source.name}/{role}")
        matched = [row for row in read_rows(path) if row["sample_id"] in ids]
        if len(matched) != len(ids):
            raise ValueError(f"Missing saved scores for {source.name}/{role}")
        output = folder / path.name
        if not output.exists():
            write_rows(output, matched)


def mobile_scores(base, checkpoint, folder, geometry=None):
    for role in SUBSET_GROUPS:
        target = folder / f"{role}.csv"
        if not target.exists():
            temporary = target.with_suffix(".partial")
            if temporary.exists():
                temporary = folder / f"{role}-retry.partial"
            receipt_model.score(SimpleNamespace(**base, checkpoint=checkpoint, split=role, output=temporary,
                                                condition="clean", limit_sources=None, geometry=geometry))
            temporary.rename(target)


def legacy_scores(folder, cached, seed, map_weight):
    folder.mkdir(exist_ok=True)
    if not (folder / "model.pt").exists():
        model, curve = legacy.train("local", {"width": 8}, cached["train"][1], cached["validation"][1], 8, seed,
                                    map_weight=map_weight, checkpoint_rule="final epoch")
        torch.save(model.state_dict(), folder / "model.pt")
        write_json(folder / "fit.json", {"seed": seed, "map_weight": map_weight, "training_log": curve,
                                        "checkpoint_sha256": file_sha256(folder / "model.pt")})
    fit = read_json(folder / "fit.json")
    if file_sha256(folder / "model.pt") != fit["checkpoint_sha256"]:
        raise ValueError(f"Legacy checkpoint hash differs for {folder.name}")
    model = legacy.ReceiptNet(width=8).eval()
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    for role in SUBSET_GROUPS:
        output = folder / f"{role}.csv"
        if output.exists():
            continue
        documents, tensors = cached[role]
        logits, _, _, _ = legacy.collect(model, tensors)
        write_rows(output, [{"sample_id": row["sample_id"], "source_group": row["source_group"],
                             "label": int(row["label"] == "manipulated"), "logit": float(logit),
                             "iou": "", "hit": "", "milliseconds": ""}
                            for row, logit in zip(documents.rows, logits)])


def metrics(folder, audiences, reference=None):
    for role, ids in audiences.items():
        rows = read_rows(folder / f"{role}.csv")
        if (len(rows) != len(ids) or any(ids.get(row["sample_id"]) != (row["source_group"], int(row["label"]))
                                        for row in rows) or len({row["sample_id"] for row in rows}) != len(ids)):
            raise ValueError(f"Unmatched or mislabeled scores for {folder.name}/{role}")
    output = folder / "metrics.json"
    if not output.exists():
        evaluate(folder / "calibration.csv", folder / "validation.csv", output, repeats=1000,
                 compare=reference / "validation.csv" if reference else None)
    report = read_json(output)
    point = report["operating_points"]["0.05"]
    result = {"auroc": report["auroc"], "recall": point["recall"], "fpr": point["fpr"]}
    if reference:
        result["source_bootstrap_auroc_delta_95"] = report["paired"]["source_bootstrap_95"]["paired_auroc_delta"]
    return result


def aggregate(rows, reference=None):
    result = {"per_seed": rows}
    for field in ("auroc", "recall", "fpr"):
        values = [row[field] for row in rows]
        result[field] = {"mean": mean(values), "sample_sd": stdev(values)}
        if reference:
            deltas = [value - row[field] for value, row in zip(values, reference)]
            result[field]["paired_delta"] = {"per_seed": deltas, "mean": mean(deltas), "sample_sd": stdev(deltas)}
    return result


def run(args):
    torch.set_num_threads(4)
    manifest_hash = file_sha256(args.manifest)
    weights_hash = file_sha256(args.weights)
    verify_groups(args.manifest)
    saved = {(mode, seed): saved_checkpoint(args.study, mode, seed, manifest_hash, weights_hash)
             for mode in ("residual", "full") for seed in SEEDS}
    scoped = development_manifest(args.manifest)
    manifest = args.output / "scoped-manifest.csv"
    args.output.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        write_rows(manifest, scoped)
    audiences = {role: {row["sample_id"]: (row["source_group"], int(row["label"] == "manipulated"))
                        for row in scoped if row["split"] == role} for role in SUBSET_GROUPS}
    training = ManifestDocuments(manifest, args.root, "train")
    lock = {"run_id": args.output.name, "source_manifest_sha256": manifest_hash, "initial_weights_sha256": weights_hash,
            "scoped_manifest_sha256": file_sha256(manifest), "development_group_limits": SUBSET_GROUPS, "seeds": SEEDS,
            "mobile": {"epochs": 3, "tile_batch": 64, "threads": 4, "checkpoint_rule": "final epoch",
                       "adamw_weight_decay": .01, "reference": "residual, weighted BCE, no augmentation, full fine-tuning",
                       "controls": MOBILE_CHANGES, "geometry": "fixed residual weights, whole-resize versus native inference"},
            "legacy": {"epochs": 8, "batch_size": 32, "width": 8, "checkpoint_rule": "final epoch", "map_weights": [0., .65]},
            "reference_checkpoints": {path.name: file_sha256(path / "model.pt") for path in saved.values()}}
    lock_path = args.output / "lock.json"
    stable = json.loads(json.dumps(lock))
    if read_rows(manifest) != scoped or (lock_path.exists() and read_json(lock_path) != stable):
        raise ValueError("Scoped manifest or ablation lock differs")
    if not lock_path.exists():
        write_json(lock_path, lock)

    base = dict(manifest=manifest, root=args.root, weights=args.weights, epochs=3, batch_size=64, threads=4)
    prepared = receipt_model.training_data(training, 1., "residual")
    if (prepared[2], prepared[3], len(prepared[1]), int(prepared[1].sum())) != (556, 556, 1755, 585):
        raise ValueError("Training composition differs from the saved baseline")
    rows = {name: [] for name in ("residual", "rgb", *MOBILE_CHANGES, "whole-inference", "legacy-mask", "legacy-no-mask")}
    cached = {role: (docs, legacy.cache(docs)) for role in ("train", *SUBSET_GROUPS)
              for docs in (ManifestDocuments(manifest, args.root, role),)}
    for seed in SEEDS:
        reference = args.output / f"residual-{seed}"
        reuse_scores(saved["residual", seed], reference, audiences)
        rows["residual"].append(metrics(reference, audiences))
        rgb = args.output / f"rgb-{seed}"
        reuse_scores(saved["full", seed], rgb, audiences)
        rows["rgb"].append(metrics(rgb, audiences, reference))
        for name, controls in MOBILE_CHANGES.items():
            folder = args.output / f"{name}-{seed}"
            if not folder.exists():
                receipt_model.train(SimpleNamespace(**base, mode="residual", seed=seed, fraction=1.,
                                                    output=folder, **controls), prepared)
            mobile_scores(base, folder, folder)
            rows[name].append(metrics(folder, audiences, reference))
            print(f"Scored {name} seed {seed}", flush=True)
        whole = args.output / f"whole-inference-{seed}"
        mobile_scores(base, saved["residual", seed], whole, geometry="whole")
        rows["whole-inference"].append(metrics(whole, audiences, reference))
        legacy_ref = args.output / f"legacy-mask-{seed}"
        legacy_scores(legacy_ref, cached, seed, .65)
        rows["legacy-mask"].append(metrics(legacy_ref, audiences))
        no_mask = args.output / f"legacy-no-mask-{seed}"
        legacy_scores(no_mask, cached, seed, 0.)
        rows["legacy-no-mask"].append(metrics(no_mask, audiences, legacy_ref))
        print(f"Completed all controls for seed {seed}", flush=True)
    results = {name: aggregate(values, None if name in ("residual", "legacy-mask") else
                               rows["legacy-mask" if name == "legacy-no-mask" else "residual"])
               for name, values in rows.items()}
    write_json(args.output / "summary.json", {"run_id": args.output.name, "seeds": SEEDS,
                                              "lock_sha256": file_sha256(lock_path),
                                              "cohort_documents": {role: len(ids) for role, ids in audiences.items()},
                                              "results": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("manifest", "weights", "root", "study", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    run(parser.parse_args())
