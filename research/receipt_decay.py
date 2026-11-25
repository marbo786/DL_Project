"""Compare AdamW decay on the original development-only ablation cohort."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from research import receipt_ablations as a
from research import receipt_model
from research.data import ManifestDocuments
from research.manifest import file_sha256


def run(root, previous, weights, output):
    manifest = previous / "scoped-manifest.csv"
    lock = a.read_json(previous / "lock.json")
    if file_sha256(manifest) != lock["scoped_manifest_sha256"] or file_sha256(weights) != lock["initial_weights_sha256"]:
        raise ValueError("Inputs differ from the matched ablation")
    rows = a.read_rows(manifest)
    if {r["split"] for r in rows} != {"train", "validation", "calibration"}:
        raise ValueError("Expected a development-only manifest")
    output.mkdir(parents=True, exist_ok=False)
    a.write_json(output / "lock.json", {"parent_lock": file_sha256(previous / "lock.json"),
                 "seeds": a.SEEDS, "factor": "AdamW weight decay", "control": .01, "variant": 0,
                 "scoring": "same cohorts and calibration procedure as parent"})
    audiences = {role: {r["sample_id"]: (r["source_group"], int(r["label"] == "manipulated"))
                        for r in rows if r["split"] == role} for role in a.SUBSET_GROUPS}
    torch.set_num_threads(4)
    base = dict(manifest=manifest, root=root, weights=weights, epochs=3, batch_size=64, threads=4)
    prepared = receipt_model.training_data(ManifestDocuments(manifest, root, "train"), 1., "residual")
    results, controls = [], []
    for seed in a.SEEDS:
        folder = output / f"no-decay-{seed}"
        reference = previous / f"residual-{seed}"
        controls.append(a.metrics(reference, audiences))
        receipt_model.train(SimpleNamespace(**base, seed=seed, mode="residual", output=folder,
                                            fraction=1., weight_decay=0.), prepared)
        a.mobile_scores(base, folder, folder)
        results.append(a.metrics(folder, audiences, reference))
    a.write_json(output / "summary.json", a.aggregate(results, controls))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for flag in ("root", "previous", "weights", "output"):
        cli.add_argument("--" + flag, required=True, type=Path)
    run(**vars(cli.parse_args()))
