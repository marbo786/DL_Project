"""Export research scores and reproducibility metadata without image paths or native IDs."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

from research.manifest import file_sha256


def anonymous(value):
    return sha256(f"ForgeryLens-receipt-v1:{value}".encode()).hexdigest()[:24]


def export(study, output, frozen_evaluation=(), stress=()):
    output.mkdir(parents=True)
    index = []

    def export_table(table, relative, fit):
        if table.name.endswith(".in-progress.csv") or table.name == "manifest.csv":
            return
        with table.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        fields = ("sample_id", "source_group", "corpus", "label", "logit", "iou", "hit", "condition")
        with target.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                anonymized = {**row, "sample_id": anonymous(row["sample_id"]),
                              "source_group": anonymous(row["source_group"])}
                writer.writerow(anonymized)
        index.append({"file": target.relative_to(output).as_posix(), "rows": len(rows),
                      "table_sha256": file_sha256(target), "private_table_sha256": file_sha256(table),
                      "checkpoint_sha256": fit["checkpoint_sha256"],
                      "fit_manifest_sha256": fit["training_manifest_sha256"],
                      "evaluation_manifest_sha256": fit.get("evaluation_manifest_sha256"),
                      "mode": fit["mode"], "seed": fit["seed"],
                      "dataset": "AIForge-Doc v2.0.1", "license_reference":
                      "https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2"})

    for metadata in sorted(study.rglob("fit.json")):
        run = metadata.parent
        fit = json.loads(metadata.read_text(encoding="utf-8"))
        for directory in (run, run / "stress"):
            for table in sorted(directory.glob("*.csv")):
                export_table(table, table.relative_to(study), fit)
    for folder in stress:
        fit = json.loads((study / folder.name / "fit.json").read_text(encoding="utf-8"))
        for table in sorted(folder.glob("*.csv")):
            export_table(table, Path("fixed-stress") / folder.name / table.name, fit)
    for folder in frozen_evaluation:
        lock = folder / "freeze-before-labels.json"
        if not lock.exists():
            lock = folder / "freeze.json"
        selection = json.loads(lock.read_text(encoding="utf-8"))
        fit = {**selection, "training_manifest_sha256": selection["manifest_sha256"], "mode": selection["configuration"]}
        if (folder / "manifest.csv").exists():
            fit["evaluation_manifest_sha256"] = file_sha256(folder / "manifest.csv")
        for table in sorted(folder.glob("*.csv")):
            export_table(table, Path(folder.name) / table.name, fit)
    if not index:
        raise ValueError("No scored runs found")
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--frozen-evaluation", type=Path, action="append", default=[])
    cli.add_argument("--stress", type=Path, action="append", default=[])
    export(**vars(cli.parse_args()))
