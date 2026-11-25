"""Prepare the v2 side of a paired generator study from the local official release."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from research.manifest import file_sha256
from research.receipt.generator_pairs import read_csv


def build_inventory(dataset, assignments, output):
    mapping_file = dataset / "source_mapping.csv"
    metadata_file = dataset / "metadata.jsonl"
    mapping = {row["spec_id"]: row for row in read_csv(mapping_file)}
    metadata = {}
    with metadata_file.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["spec_id"] in metadata:
                raise ValueError(f"Repeated release spec {row['spec_id']}")
            metadata[row["spec_id"]] = row
    output.mkdir(parents=True)
    counts = Counter()
    fields = ("spec_id", "source_document_id", "source_dataset", "generator",
              "image_path", "authentic_path", "mask_path", "source_group", "split",
              "v1_generator_hint")
    with (output / "v2.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in read_csv(assignments):
            if row["label"] != "manipulated" or row["split"] not in {"test", "shifted_test"}:
                continue
            spec = row["sample_id"].split(":", 1)[0]
            source, release = mapping[spec], metadata[spec]
            corpus = source["source_dataset"]
            if (source["verified"] != "True"
                    or (corpus, source["image_id"]) != (release["source_dataset"], release["image_id"])
                    or f"{corpus}:{source['image_id']}" != row["source_document_id"]
                    or release["assigned_tool"] != row["generator"]
                    or (source["forged"], source["authentic"], source["mask"])
                    != (row["image_path"], row["original_image_path"], row["mask_path"])):
                raise ValueError(f"Release and audited assignment disagree for {spec}")
            hint = release.get("v1_assigned_tool") or "unspecified"
            writer.writerow(dict(spec_id=spec, source_document_id=row["source_document_id"],
                                 source_dataset=corpus, generator=row["generator"], image_path=source["forged"],
                                 authentic_path=source["authentic"], mask_path=source["mask"],
                                 source_group=row["source_group"], split=row["split"], v1_generator_hint=hint))
            counts[f"{row['split']}/{corpus}/{hint}"] += 1
    summary = {"v2_candidates": sum(counts.values()), "v1_pairs_verified": 0, "counts": dict(sorted(counts.items())),
               "sha256": {p.name: file_sha256(p) for p in (assignments, mapping_file, metadata_file, output / "v2.csv")}}
    (output / "inventory.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "assignments", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    build_inventory(**vars(cli.parse_args()))
