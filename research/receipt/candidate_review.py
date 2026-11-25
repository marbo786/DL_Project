"""Conservatively contain perceptual duplicate candidates within source groups."""

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from research.manifest import file_sha256, resolve_image
from research.receipt.protocol import decoded_image


def load_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def contain(base, root, output):
    rows = load_rows(base / "manifest.csv")
    candidates = load_rows(base / "near_duplicates.csv")
    source_group = {row["source_document_id"]: row["source_group"] for row in rows}
    parent = {group: group for group in source_group.values()}
    originals = {row["source_document_id"]: row["image_path"] for row in rows if row["label"] == "genuine"}
    root = root.resolve()

    def component(source):
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    @lru_cache(maxsize=64)
    def thumbnail(source):
        image = decoded_image(resolve_image(root, originals[source]))
        return np.asarray(image.convert("L").resize((144, 216), Image.Resampling.LANCZOS), dtype=np.float32)

    for candidate in candidates:
        left, right = candidate["source_a"], candidate["source_b"]
        a, b = thumbnail(left), thumbnail(right)
        candidate["thumbnail_gray_mae"] = float(np.mean(np.abs(a - b)))
        if candidate["decision"] != "distinct":
            first, second = sorted((component(source_group[left]), component(source_group[right])))
            parent[second] = first
            candidate["handling"] = "contained_as_possible_alias"
        else:
            candidate["handling"] = "reviewed_distinct"

    historical = defaultdict(set)
    for row in rows:
        row["source_group"] = component(row["source_group"])
        historical[row["source_group"]].add(row["historical_split"])
    counts = Counter()
    for row in rows:
        source = row["source_group"]
        role = row["historical_split"]
        if len(historical[source]) > 1:
            role = "quarantined"
        elif role == "train":
            digest = sha256(f"forgerylens-calibration-v1:{source}".encode()).hexdigest()
            if int(digest[:8], 16) % 5 == 0:
                role = "calibration"
        row["split"] = role
        counts[f"{role}/{row['source_dataset']}/{row['label']}"] += 1

    output.mkdir(parents=True)
    write_rows(output / "manifest.csv", rows, list(rows[0]))
    write_rows(output / "candidate_review.csv", candidates,
               ["source_a", "source_b", "hamming_distance", "decision", "thumbnail_gray_mae", "handling"])
    report = {"version": "1-contained", "input_manifest_sha256": file_sha256(base / "manifest.csv"),
              "output_manifest_sha256": file_sha256(output / "manifest.csv"),
              "candidate_table_sha256": file_sha256(output / "candidate_review.csv"),
              "source_ids": len(source_group), "connected_groups": len(historical),
              "quarantined_groups": sum(len(roles) > 1 for roles in historical.values()),
              "counts": dict(sorted(counts.items())), "candidate_handling": dict(Counter(
                  row["handling"] for row in candidates))}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "root", "output"):
        cli.add_argument(field, type=Path)
    contain(**vars(cli.parse_args()))
