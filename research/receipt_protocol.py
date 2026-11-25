"""Freeze source-connected AIForge development partitions and audit image evidence."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from research.manifest import file_sha256, resolve_image


def decoded_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def pixel_digest(image):
    return sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()


def difference_hash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes(), "big")


def perceptually_close(left_hash, left_size, right_hash, right_size):
    if (left_hash ^ right_hash).bit_count() > 3:
        return False
    aspect_ratio = (left_size[0] * right_size[1]) / (left_size[1] * right_size[0])
    return 0.95 <= aspect_ratio <= 1.05


def verify_groups(manifest):
    roles = {}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            group = row["source_group"]
            role = row["split"]
            if not group or roles.setdefault(group, role) != role:
                raise ValueError("A connected source group crosses roles")
    if not roles:
        raise ValueError("Empty source protocol")


def freeze(manifest, root, output, review=None, package=None):
    root = root.resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    sample_ids = set()
    for row in rows:
        sample = row["sample_id"]
        source = row["source_document_id"]
        if not source or not sample or sample in sample_ids:
            raise ValueError("Missing identity or duplicate sample")
        if row["split"] not in {"train", "validation", "test", "shifted_test"}:
            raise ValueError("Unexpected historical split")
        sample_ids.add(sample)
        groups[source].append(row)
    parent = {source: source for source in groups}

    def component(source):
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    def join(left, right):
        first, second = sorted((component(left), component(right)))
        parent[second] = first

    fingerprints, authentic = {}, []
    mask_areas = defaultdict(list)
    audit = Counter()
    for source, items in sorted(groups.items()):
        originals = [row for row in items if row["label"] == "genuine"]
        edits = [row for row in items if row["label"] == "manipulated"]
        if len(originals) != 1 or not edits or len(originals) + len(edits) != len(items):
            raise ValueError(f"Unexpected pairing for {source}")
        original = None
        for row in originals + edits:
            path = resolve_image(root, row["image_path"])
            image = decoded_image(path)
            if image.size != (int(row["image_width"]), int(row["image_height"])):
                raise ValueError(f"Dimensions disagree for {source}")
            digest = pixel_digest(image)
            join(source, fingerprints.setdefault(digest, source))
            row["pixel_sha256"] = digest
            pixels = np.asarray(image)
            if row["label"] == "genuine":
                original = pixels
                original_path = path
                authentic.append((source, difference_hash(image), image.size))
                continue
            if resolve_image(root, row["original_image_path"]) != original_path or not row["generator"]:
                raise ValueError(f"Unverified counterpart or generator for {source}")
            with Image.open(resolve_image(root, row["mask_path"])) as target:
                mask = np.asarray(ImageOps.exif_transpose(target).convert("L"))
            if mask.shape != pixels.shape[:2] or not np.isin(mask, [0, 255]).all() or not mask.any():
                raise ValueError(f"Invalid mask for {source}")
            inside = mask > 0
            changed = np.any(original != pixels, axis=2)
            audit["edits"] += 1
            audit["edits_with_changes_outside_mask"] += bool((changed & ~inside).any())
            audit["edits_without_changed_pixels_inside"] += not (changed & inside).any()
            mask_areas[row["source_dataset"]].append(float(inside.mean()))

    reviewed = {}
    if review:
        with review.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                key = tuple(sorted((row["source_a"], row["source_b"])))
                if key[0] == key[1] or any(source not in groups for source in key) or key in reviewed:
                    raise ValueError("Invalid or repeated review pair")
                decision = row["decision"]
                if decision not in {"same_source", "distinct", "uncertain"} or not row["evidence"].strip():
                    raise ValueError("Review requires a decision and evidence")
                reviewed[key] = decision
                if decision == "same_source":
                    join(*key)

    candidates = []
    for (left, left_hash, left_size), (right, right_hash, right_size) in itertools.combinations(authentic, 2):
        if perceptually_close(left_hash, left_size, right_hash, right_size) and component(left) != component(right):
            candidates.append((left, right, (left_hash ^ right_hash).bit_count(), reviewed.get((left, right), "unreviewed")))

    historical_roles = defaultdict(set)
    for row in rows:
        source = component(row["source_document_id"])
        historical_roles[source].add(row["split"])
    counts = Counter()
    for row in rows:
        source = component(row["source_document_id"])
        row["source_group"] = source
        row["historical_split"] = row["split"]
        role = row["split"]
        if len(historical_roles[source]) > 1:
            role = "quarantined"
        elif role == "train":
            digest = sha256(f"forgerylens-calibration-v1:{source}".encode()).hexdigest()
            if int(digest[:8], 16) % 5 == 0:
                role = "calibration"
        row["split"] = role
        counts[f"{role}/{row['source_dataset']}/{row['label']}"] += 1

    output.mkdir(parents=True)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "near_duplicates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_a", "source_b", "hamming_distance", "decision"))
        writer.writerows(candidates)
    summary = {
        "version": 1, "input_manifest_sha256": file_sha256(manifest),
        "output_manifest_sha256": file_sha256(output / "manifest.csv"),
        "counts": dict(sorted(counts.items())), "audit": dict(audit),
        "source_ids": len(groups), "connected_sources": len(historical_roles),
        "near_duplicate_decisions": dict(Counter(row[-1] for row in candidates)),
        "mask_fraction_quantiles": {key: np.quantile(values, [.05, .5, .95]).tolist()
                                    for key, values in sorted(mask_areas.items())},
    }
    if package:
        metadata = package / "metadata.jsonl"
        records = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
        summary["package"] = {
            "metadata_sha256": file_sha256(metadata),
            "mapping_sha256": file_sha256(package / "source_mapping.csv"),
            "versions": json.loads((package / "dataset_info.json").read_text(encoding="utf-8"))["version"],
            "categories": dict(Counter(record["doc_type"] for record in records)),
            "v1_assignment_leads": dict(Counter(str(record.get("v1_assigned_tool")) for record in records)),
        }
    (output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--package", type=Path)
    freeze(**vars(parser.parse_args()))
