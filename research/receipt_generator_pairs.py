"""Join normalized AIForge versions by edit spec and verified source crosswalk."""

import argparse
import csv
from pathlib import Path

from research.manifest import resolve_image
from research.receipt_protocol import decoded_image, pixel_digest


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def version_rows(path):
    by_spec = {}
    for row in read_csv(path):
        spec = row["spec_id"]
        if not spec or spec in by_spec or not all(row[key] for key in (
                "source_document_id", "source_dataset", "generator", "image_path", "authentic_path", "mask_path")):
            raise ValueError("A version manifest requires unique specs and complete source provenance")
        by_spec[spec] = row
    return by_spec


def join_versions(v1, v2, v1_root, v2_root, assignments, crosswalk, output, normalized_output=None):
    first, second = version_rows(v1), version_rows(v2)
    old_to_new = {}
    for row in read_csv(crosswalk):
        left, right = row["v1_source_id"], row["v2_source_id"]
        if not left or not right or not row["evidence"].strip() or left in old_to_new:
            raise ValueError("Crosswalk requires explicit one-row-per-v1-ID provenance")
        old_to_new[left] = right
    assignment = {}
    for row in read_csv(assignments):
        source = row["source_document_id"]
        state = (row["source_group"], row["split"])
        if assignment.setdefault(source, state) != state:
            raise ValueError("Assigned source has inconsistent group or historical split")
    fitted = {group for group, role in assignment.values() if role in {"train", "calibration", "validation", "quarantined"}}
    v1_root, v2_root = v1_root.resolve(), v2_root.resolve()
    results = []
    normalized = {"v1": [], "v2": []}
    for spec in sorted(set(first) & set(second)):
        a, b = first[spec], second[spec]
        source = b["source_document_id"]
        if source not in assignment or source != old_to_new.get(a["source_document_id"]):
            continue
        group, role = assignment[source]
        if group in fitted or role not in {"test", "shifted_test"}:
            continue
        if a["source_dataset"] != b["source_dataset"] or a["generator"] == b["generator"]:
            raise ValueError(f"Spec {spec} does not isolate generator shift")
        counterparts = []
        for row, root in ((a, v1_root), (b, v2_root)):
            original = decoded_image(resolve_image(root, row["authentic_path"]))
            edited = decoded_image(resolve_image(root, row["image_path"]))
            mask = decoded_image(resolve_image(root, row["mask_path"]))
            if original.size != edited.size or mask.size != edited.size:
                raise ValueError(f"Spec {spec} lacks aligned version assets")
            counterparts.append(pixel_digest(original))
        if counterparts[0] != counterparts[1]:
            raise ValueError(f"Spec {spec} requires independent review of differing authentic pixels")
        results.append((spec, source, group, role, b["source_dataset"], a["generator"], b["generator"],
                        a["image_path"], b["image_path"], a["mask_path"], b["mask_path"],
                        a["authentic_path"], b["authentic_path"], counterparts[0]))
        for version, row in (("v1", a), ("v2", b)):
            for label, path, mask in (("genuine", row["authentic_path"], ""),
                                      ("manipulated", row["image_path"], row["mask_path"])):
                normalized[version].append((f"{spec}:{label}", source, group, role, b["source_dataset"],
                                            label, path, mask))
    if not results:
        raise ValueError("No eligible generator pairs with verified local images")
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("spec_id", "source_document_id", "source_group", "historical_role", "source_dataset",
                         "v1_generator", "v2_generator", "v1_image", "v2_image", "v1_mask", "v2_mask",
                         "v1_authentic", "v2_authentic", "authentic_pixel_sha256"))
        writer.writerows(results)
    if normalized_output:
        normalized_output.mkdir(parents=True)
        for version, rows in normalized.items():
            with (normalized_output / f"{version}.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("sample_id", "source_document_id", "source_group", "split", "source_dataset",
                                 "label", "image_path", "mask_path"))
                writer.writerows(rows)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("v1", "v2", "v1-root", "v2-root", "assignments", "crosswalk", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--normalized-output", type=Path)
    join_versions(**vars(cli.parse_args()))
