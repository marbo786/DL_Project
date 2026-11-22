"""Audit already normalized CSV manifests without assigning research splits."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path

from PIL import Image

COLUMNS = {
    "sample_id", "source_document_id", "variant_id", "label", "source_dataset",
    "generator", "document_category", "image_path", "mask_path",
    "original_image_path", "split", "image_width", "image_height",
}
SPLITS = {"train", "validation", "test", "shifted_test"}


def resolve_image(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not value or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing or out-of-root file {value!r}")
    return path


def check_asset(path: Path, expected: tuple[int, int]) -> str:
    with Image.open(path) as image:
        if image.size != expected:
            raise ValueError(f"{path.name} is {image.size}, expected {expected}")
        image.load()
    return file_sha256(path)


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(128 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def deduplicate(manifest: Path, root: Path, output: Path) -> dict[str, int]:
    """Keep one source per exact-image component, prioritizing the held-out splits."""
    root = root.resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames

    priority = {"shifted_test": 0, "test": 1, "validation": 2, "train": 3}
    source_splits = {}
    fingerprints = {}
    sources_by_image = defaultdict(set)
    for row in rows:
        source, split = row["source_document_id"], row["split"]
        if not source or source_splits.setdefault(source, split) != split:
            raise ValueError(f"Invalid source partition for {source!r}")
        for field in ("image_path", "original_image_path"):
            if not row[field]:
                continue
            path = resolve_image(root, row[field])
            if path not in fingerprints:
                fingerprints[path] = file_sha256(path)
            sources_by_image[fingerprints[path]].add(source)

    neighbors = defaultdict(set)
    for sources in sources_by_image.values():
        if len(sources) > 1:
            anchor, *other_sources = sorted(sources)
            neighbors[anchor].update(other_sources)
            for source in other_sources:
                neighbors[source].add(anchor)

    rejected = set()
    for source in sorted(source_splits, key=lambda item: (priority[source_splits[item]], item)):
        if source in rejected:
            continue
        pending = list(neighbors[source])
        while pending:
            other = pending.pop()
            if other != source and other not in rejected:
                rejected.add(other)
                pending.extend(neighbors[other])

    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if row["source_document_id"] not in rejected)
    return dict(Counter(source_splits[source] for source in rejected))


def audit(manifest: Path, root: Path) -> list[str]:
    errors: list[str] = []
    source_splits: dict[str, str] = {}
    asset_splits: dict[str, str] = {}
    sample_ids: set[str] = set()
    variants: set[tuple[str, str]] = set()
    verified_images: dict[tuple[Path, tuple[int, int]], str] = {}
    root = root.resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        missing = COLUMNS.difference(rows.fieldnames or [])
        if missing:
            return [f"Missing columns: {', '.join(sorted(missing))}"]
        count = 0
        for line, raw in enumerate(rows, start=2):
            count += 1
            row = {field: (raw.get(field) or "").strip() for field in COLUMNS}
            try:
                sample, source, variant = (row[key] for key in ("sample_id", "source_document_id", "variant_id"))
                split = row["split"]
                if not all((sample, source, variant, row["source_dataset"], row["document_category"])):
                    raise ValueError("missing required identifier or provenance field")
                if split not in SPLITS or row["label"] not in ("genuine", "manipulated"):
                    raise ValueError("invalid split or label")
                if sample in sample_ids or (source, variant) in variants:
                    raise ValueError("duplicate sample or source/variant")
                sample_ids.add(sample)
                variants.add((source, variant))
                if source_splits.setdefault(source, split) != split:
                    raise ValueError(f"source group {source!r} crosses splits")
                expected = (int(row["image_width"]), int(row["image_height"]))
                for field in ("image_path", "original_image_path"):
                    if field == "original_image_path" and not row[field]:
                        continue
                    path = resolve_image(root, row[field])
                    key = (path, expected)
                    if key not in verified_images:
                        verified_images[key] = check_asset(path, expected)
                    fingerprint = verified_images[key]
                    if asset_splits.setdefault(fingerprint, split) != split:
                        raise ValueError("identical image bytes cross splits")
                if row["label"] == "manipulated" and not row["generator"]:
                    raise ValueError("manipulated sample requires a generator identifier")
                if row["mask_path"]:
                    mask_path = resolve_image(root, row["mask_path"])
                    with Image.open(mask_path) as mask:
                        if mask.size != expected or mask.mode not in ("1", "L"):
                            raise ValueError("mask must be grayscale and aligned to the image")
                        colors = mask.convert("L").getcolors(maxcolors=3)
                        values = {value for _, value in colors or []}
                        if not values or not values <= {0, 255}:
                            raise ValueError("mask must contain only 0 and 255")
                        if (255 in values) != (row["label"] == "manipulated"):
                            raise ValueError("mask conflicts with label")
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                errors.append(f"Row {line}: {error}")
    return errors if count else ["Manifest has no rows."]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("asset_root", type=Path)
    parser.add_argument("--deduplicated-output", type=Path,
                        help="write a new manifest with exact duplicate source groups removed")
    args = parser.parse_args()
    if args.deduplicated_output:
        removed = deduplicate(args.manifest, args.asset_root, args.deduplicated_output)
        print("Removed source groups:", removed)
        raise SystemExit(0)
    findings = audit(args.manifest, args.asset_root)
    print("\n".join(findings) if findings else "Manifest checks passed.")
    raise SystemExit(bool(findings))
