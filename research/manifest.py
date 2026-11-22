"""Audit already normalized CSV manifests without assigning research splits."""

import argparse
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
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(128 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def audit(manifest: Path, root: Path) -> list[str]:
    errors: list[str] = []
    source_splits: dict[str, str] = {}
    asset_splits: dict[str, str] = {}
    sample_ids: set[str] = set()
    variants: set[tuple[str, str]] = set()
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
                    fingerprint = check_asset(resolve_image(root, row[field]), expected)
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
    args = parser.parse_args()
    findings = audit(args.manifest, args.asset_root)
    print("\n".join(findings) if findings else "Manifest checks passed.")
    raise SystemExit(bool(findings))