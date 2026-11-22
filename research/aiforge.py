"""Import the full AIForge-Doc v2 archive into a receipt-only experiment."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from zipfile import ZipFile

from PIL import Image

from research.manifest import COLUMNS


def relative_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or any(
        ":" in part or "\\" in part or part.endswith((".", " ")) for part in path.parts
    ):
        raise ValueError("Invalid package member path.")
    return path


def partition(source, corpus, published_test, held_out):
    if corpus == held_out:
        return "shifted_test"
    if source in published_test:
        return "test"
    bucket = int.from_bytes(sha256(source.encode("utf-8")).digest()[:4], "big") % 5
    return "validation" if bucket == 0 else "train"


def copy_asset(archive, prefix, member, root):
    path = relative_path(member)
    entry = archive.getinfo((prefix / path).as_posix())
    if entry.is_dir() or entry.file_size > 64 * 1024 * 1024:
        raise ValueError(f"Invalid image member: {member}")
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Image destination must stay inside the import root.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        with archive.open(entry) as source, target.open("xb") as destination:
            copyfileobj(source, destination)
    return target


def prepare(package, destination, held_out):
    if destination.exists():
        raise FileExistsError("Choose a new output directory for this import.")
    destination = destination.resolve()
    with ZipFile(package) as archive:
        candidates = [name for name in archive.namelist() if PurePosixPath(name).name == "metadata.jsonl"]
        if len(candidates) != 1:
            raise ValueError("Expected one metadata.jsonl in the full AIForge-Doc v2 archive.")
        prefix = relative_path(candidates[0]).parent
        metadata = archive.getinfo(candidates[0])
        if metadata.file_size > 32 * 1024 * 1024:
            raise ValueError("Unexpected metadata size.")
        with archive.open(metadata) as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        receipts = []
        published_test = set()
        seen_specs = set()
        for record in records:
            if record["doc_type"].casefold() != "receipt":
                continue
            source = f"{record['source_dataset']}:{record['image_id']}"
            spec = str(record["spec_id"])
            if spec in seen_specs or not record["source_dataset"] or not str(record["image_id"]).strip():
                raise ValueError("Duplicate or incomplete receipt provenance.")
            seen_specs.add(spec)
            forged = relative_path(record["image"])
            if len(forged.parts) != 3 or forged.parts[1] != "images":
                raise ValueError("Unexpected AIForge forged-image directory layout.")
            authentic = forged.parent.parent / "authentic" / forged.name
            mask = relative_path(record["mask"])
            expected_directory = {"training": "TrainingSet", "testing": "TestingSet"}.get(record["split"])
            if (forged.parts[0] != expected_directory or forged.suffix.lower() != ".png"
                    or len(forged.stem) != 9 or not forged.stem.isascii() or not forged.stem.isdecimal()
                    or mask != forged.parent.parent / "masks" / forged.name):
                raise ValueError("Unexpected AIForge image or mask filename.")
            for path in (forged, mask, authentic):
                if (prefix / path).as_posix() not in archive.NameToInfo:
                    raise ValueError("The full package with paired authentic images and masks is required.")
            if record["split"] == "testing":
                published_test.add(source)
            elif record["split"] != "training":
                raise ValueError("Unexpected publisher partition.")
            receipts.append((record, source, spec, forged, authentic, mask))

        rows = []
        counts = Counter()
        root = destination / "assets"
        assignments = [
            partition(source, record["source_dataset"], published_test, held_out)
            for record, source, *_ in receipts
        ]
        if set(assignments) != {"train", "validation", "test", "shifted_test"}:
            raise ValueError("Each partition needs at least one receipt pair. Check the held-out corpus.")
        root.mkdir(parents=True, exist_ok=False)
        for (record, source, spec, forged, authentic, mask), split in zip(receipts, assignments):
            copied = {path: copy_asset(archive, prefix, str(path), root) for path in (forged, authentic, mask)}
            with Image.open(copied[forged]) as edited, Image.open(copied[authentic]) as original, Image.open(copied[mask]) as target:
                if edited.size != original.size or edited.size != target.size:
                    raise ValueError("Paired image and mask sizes do not match.")
                width, height = edited.size
            fields = {
                "source_document_id": source, "source_dataset": record["source_dataset"], "document_category": "receipt",
                "split": split, "image_width": width, "image_height": height,
            }
            rows.append({**fields, "sample_id": f"{spec}:original", "variant_id": f"{spec}:original",
                         "label": "genuine", "image_path": authentic.as_posix()})
            rows.append({**fields, "sample_id": f"{spec}:edited", "variant_id": f"{spec}:edited",
                         "label": "manipulated", "generator": record["assigned_tool"], "image_path": forged.as_posix(),
                         "mask_path": mask.as_posix(), "original_image_path": authentic.as_posix()})
            counts[split] += 1

    manifest = destination / "manifest.csv"
    with manifest.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, dict(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--held-out", default="wildreceipt", choices=("wildreceipt", "cord", "sroie"))
    args = parser.parse_args()
    manifest, counts = prepare(args.package, args.destination, args.held_out)
    print(manifest)
    print("Receipt pairs:", counts)
