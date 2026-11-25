"""Evaluate held-back XFUND forms after checkpoint and operating-point selection."""

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps
import torch

from research.manifest import file_sha256, resolve_image
from research.receipt.metrics import evaluate
from research.receipt.model import score
from research.receipt.protocol import decoded_image, difference_hash, perceptually_close, pixel_digest, verify_groups


def forms_manifest(package, receipt_manifest, receipt_root, output):
    with receipt_manifest.open(newline="", encoding="utf-8") as stream:
        receipts = list(csv.DictReader(stream))
    receipt_pixels = {row["pixel_sha256"] for row in receipts}
    receipt_ids = {row["source_document_id"] for row in receipts}
    thumbs = []
    for row in receipts:
        if row["label"] == "genuine":
            image = decoded_image(resolve_image(receipt_root, row["image_path"]))
            thumbs.append((difference_hash(image), image.size))

    with (package / "source_mapping.csv").open(newline="", encoding="utf-8-sig") as stream:
        mapping = {row["spec_id"]: row for row in csv.DictReader(stream)}
    records = [json.loads(line) for line in (package / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    forms = [record for record in records if record["doc_type"] == "form" and record["source_dataset"] == "xfund"]
    if len(forms) != 418:
        raise ValueError("Unexpected XFUND cohort for the pinned package")
    items = {}
    compared_pixels = defaultdict(list)
    rejected = set()
    form_thumbs = {}
    mask_sizes = []
    outside = 0
    for record in forms:
        spec = record["spec_id"]
        entry = mapping[spec]
        source = f"xfund:{record['image_id']}"
        if (source in items or entry["source_dataset"] != "xfund" or entry["image_id"] != record["image_id"]
                or entry["verified"].casefold() != "true" or record["assigned_tool"] != "openai-gpt-image-2"
                or entry["forged"] != record["image"] or entry["mask"] != record["mask"]):
            raise ValueError(f"Unverified cohort provenance for {source}")
        genuine = decoded_image(resolve_image(package, entry["authentic"]))
        edited = decoded_image(resolve_image(package, entry["forged"]))
        with Image.open(resolve_image(package, entry["mask"])) as target:
            mask = np.asarray(ImageOps.exif_transpose(target).convert("L"))
        if (genuine.size != edited.size or edited.size != (int(entry["width"]), int(entry["height"]))
                or mask.shape != (edited.height, edited.width)
                or ((mask != 0) & (mask != 255)).any() or not mask.any()):
            raise ValueError(f"Unaligned form assets for {source}")
        differences = np.any(np.asarray(genuine) != np.asarray(edited), axis=2)
        inside = mask > 0
        if not (differences & inside).any():
            raise ValueError(f"Missing changed pixels in {source}")
        outside += int((differences & ~inside).any())
        mask_sizes.append(float(inside.mean()))
        digests = [pixel_digest(genuine), pixel_digest(edited)]
        for digest in digests:
            compared_pixels[digest].append(source)
        form_hash = difference_hash(genuine)
        form_thumbs[source] = (form_hash, genuine.size)
        if (source in receipt_ids or any(digest in receipt_pixels for digest in digests)
                or any(perceptually_close(form_hash, genuine.size, *item) for item in thumbs)):
            rejected.add(source)
        items[source] = (spec, entry)

    parent = dict.fromkeys(items)
    for source in parent:
        parent[source] = source

    def component(source):
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    def join(left, right):
        head, tail = sorted((component(left), component(right)))
        parent[tail] = head

    for sources in compared_pixels.values():
        for source in sources[1:]:
            join(sources[0], source)
    aliases = 0
    for (left, left_info), (right, right_info) in itertools.combinations(form_thumbs.items(), 2):
        if perceptually_close(*left_info, *right_info):
            aliases += 1
            join(left, right)
    rejected_groups = {component(source) for source in rejected}
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_id", "source_document_id", "source_group", "split", "source_dataset",
                         "label", "image_path", "mask_path"))
        retained = []
        for source, (spec, entry) in sorted(items.items()):
            group = component(source)
            if group in rejected_groups:
                continue
            retained.append(group)
            for label, path, mask in (("genuine", entry["authentic"], ""),
                                      ("manipulated", entry["forged"], entry["mask"])):
                writer.writerow((f"{spec}:{label}", source, group, "test", "xfund", label, path, mask))
    return {"forms": len(items), "retained_pairs": len(retained), "retained_groups": len(set(retained)),
            "quarantined_pairs": len(items) - len(retained), "form_near_alias_edges": aliases,
            "pairs_with_changes_outside_mask": outside, "mask_fraction_median": float(np.median(mask_sizes))}


def run(args):
    torch.set_num_threads(args.threads)
    verify_groups(args.manifest)
    selection = json.loads(args.lock.read_text(encoding="utf-8"))
    chosen = args.study / f"{selection['configuration']}-{selection['seed']}"
    calibration = chosen / "calibration.csv"
    if (selection["manifest_sha256"] != file_sha256(args.manifest)
            or selection["checkpoint_sha256"] != file_sha256(chosen / "model.pt")
            or selection["calibration_sha256"] != file_sha256(calibration)):
        raise ValueError("Selected checkpoint, receipt manifest or calibration differs from lock")
    args.output.mkdir(parents=True)
    package = args.package.resolve()
    frozen = {**selection, "evaluation_status": "previously_excluded_xfund_forms",
              "package_metadata_sha256": file_sha256(package / "metadata.jsonl"),
              "source_mapping_sha256": file_sha256(package / "source_mapping.csv")}
    (args.output / "freeze-before-labels.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    manifest = args.output / "manifest.csv"
    audit = forms_manifest(package, args.manifest, args.receipt_root.resolve(), manifest)
    audit["manifest_sha256"] = file_sha256(manifest)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    output = args.output / "scores.csv"
    score(SimpleNamespace(manifest=manifest, root=package, checkpoint=chosen, batch_size=64, split="test",
                          output=output, condition="clean", limit_sources=None))
    report = evaluate(calibration, output, args.output / "metrics.json")
    print(f"Previously excluded XFUND forms: AUROC={report['auroc']:.4f}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("package", "manifest", "receipt-root", "study", "lock", "output"):
        cli.add_argument("--" + field, type=Path, required=True)
    cli.add_argument("--threads", type=int, default=3)
    run(cli.parse_args())
