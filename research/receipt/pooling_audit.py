"""Measure how often edit-site evidence survives ordinary document pooling on development."""
import argparse
import csv
from pathlib import Path

import torch

from research.data import ManifestDocuments
from research.manifest import file_sha256
from research.receipt.ablations import read_json, write_json, write_rows
from research.receipt.demo import FrozenReceiptDemo, ROOT, RUN_PATH
from research.receipt.explain import pair_audit


def summarize(rows):
    report = {"pairs": len(rows)}
    for scope in ("edit", "background", "document"):
        values = [row[scope] for row in rows if row[scope] is not None]
        report[scope] = {"eligible": len(values), "edited_higher": sum(v > 0 for v in values),
                         "ties": sum(v == 0 for v in values)}
    return report


def run(manifest, assets, output):
    torch.set_num_threads(2)
    demo = FrozenReceiptDemo()
    documents = ManifestDocuments(manifest, assets, "validation")
    source = ROOT / RUN_PATH / "validation.csv"
    if file_sha256(source) != read_json(ROOT / RUN_PATH / "metrics.json")["evaluation_sha256"]:
        raise ValueError("Development reference scores have changed")
    with source.open(newline="") as stream:
        reference = {r["sample_id"]: r for r in csv.DictReader(stream)}
    lookup = {r["sample_id"]: i for i, r in enumerate(documents.rows)}
    if set(lookup) != set(reference):
        raise ValueError("Development inputs differ from the original cohort")
    output.mkdir(parents=True)
    write_json(output / "lock.json", {"manifest": file_sha256(manifest), "scores": file_sha256(source),
               "split": "validation", "mask_area_fraction_cutoff": .005, "tile_count_cutoff": 50})
    records = []
    for sample_id, index in lookup.items():
        if not sample_id.endswith(":original"):
            continue
        pair = (index, lookup[sample_id.split(":")[0] + ":edited"])
        evidence, image, mask = pair_audit(demo, documents, pair, reference, False)
        genuine, edited = (r for r, _ in evidence)
        record = {"sample_id": sample_id, "source_group": genuine["source_group"],
                  "mask_fraction": mask.histogram()[255] / (image.width * image.height),
                  "tiles": edited["edit_tiles"] + edited["background_tiles"]}
        for scope, field in (("edit", "edit_logit"), ("background", "background_logit"), ("document", "logit")):
            record[scope] = None if edited[field] is None else edited[field] - genuine[field]
        records.append(record)
    if len(records) * 2 != len(reference):
        raise ValueError("Incomplete pairs")
    write_rows(output / "pairs.csv", records)
    strata = {"all": records, "mask_under_0.5pct": [r for r in records if r["mask_fraction"] < .005],
              "mask_at_least_0.5pct": [r for r in records if r["mask_fraction"] >= .005],
              "tiles_under_50": [r for r in records if r["tiles"] < 50],
              "tiles_at_least_50": [r for r in records if r["tiles"] >= 50]}
    write_json(output / "summary.json", {k: summarize(v) for k, v in strata.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.assets, args.output)
