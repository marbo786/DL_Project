"""Freeze and stage the native-resolution CPU panel from audited receipt assignments."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from research.manifest import file_sha256
from research.receipt_trufor import SPLITS, stage


def build_panel(manifest, root, output):
    quotas = {role: {"cord": 6, "sroie": 4} for role in SPLITS[:-1]}
    quotas["shifted_test"] = {"wildreceipt": 10}
    by_pair = defaultdict(list)
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for row in reader:
            if row["split"] in quotas:
                key = (row["split"], row["source_dataset"], row["source_group"], row["sample_id"].split(":")[0])
                by_pair[key].append(row)

    chosen, strata = [], []
    groups_seen = set()
    for role in SPLITS:
        cap = 250_000 if role == "shifted_test" else 600_000
        for corpus, quota in quotas[role].items():
            candidates = [
                (key, rows) for key, rows in by_pair.items()
                if key[:2] == (role, corpus) and len(rows) == 2
                and {row["label"] for row in rows} == {"genuine", "manipulated"}
                and all(int(row["image_width"]) * int(row["image_height"]) <= cap for row in rows)
            ]
            candidates.sort(key=lambda pair: sha256(("trufor-native-panel-v1:" + ":".join(pair[0])).encode()).digest())
            selected = set()
            for key, rows in candidates:
                if key[2] in selected:
                    continue
                if key[2] in groups_seen:
                    raise ValueError(f"Source group {key[2]} crosses panel strata")
                selected.add(key[2])
                chosen.extend(rows)
                if len(selected) == quota:
                    break
            if len(selected) != quota:
                raise ValueError(f"Insufficient eligible source groups for {role}/{corpus}")
            groups_seen.update(selected)
            strata.append(dict(split=role, corpus=corpus, max_pixels=cap,
                               eligible_pairs=len(candidates), selected_groups=len(selected)))

    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    panel_manifest = output / "manifest.csv"
    with panel_manifest.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)
    selection = {
        "design": "deterministic native-resolution CPU panel, one balanced pair per source group, selected before panel inference",
        "full_manifest_sha256": file_sha256(manifest),
        "panel_manifest_sha256": file_sha256(panel_manifest),
        "strata": strata,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    for role in SPLITS:
        staged = output / f"inputs-{role}"
        stage(SimpleNamespace(manifest=panel_manifest, root=root, split=role, smoke=False, output=staged))
        for image in staged.glob("*.png"):
            target = inputs / image.name
            if target.exists():
                raise ValueError(f"Duplicate input {image.name}")
            shutil.move(image, target)
        staged.rmdir()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "root", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    build_panel(**vars(cli.parse_args()))
