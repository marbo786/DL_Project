import csv
from io import BytesIO
import json
from zipfile import ZipFile

import pytest
from PIL import Image

from research.aiforge import partition, prepare
from research.manifest import audit


def image_bytes(color):
    buffer = BytesIO()
    Image.new("L", (20, 20), color).save(buffer, format="PNG")
    return buffer.getvalue()


def write_fixture(path, originals=True):
    selected = {
        split: next(f"cord:{i}" for i in range(50) if partition(f"cord:{i}", "cord", set(), "wildreceipt") == split)
        for split in ("train", "validation")
    }
    rows = [("cord", source.split(":")[1], "training") for source in selected.values()]
    rows += [("cord", "test", "testing"), ("wildreceipt", "shifted", "training")]
    metadata = []
    with ZipFile(path, "w") as archive:
        for index, (corpus, source, split) in enumerate(rows):
            directory = "TrainingSet" if split == "training" else "TestingSet"
            name = f"{index:09}.png"
            forged, mask = f"{directory}/images/{name}", f"{directory}/masks/{name}"
            metadata.append({
                "source_dataset": corpus, "image_id": source, "doc_type": "receipt", "spec_id": str(index),
                "assigned_tool": "fixture-generator", "split": split, "image": forged, "mask": mask,
            })
            archive.writestr("dataset/" + forged, image_bytes(10 + index))
            archive.writestr("dataset/" + mask, image_bytes(255))
            if originals:
                archive.writestr(f"dataset/{directory}/authentic/{name}", image_bytes(20 + index))
        archive.writestr("dataset/metadata.jsonl", "\n".join(map(json.dumps, metadata)))


def test_paired_manifest_and_audit(tmp_path):
    source = tmp_path / "benchmark.zip"
    write_fixture(source)
    parent = tmp_path / "import"
    manifest, counts = prepare(source, parent, "wildreceipt")
    assert counts == dict.fromkeys(["train", "validation", "test", "shifted_test"], 1)
    assert audit(manifest, parent / "assets") == []
    with manifest.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 8


def test_hugging_face_partial_archive_is_rejected(tmp_path):
    source = tmp_path / "partial.zip"
    write_fixture(source, originals=False)
    parent = tmp_path / "import"
    with pytest.raises(ValueError, match="paired authentic"):
        prepare(source, parent, "wildreceipt")
    assert not parent.exists()


def test_published_test_takes_priority_over_validation():
    assert partition("cord:1", "cord", {"cord:1"}, "wildreceipt") == "test"
    assert partition("wildreceipt:1", "wildreceipt", {"wildreceipt:1"}, "wildreceipt") == "shifted_test"
