import csv
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image, PngImagePlugin
import pytest
import torch

from research.manifest import file_sha256
from research.receipt.metrics import bootstrap, frozen_thresholds, reliability, risk_coverage
from research.receipt.model import Detector, disturb, inference_tiles, load_detector, predict
from research.receipt.protocol import freeze, pixel_digest, verify_groups
from research.receipt.trufor import collect, input_name, read_map, stage


def test_decoded_duplicates_quarantine_connected_historical_roles(tmp_path):
    rows = []
    for source, split in (("one", "train"), ("two", "validation")):
        authentic = Image.new("RGB", (16, 20), "white")
        edited = authentic.copy()
        edited.putpixel((2, 3), (0, 0, 0))
        mask = Image.new("L", authentic.size)
        mask.putpixel((2, 3), 255)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("identity", source)
        original_name, mask_name = f"{source}-original.png", f"{source}-mask.png"
        authentic.save(tmp_path / original_name, pnginfo=metadata)
        mask.save(tmp_path / mask_name)
        for label, image in (("genuine", authentic), ("manipulated", edited)):
            name = original_name if label == "genuine" else f"{source}-{label}.png"
            image.save(tmp_path / name, pnginfo=metadata)
            rows.append({
                "source_document_id": source, "source_dataset": "test", "split": split, "sample_id": name,
                "label": label, "generator": "fixture" if label == "manipulated" else "", "image_path": name,
                "image_width": 16, "image_height": 20, "mask_path": mask_name,
                "original_image_path": original_name,
            })
    manifest = tmp_path / "in.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    assert (tmp_path / "one-original.png").read_bytes() != (tmp_path / "two-original.png").read_bytes()
    assert pixel_digest(authentic) != pixel_digest(edited)
    freeze(manifest, tmp_path, tmp_path / "out")
    with (tmp_path / "out" / "manifest.csv").open(newline="") as stream:
        result = list(csv.DictReader(stream))
    assert {row["split"] for row in result} == {"quarantined"}
    assert {row["source_group"] for row in result} == {"one"}


def test_inference_geometry_and_symmetric_noise(tmp_path):
    first = Image.new("RGB", (130, 56), (50, 50, 50))
    second = Image.new("RGB", first.size, (90, 90, 90))
    boxes, tensors = inference_tiles(first, "full")
    assert boxes == inference_tiles(second, "full")[0]
    assert tuple(tensors.shape) == (2, 3, 112, 112)
    original = np.asarray(disturb(first, "noise-1", "source"), dtype=int)
    edited = np.asarray(disturb(second, "noise-1", "source"), dtype=int)
    np.testing.assert_array_equal(edited - original, 40)
    torch.set_num_threads(2)
    detector = Detector("partial").eval()
    model_path = tmp_path / "model.pt"
    torch.save(detector.state_dict(), model_path)
    (tmp_path / "fit.json").write_text(json.dumps({
        "mode": "partial", "checkpoint_sha256": file_sha256(model_path),
    }), encoding="utf-8")
    clone = load_detector(tmp_path)
    score, region = predict(detector, first, 4)
    clone_score, clone_region = predict(clone, first, 4)
    assert clone_score == pytest.approx(score, abs=1e-6)
    assert region == clone_region
    assert all(not block.training for block in detector.train().features[:detector.frozen_blocks])


def test_uncertainty_and_paired_bootstrap():
    y = np.array([0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b"])
    logits = np.array([.1, .4, .2, .5])
    assert frozen_thresholds(y, logits)["0.05"] == .2
    calibrated = np.full(4, .5)
    stats = reliability(y, calibrated)
    assert stats["ece"] == 0
    assert stats["brier"] == .25
    risk = risk_coverage(y, calibrated)
    assert risk[0]["accepted_error"] == .5
    assert risk[-1]["accepted_error"] is None
    assert bootstrap(groups, y, logits, 20, paired=logits) == {"paired_auroc_delta": [0., 0.]}


def test_protocol_rejects_calibration_leak_and_invalid_external_map(tmp_path):
    table = tmp_path / "splits.csv"
    table.write_text("source_group,split\na,train\na,calibration\n", encoding="utf-8")
    with pytest.raises(ValueError, match="crosses roles"):
        verify_groups(table)
    prediction = tmp_path / "map.npz"
    np.savez(prediction, score=np.array(.3), map=np.zeros((4, 5)))
    assert read_map(prediction, (4, 5))[0] == .3
    with pytest.raises(ValueError, match="raster"):
        read_map(prediction, (5, 4))


def test_trufor_holdout(tmp_path):
    Image.new("RGB", (7, 5), "white").save(tmp_path / "form.png")
    row = {"sample_id": "form:original", "source_group": "form", "source_dataset": "xfund",
           "split": "test", "label": "genuine", "image_path": "form.png", "mask_path": ""}
    manifest = tmp_path / "holdout.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    args = SimpleNamespace(manifest=manifest, root=tmp_path, split="test", smoke=False,
                           output=tmp_path / "staged")
    stage(args)
    with Image.open(args.output / input_name(row)) as staged:
        assert staged.size == (7, 5)
    args.predictions = tmp_path / "npz"
    args.predictions.mkdir()
    np.savez(args.predictions / f"{input_name(row)}.npz", score=np.array(.3), map=np.zeros((5, 7)))
    args.weights = tmp_path / "fixture-checkpoint"
    args.weights.touch()
    args.output = tmp_path / "scores.csv"
    collect(args)
    with args.output.open(newline="", encoding="utf-8") as stream:
        result, = csv.DictReader(stream)
    assert result["sample_id"] == row["sample_id"]
    assert result["bbox"] == "0 0 7 5"
