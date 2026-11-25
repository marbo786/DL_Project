import csv
from io import BytesIO
import json
from math import exp
import shutil

from PIL import Image
import pytest
import torch

from research.manifest import file_sha256, resolve_image
from research.receipt.demo import FrozenReceiptDemo, LOCK_PATH, ROOT, RUN_PATH, decode_upload, route_receipt


@pytest.mark.parametrize(("logit", "probability", "outcome"), [
    (2., .56, "Prioritize review"),
    (.5, .43, "Normal queue"),
    (2., .51, "Inconclusive"),
    (1., .56, "Inconclusive"),
])
def test_routing(logit, probability, outcome):
    assert route_receipt(logit, probability, 1.)[0] == outcome


def test_upload_decode_applies_exif_orientation():
    image = Image.new("RGB", (10, 20))
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    assert decode_upload(buffer.getvalue()).size == (20, 10)
    with pytest.raises(ValueError, match="readable"):
        decode_upload(b"not an image")


def test_changed_freeze_file_is_rejected(tmp_path):
    lock = tmp_path / LOCK_PATH
    lock.parent.mkdir(parents=True)
    lock.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="demo lock digest"):
        FrozenReceiptDemo(tmp_path)


@pytest.mark.artifacts(LOCK_PATH, RUN_PATH / "fit.json")
def test_modified_calibration_is_rejected(tmp_path):
    (tmp_path / LOCK_PATH).parent.mkdir(parents=True)
    (tmp_path / RUN_PATH).mkdir(parents=True)
    shutil.copyfile(ROOT / LOCK_PATH, tmp_path / LOCK_PATH)
    shutil.copyfile(ROOT / RUN_PATH / "fit.json", tmp_path / RUN_PATH / "fit.json")
    (tmp_path / RUN_PATH / "calibration.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration table"):
        FrozenReceiptDemo(tmp_path)


@pytest.mark.data("data/aiforge/assets", "artifacts/receipt-protocol-v1-contained/manifest.csv")
@pytest.mark.artifacts(
    LOCK_PATH, RUN_PATH / "fit.json", RUN_PATH / "model.pt", RUN_PATH / "calibration.csv",
    RUN_PATH / "validation.csv", RUN_PATH / "metrics.json",
)
def test_demo_matches_stored_development_scores():
    asset_root = ROOT / "data/aiforge/assets"
    manifest = ROOT / "artifacts/receipt-protocol-v1-contained/manifest.csv"
    run = ROOT / RUN_PATH
    validation = run / "validation.csv"
    assert file_sha256(validation) == json.loads((run / "metrics.json").read_text())["evaluation_sha256"]
    with validation.open(newline="", encoding="utf-8") as stream:
        saved = {row["sample_id"]: row for row in csv.DictReader(stream)}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "validation"]
    groups = sorted(rows, key=lambda row: (int(row["image_width"]) * int(row["image_height"]), row["source_group"]))
    chosen = set(list(dict.fromkeys(row["source_group"] for row in groups))[:2])
    examples = [row for row in rows if row["source_group"] in chosen]
    assert len(examples) == 4 and {row["label"] for row in examples} == {"genuine", "manipulated"}

    torch.set_num_threads(2)
    demo = FrozenReceiptDemo()
    lock = json.loads((ROOT / LOCK_PATH).read_text())
    maximum_error = 0.
    for example in examples:
        expected = saved[example["sample_id"]]
        expected_logit = float(expected["logit"])
        parameters = lock["calibration"]
        expected_p = 1 / (1 + exp(-(parameters["slope"] * expected_logit + parameters["intercept"])))
        image = decode_upload(resolve_image(asset_root, example["image_path"]).read_bytes())
        result = demo.score(image)
        assert result.logit == pytest.approx(expected_logit, abs=1e-5, rel=0)
        assert result.probability == pytest.approx(expected_p, abs=1e-6, rel=0)
        assert result.confidence == pytest.approx(max(expected_p, 1 - expected_p), abs=1e-6, rel=0)
        assert result.tile == tuple(map(int, expected["bbox"].split()))
        assert result.decision == "Inconclusive"
        maximum_error = max(maximum_error, abs(result.logit - expected_logit))
    print(f"Development parity: {len(examples)} receipts, max absolute logit difference {maximum_error:.9g}")
