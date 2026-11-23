from pathlib import Path

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")

from research.reference_classical import grid_error
from research.reference_compare import difference, register, regions
from research.reference_data import nuisance, pairs, project, random_stream
from research.reference_metrics import calibration_split, pixel_metrics, summary
from research.reference_siamese import SiameseNet


def test_nuisance_has_reproducible_independent_draws_and_warped_mask():
    photo = np.full((240, 150, 3), 215, dtype=np.uint8)
    mask = np.zeros(photo.shape[:2], np.uint8)
    mask[80:90, 60:78] = 1
    a = nuisance(photo, mask, random_stream("source", 0, "transformed", 42), "transformed")
    b = nuisance(photo, mask, random_stream("source", 0, "transformed", 42), "transformed")
    c = nuisance(photo, mask, random_stream("source", 1, "transformed", 42), "transformed")
    assert all(np.array_equal(first, second) for first, second in zip(a, b))
    assert not np.array_equal(a[0], c[0])
    assert np.array_equal(a[1], project(mask, a[3], mask.shape, nearest=True))
    assert a[1].any() and set(np.unique(a[2])) <= {0, 1}


def test_registration_geometry_valid_support_and_failure():
    rng = np.random.default_rng(4)
    reference = np.full((280, 200, 3), 240, dtype=np.uint8)
    for _ in range(80):
        x, y = rng.integers(15, 170), rng.integers(15, 250)
        cv2.circle(reference, (int(x), int(y)), int(rng.integers(2, 7)), (20, 20, 20), -1)
    warp = np.float64([[1, 0, 7], [0, 1, -4], [0, 0, 1]])
    candidate = project(reference, warp, reference.shape[:2], reflect=True)
    aligned = register(reference, candidate)
    assert aligned.accepted
    assert grid_error(aligned.matrix, warp, reference.shape[:2]) < 1
    assert not aligned.valid[:, -1].any()
    assert not register(np.zeros_like(reference), candidate).accepted


def test_spatial_difference_and_invalid_overlap():
    original = np.full((48, 48, 3), 230, dtype=np.uint8)
    edited = original.copy()
    edited[20:30, 20:30] = 20
    edited[:, :4] = 0
    valid = np.ones(original.shape[:2], dtype=bool)
    valid[:, :4] = False
    heat = difference(original, edited, valid, "rgb")
    predicted = regions(heat, valid, .1)
    target = np.zeros_like(valid)
    target[20:30, 20:30] = True
    assert not heat[:, :4].any() and not predicted[:, :4].any()
    assert pixel_metrics(predicted, target, valid)["iou"] > .65


def test_paired_model_is_symmetric_and_masks_invalid_regions():
    torch.manual_seed(42)
    model = SiameseNet().eval()
    a = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
    b = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
    valid = torch.ones(1, 32, 32, dtype=torch.uint8)
    valid[:, :, :8] = 0
    forward = model(a, b, valid)
    reverse = model(b, a, valid)
    for left, right in zip(forward, reverse):
        torch.testing.assert_close(left, right)
    assert not forward[2][:, :, :2].any()


def test_no_final_split_and_missing_reference(tmp_path):
    path = tmp_path / "manifest.csv"
    assert not path.exists()
    with pytest.raises(ValueError, match="development partitions"):
        next(pairs(path, tmp_path, "test", 1))
    path.write_text("split,source_document_id,source_dataset,label,image_path,mask_path,original_image_path\n"
                    "train,x,fixture,genuine,missing.png,,\n"
                    "train,x,fixture,manipulated,edit.png,mask.png,missing.png\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing.png"):
        next(pairs(path, tmp_path, "train", 1))


def test_calibration_keeps_source_pairs_and_rejects_failed_alignment():
    records = []
    for i in range(6):
        for label in (0, 1):
            records.append({"source": str(i), "label": label, "score": float(label), "accepted": i % 2 == 0,
                            "milliseconds": 10, "iou": 0., "pixel_f1": 0.})
    calibration = calibration_split([x["source"] for x in records])
    assert np.all(calibration[::2] == calibration[1::2])
    result = summary(records, calibration)
    assert result["auroc"] == 1 and result["abstention_rate"] == .5
    assert result["risk_coverage"][0]["coverage"] == .5
    assert result["risk_coverage"][-1]["coverage"] <= .5
