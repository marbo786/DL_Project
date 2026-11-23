from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from research.patch_data import MatchedCrops, check_source_splits, oracle_boxes, rgb_crop, supervised_tiles, tile_boxes
from research.patch_probe import metrics, pool


def test_oracle_covers_large_non_square_edit_touching_edge():
    mask = Image.new("L", (430, 173))
    ImageDraw.Draw(mask).rectangle((103, 40, 429, 172), fill=255)
    boxes = oracle_boxes(mask, 112)
    assert boxes == oracle_boxes(mask, 112)
    covered = Image.new("L", mask.size)
    for box in boxes:
        covered.paste(mask.crop(box), box)
    assert np.array_equal(np.asarray(mask), np.asarray(covered))


def test_aligned_rgb_and_edge_padding():
    original = Image.new("RGB", (35, 64), (15, 22, 99))
    edited = original.copy()
    edited.putpixel((34, 50), (32, 44, 55))
    box = tile_boxes(original.size, 64, 32)[0]
    authentic, manipulated = (rgb_crop(image, box) for image in (original, edited))
    assert authentic.shape == (3, 64, 64)
    assert authentic[:, 49, 63].tolist() == manipulated[:, 49, 63].tolist() == [15, 22, 99]
    assert manipulated[:, 50, 34].tolist() == [32, 44, 55]
    assert manipulated[:, 50, 63].tolist() == [32, 44, 55]


def test_tiles_reach_page_boundary_and_reject_sparse_supervision():
    boxes = tile_boxes((251, 111), 64, 32)
    assert boxes[-1] == (187, 47, 251, 111)
    mask = Image.new("L", (251, 111))
    mask.putpixel((250, 110), 255)
    assert supervised_tiles(mask, 64, 32, .01) == []
    assert supervised_tiles(mask, 64, 32, 1 / 64 ** 2) == [boxes[-1]]


def test_split_overlap_and_paired_metrics():
    dataset = SimpleNamespace(rows=[{"source_document_id": "receipt-1"}])
    with pytest.raises(ValueError, match="more than one partition"):
        check_source_splits(dataset, dataset)
    assert metrics([[-2, 3], [-1, 4]])["auroc"] == 1
    assert metrics([[0, 0]])["paired_win_rate"] == .5
    assert pool(np.array([1, 2, 3, 4])) == [4., 3., 2.5]


def test_matched_dataset_uses_the_same_spatial_crop_for_both_classes():
    original = Image.new("RGB", (200, 150), (20, 80, 100))
    edited = original.copy()
    edited.putpixel((180, 100), (99, 12, 30))
    mask = Image.new("L", original.size)
    mask.putpixel((180, 100), 255)
    images = [(original, Image.new("L", original.size), False), (edited, mask, True)]
    rows = [
        {"source_dataset": "fixture", "source_document_id": "source", "label": "genuine", "image_path": "original.png"},
        {"source_dataset": "fixture", "source_document_id": "source", "label": "manipulated",
         "original_image_path": "original.png"},
    ]
    docs = SimpleNamespace(rows=rows, load=images.__getitem__)
    data = MatchedCrops(docs, 1, 42, 112)
    assert len(data) == 1
    pair = data[0]
    delta = (pair[0] != pair[1]).any(dim=0).nonzero()
    assert delta.shape == (1, 2)
    assert data.audit["changed_inside"] == [1]
    assert data.audit["changed_outside"] == [0]
