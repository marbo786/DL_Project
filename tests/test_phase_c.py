from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from research.hard_negatives import BackgroundTiles, require_training, select_background
from research.patch_data import check_source_splits, tile_boxes
from research.phase_c import localization, paired_intervals, pooling


def fixture(size, split="train"):
    original = Image.new("RGB", size, (55, 80, 110))
    edited = original.copy()
    edited.putpixel((3, 4), (0, 0, 0))
    mask = Image.new("L", size)
    mask.putpixel((3, 4), 255)
    rows = [
        {"split": split, "source_document_id": "a", "source_dataset": "fixture", "label": "genuine",
         "image_path": "genuine.png"},
        {"split": split, "source_document_id": "a", "source_dataset": "fixture", "label": "manipulated",
         "original_image_path": "genuine.png"},
    ]
    data = [(original, Image.new("L", size), False), (edited, mask, True)]
    return SimpleNamespace(rows=rows, load=data.__getitem__)


def test_labels_alignment_and_source_determinism():
    docs = fixture((103, 53))
    a, b = (BackgroundTiles(docs, 1, 42, 32, 24, .0001, 2) for _ in range(2))
    assert a.samples == b.samples
    _, positive, backgrounds = a.samples[0]
    grid = tile_boxes((103, 53), 32, 24)
    assert positive in grid and all(box in grid for box, _ in backgrounds)
    images, labels = a[0]
    assert labels.tolist() == [0, 1, 0, 0, 0, 0]
    assert (images[0] != images[1]).any(dim=0).nonzero().tolist() == [[4, 3]]
    assert torch.equal(images[2], images[3]) and torch.equal(images[4], images[5])


def test_training_guard_overlap_and_empty_background():
    docs = fixture((22, 24))
    with pytest.raises(ValueError, match="training partition"):
        require_training(fixture((22, 24), split="validation"))
    with pytest.raises(ValueError, match="more than one partition"):
        check_source_splits(docs, fixture((22, 24), "validation"))
    data = BackgroundTiles(docs, 1, 42, 32, 24, .0001, 2)
    images, labels = data[0]
    assert images.shape == (2, 3, 32, 32)
    assert labels.tolist() == [0, 1]
    assert data.counts[0]["background_candidate_locations"] == 0


def test_bounded_mining_retains_random_slots_and_requires_scores():
    candidates = [(i, 0, i + 1, 1) for i in range(4)]
    result = select_background(candidates, 2, 1, 42, [0, 1, 8, 3])
    assert len({box for box, _ in result}) == 3
    assert result[-1] == (candidates[2], "hard")
    assert sum(kind == "random" for _, kind in result) == 2
    assert select_background(candidates[:1], 1, 1, 42, [9]) == [(candidates[0], "random")]
    with pytest.raises(ValueError, match="requires training predictions"):
        select_background(candidates, 1, 1, 42, None)


def test_localization_and_count_normalized_pooling():
    result = localization(np.array([5., 4., 3., 2.]), np.array([0, 3, 0, 0]),
                          np.array([8, 8, 8, 8]), 5, ranks=(1, 2, 5))
    assert result["hit_at_1"] == 0
    assert result["hit_at_2"] == result["hit_at_5"] == 1
    assert result["uniform_hit_at_1"] == .25
    assert result["uniform_hit_at_2"] == .5
    assert result["uniform_hit_at_5"] == 1
    assert pooling(np.asarray([2])) == [2., 2., 2., 2.]
    assert pooling(np.arange(100))[-1] == 97.
    pairs = np.array([[0, 3], [1, 4]])
    assert paired_intervals(pairs, pairs, draws=10) == [0, 0]
