from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from research.mil_bags import load_bag, make_encoder
from research.mil_heads import BagAttention, normalized_centers, spatial_groups
from research.patch_data import check_source_splits, tile_boxes
from research.patch_probe import PatchNet


def test_variable_length_padding_normalization_and_gradient():
    torch.manual_seed(42)
    features = torch.randn(2, 9, 5, requires_grad=True)
    mask = torch.tensor([[True] * 9, [True] * 4 + [False] * 5])
    head = BagAttention(5)
    scores, weights = head(features, mask)
    short, _ = head(features[1, :4])
    torch.testing.assert_close(short, scores[1])
    torch.testing.assert_close(weights.sum(-1), torch.ones(2))
    assert not weights[1, 4:].any()
    scores.sum().backward()
    assert features.grad[1, :4].abs().sum() > 0
    assert not features.grad[1, 4:].any()
    assert all(parameter.grad is not None for parameter in head.parameters())
    with pytest.raises(ValueError, match="valid instance"):
        head(features[0], torch.zeros(9, dtype=torch.bool))


def test_large_bags_and_permutation_invariance():
    torch.manual_seed(11)
    boxes = np.asarray(tile_boxes((800, 1500), 112, 84))
    groups = spatial_groups(boxes, (800, 1500))
    features = 50 * torch.randn(len(groups), 8)
    permutation = torch.randperm(len(groups))
    model = BagAttention(8)
    for call in (lambda x, _: model(x), model.grouped):
        score, weights = call(features, groups)
        permuted, reordered = call(features[permutation], groups[permutation])
        torch.testing.assert_close(score, permuted)
        torch.testing.assert_close(weights[permutation], reordered)
        torch.testing.assert_close(weights.sum(), torch.tensor(1.))
        assert torch.isfinite(weights).all()


def test_spatial_coordinates_clip_padded_page_edges():
    boxes = np.array(tile_boxes((40, 286), 112, 84))
    coords = normalized_centers(boxes, (40, 286))
    assert np.all(coords[:, 0] == .5)
    assert np.all((coords > 0) & (coords < 1))
    groups = spatial_groups(boxes, (40, 286))
    assert groups.tolist() == [0, 0, 1, 1]


def test_frozen_cache_independent_of_validation_masks(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(PatchNet().state_dict(), checkpoint)
    device = torch.device("cpu")
    encoder, identity = make_encoder("patchnet", checkpoint, device, 42)
    assert not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    Image.new("RGB", (80, 157), "red").save(tmp_path / "receipt.png")
    row = {"split": "validation", "label": "manipulated", "source_document_id": "source",
           "source_dataset": "fixture", "image_path": "receipt.png", "mask_path": "missing.png"}
    documents = SimpleNamespace(root=tmp_path, rows=[row])
    bag = load_bag(encoder, documents, 0, identity, "manifest", tmp_path, device)
    assert bag.label == 1 and bag.source == "source"
    assert bag.features.shape == (2, 64) and not bag.features.requires_grad
    repeated = load_bag(encoder, documents, 0, identity, "manifest", tmp_path, device)
    assert repeated.path == bag.path
    changed = load_bag(encoder, documents, 0, {**identity, "preprocessing": "new"}, "manifest", tmp_path, device)
    assert changed.path != bag.path
    with pytest.raises(ValueError, match="more than one partition"):
        check_source_splits(documents, documents)
    row["split"] = "test"
    with pytest.raises(ValueError, match="development partitions"):
        load_bag(encoder, documents, 0, identity, "manifest", tmp_path, device)
