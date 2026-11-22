from types import SimpleNamespace

import numpy as np
import pytest

from research.diagnose import change_stats, score_diagnostics, select_pairs


def pair_rows(corpus, source):
    original = f"{source}.png"
    provenance = dict(source_dataset=corpus, source_document_id=source)
    return [
        dict(provenance, label="genuine", image_path=original),
        dict(provenance, label="manipulated", original_image_path=original),
    ]


def test_pair_sampling_is_deterministic_and_checks_counterparts():
    dataset = SimpleNamespace(rows=pair_rows("cord", "a") + pair_rows("cord", "b") + pair_rows("sroie", "c"))
    chosen = select_pairs(dataset, 2, 17)
    assert chosen == select_pairs(dataset, 2, 17)
    assert {dataset.rows[edit]["source_dataset"] for _, edit in chosen} == {"cord", "sroie"}
    dataset.rows[-1]["original_image_path"] = "wrong.png"
    with pytest.raises(ValueError, match="counterpart mismatch"):
        select_pairs(dataset, 2, 17)


def test_mask_statistics_and_score_ranking():
    delta = np.array([[1, 0], [0, 7]])
    mask = np.array([[False, False], [False, True]])
    stats = change_stats(delta, mask)
    assert stats["changed_inside"] == stats["changed_outside"] == 1
    assert stats["inside_delta_fraction"] == 7 / 8

    outputs = (
        np.array([-.5, .4]), np.array([np.full((2, 2), -1), np.where(mask, 1, -1)]),
        np.array([np.zeros((2, 2)), mask]), np.array([0, 1]),
    )
    raw, scaled = score_diagnostics(outputs), score_diagnostics(outputs, 8., .01)
    assert raw["roc_auc"] == scaled["roc_auc"] == 1
    assert raw["mean_mask_iou"] == 1
    assert raw["predicted_positive_fraction_by_class"] == {"genuine": 0, "manipulated": .25}
