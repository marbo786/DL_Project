import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from research.receipt_independent_metrics import intervals, operating_point, ranking


def test_rank_metrics_match_sklearn_with_repeated_group_and_ties():
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0., 1., 2., 3., 1., 1.])
    weights = np.array([2, 2, 1, 1, 0, 0])
    assert ranking(labels, scores) == pytest.approx(
        (roc_auc_score(labels, scores), average_precision_score(labels, scores)))
    repeated = np.repeat(np.arange(len(labels)), weights)
    assert ranking(labels, scores, weights) == pytest.approx(
        (roc_auc_score(labels[repeated], scores[repeated]),
         average_precision_score(labels[repeated], scores[repeated])))
    assert ranking(labels[:4], scores[:4])[0] == .75
    assert np.all(scores[:4:2] < scores[1:4:2])


def test_paired_bootstrap_uses_same_group_draw_and_strict_threshold():
    groups = np.array(["a", "a", "b", "b"])
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0., 1., 2., 3.])
    result = intervals(groups, labels, scores, 1., alternate=scores, repeats=100)
    assert result["auroc"] == [.75, 1.]
    assert result["paired_auroc_delta"] == [0., 0.]
    assert operating_point(labels, scores, 1.)["confusion"] == [[1, 1], [1, 1]]
    assert operating_point(labels, scores, 1.)["threshold_ties"] == 1
