"""Evaluate single-image scores with source-group uncertainty and calibration."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from research.manifest import file_sha256


def read_scores(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["sample_id"] for row in rows]
    groups = np.array([row["source_group"] for row in rows])
    labels = np.array([int(row["label"]) for row in rows])
    logits = np.array([float(row["logit"]) for row in rows])
    if len(ids) != len(set(ids)) or "" in groups or set(labels) != {0, 1} or not np.isfinite(logits).all():
        raise ValueError("Scores must have unique IDs, source groups, both classes and finite logits")
    return rows, groups, labels, logits


def reliability(y, probability):
    index = np.minimum((probability * 10).astype(int), 9)
    bins = []
    for bucket in range(10):
        chosen = index == bucket
        if chosen.any():
            bins.append({"bin": bucket, "count": int(chosen.sum()),
                         "mean_probability": float(probability[chosen].mean()),
                         "observed_fraction": float(y[chosen].mean())})
    return {
        "brier": float(np.mean((y - probability) ** 2)), "bins": bins,
        "ece": sum(b["count"] * abs(b["mean_probability"] - b["observed_fraction"]) for b in bins) / len(y),
    }


def risk_coverage(y, probability):
    predictions = probability >= .5
    confidence = np.maximum(probability, 1 - probability)
    rows = []
    for cutoff in sorted(set(np.linspace(.5, 1., 21)) | set(np.quantile(confidence, np.linspace(0., 1., 21)))):
        accepted = confidence >= cutoff
        rows.append({"confidence": float(cutoff), "coverage": float(accepted.mean()),
                     "coverage_authentic": float(accepted[y == 0].mean()),
                     "coverage_edited": float(accepted[y == 1].mean()),
                     "accepted_error": float((predictions[accepted] != y[accepted]).mean()) if accepted.any() else None})
    return rows


def frozen_thresholds(labels, logits):
    authentic = logits[labels == 0]
    return {str(target): float(np.quantile(authentic, 1 - target, method="higher"))
            for target in (.01, .05, .10)}


def classification(y, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel().tolist()
    return {"confusion": [[tn, fp], [fn, tp]], "fpr": fp / (tn + fp),
            "recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else None,
            "macro_f1": float(f1_score(y, predicted, average="macro"))}


def bootstrap(groups, y, logits, repeats, paired=None, iou=None, threshold=None):
    index = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    rng = np.random.default_rng(235)
    values = []
    for _ in range(repeats):
        take = np.concatenate([index[i] for i in rng.integers(len(index), size=len(index))])
        labels, scores = y[take], logits[take]
        if np.unique(labels).size != 2:
            continue
        roc = roc_auc_score(labels, scores)
        if paired is not None:
            values.append([roc - roc_auc_score(labels, paired[take])])
        else:
            values.append([roc, average_precision_score(labels, scores),
                           iou[take][labels == 1].mean(), (scores[labels == 1] > threshold).mean(),
                           (scores[labels == 0] > threshold).mean()])
    names = ["paired_auroc_delta"] if paired is not None else ["auroc", "ap", "iou", "recall_at_5pct", "fpr_at_5pct"]
    return dict(zip(names, np.quantile(values, [.025, .975], axis=0).T.tolist())) if values else {}


def evaluate(calibration, evaluation, output, repeats=500, compare=None):
    _, calibration_groups, calibration_y, calibration_logit = read_scores(calibration)
    rows, groups, y, logits = read_scores(evaluation)
    if set(calibration_groups) & set(groups):
        raise ValueError("Calibration and evaluation share source groups")
    calibrator = LogisticRegression(C=1.).fit(calibration_logit[:, None], calibration_y)
    p = calibrator.predict_proba(logits[:, None])[:, 1]
    constant = np.full(y.shape, calibration_y.mean())
    thresholds = frozen_thresholds(calibration_y, calibration_logit)
    iou = np.array([float(row["iou"]) if row["iou"] else 0. for row in rows])
    hits = np.array([int(row["hit"]) if row["hit"] else 0 for row in rows])
    joined = [{**row, "calibrated_probability": float(probability)} for row, probability in zip(rows, p)]
    ranked = sorted(joined, key=lambda row: float(row["logit"]))
    times = [float(row["milliseconds"]) for row in rows if row["milliseconds"]]
    report = {
        "source_groups": len(set(groups)), "documents": len(y), "calibration_groups": len(set(calibration_groups)),
        "calibration_sha256": file_sha256(calibration), "evaluation_sha256": file_sha256(evaluation),
        "calibration": {"slope": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0]),
                        "prevalence": float(calibration_y.mean())},
        "auroc": float(roc_auc_score(y, logits)), "ap": float(average_precision_score(y, logits)),
        "mean_edit_iou": float(iou[y == 1].mean()), "edit_hit_rate": float(hits[y == 1].mean()),
        "classification_at_probability_half": classification(y, p >= .5),
        "operating_points": {key: {"logit_threshold": cut, **classification(y, logits > cut)}
                             for key, cut in thresholds.items()},
        "reliability": reliability(y, p), "constant_reliability": reliability(y, constant),
        "risk_coverage": risk_coverage(y, p), "constant_risk_coverage": risk_coverage(y, constant),
        "bootstrap_repetitions": repeats,
        "source_bootstrap_95": bootstrap(groups, y, logits, repeats, iou=iou, threshold=thresholds["0.05"]),
        "highest_scoring_authentic": [row for row in ranked[::-1] if row["label"] == "0"][:5],
        "lowest_scoring_edits": [row for row in ranked if row["label"] == "1"][:5],
        "processing_ms": np.quantile(times, [.5, .95]).tolist() if times else None,
    }
    if compare:
        other, _, _, _ = read_scores(compare)
        by_sample = {row["sample_id"]: row for row in other}
        if set(by_sample) != {row["sample_id"] for row in rows}:
            raise ValueError("Paired score tables must contain the same sample IDs")
        matches = [by_sample[row["sample_id"]] for row in rows]
        for first, second in zip(rows, matches):
            if any(first[key] != second[key] for key in ("label", "source_group")):
                raise ValueError("Paired predictions disagree on labels or source groups")
        paired = np.array([float(row["logit"]) for row in matches])
        report["paired"] = {"sha256": file_sha256(compare),
                            "auroc_delta": report["auroc"] - float(roc_auc_score(y, paired)),
                            "source_bootstrap_95": bootstrap(groups, y, logits, repeats, paired=paired)}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--compare", type=Path)
    evaluate(**vars(parser.parse_args()))
