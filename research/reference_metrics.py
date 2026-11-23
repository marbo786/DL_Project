"""Source-ordered scoring, separated calibration, and localization metrics."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def classify(scores, labels, threshold):
    prediction = scores >= threshold
    reference = labels.astype(bool)
    return {"recall": float(prediction[reference].mean()), "fpr": float(prediction[~reference].mean()),
            "precision": float(reference[prediction].mean()) if prediction.any() else 0.,
            "macro_f1": float(f1_score(reference, prediction, average="macro", zero_division=0))}


def reliability(probability, labels):
    group = np.minimum((probability * 10).astype(int), 9)
    return float(sum(np.mean(group == i) * abs(probability[group == i].mean() - labels[group == i].mean())
                     for i in np.unique(group)))


def calibration_split(sources):
    unique = list(dict.fromkeys(sources))
    if len(unique) < 4:
        raise ValueError("A calibration experiment needs at least four source pairs")
    size = min(64, len(unique) // 3)
    selected = set(unique[:size])
    return np.asarray([source in selected for source in sources])


def summary(rows, calibration):
    scores = np.asarray([item["score"] for item in rows])
    labels = np.asarray([item["label"] for item in rows])
    accepted = np.asarray([item["accepted"] for item in rows], dtype=bool)
    evaluation = ~calibration
    raw, truth = scores[evaluation], labels[evaluation]
    cal_score, cal_label = scores[calibration], labels[calibration]
    spread = max(float(cal_score.std()), 1e-6)
    middle = float(cal_score.mean())
    model = LogisticRegression().fit(((cal_score - middle) / spread)[:, None], cal_label)
    probability = model.predict_proba(((raw - middle) / spread)[:, None])[:, 1]
    keep = accepted[evaluation]
    operating = {}
    for target in (.05, .10, .20):
        threshold = np.quantile(cal_score[cal_label == 0], 1 - target, method="higher")
        point = classify(raw, truth, np.nextafter(threshold, np.inf))
        if len(np.unique(truth[keep])) == 2:
            point["aligned_only"] = classify(raw[keep], truth[keep], np.nextafter(threshold, np.inf))
        operating[str(target)] = point
    risk = []
    for confidence in (.5, .6, .7, .8, .9, .95):
        present = keep & (np.maximum(probability, 1 - probability) >= confidence)
        risk.append({"minimum_probability": confidence, "coverage": float(present.mean()),
                     "error": float(np.mean((probability[present] >= .5) != truth[present])) if present.any() else None})
    edits = [item for item, use in zip(rows, evaluation) if use and item["label"] == 1]
    times = [item["milliseconds"] for item, use in zip(rows, evaluation) if use]
    return {"calibration_pairs": int(calibration.sum() // 2), "evaluation_pairs": int(evaluation.sum() // 2),
            "auroc": float(roc_auc_score(truth, raw)), "average_precision": float(average_precision_score(truth, raw)),
            "calibrated_metrics": classify(probability, truth, .5), "operating_points": operating,
            "brier": float(np.mean((probability - truth) ** 2)), "ece_10_bins": reliability(probability, truth),
            "abstention_rate": float((~keep).mean()), "risk_coverage": risk,
            "pixel_iou": float(np.mean([e["iou"] for e in edits])), "localization_f1": float(np.mean([e["pixel_f1"] for e in edits])),
            "latency_median_ms": float(np.median(times)), "latency_p95_ms": float(np.quantile(times, .95))}


def pixel_metrics(predicted, target, valid):
    a, b = predicted & valid, (target > 0) & valid
    overlap = int((a & b).sum())
    return {"iou": overlap / max(int((a | b).sum()), 1),
            "pixel_f1": 2 * overlap / max(int(a.sum() + b.sum()), 1)}
