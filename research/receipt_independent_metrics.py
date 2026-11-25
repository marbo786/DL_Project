"""Independently reconstruct receipt metrics from private prediction tables."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from research.manifest import file_sha256

ARTIFACTS = Path("artifacts")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def score_arrays(table):
    ids = [row["sample_id"] for row in table]
    group = np.array([row["source_group"] for row in table])
    labels = np.array([int(row["label"]) for row in table])
    scores = np.array([float(row["logit"]) for row in table])
    if (len(set(ids)) != len(ids) or not all(ids) or not all(group)
            or set(labels) != {0, 1} or not np.isfinite(scores).all()):
        raise ValueError("Invalid score table")
    return group, labels, scores


def ranking(labels, scores, weights=None):
    """Tie-aware weighted ROC area and noninterpolated average precision."""
    weights = np.ones(len(labels)) if weights is None else weights
    _, bucket = np.unique(scores, return_inverse=True)
    positives = np.bincount(bucket, weights * labels)
    negatives = np.bincount(bucket, weights * (1 - labels))
    p, n = positives.sum(), negatives.sum()
    if not p or not n:
        raise ValueError("Both classes are required")
    lower_negatives = np.cumsum(negatives) - negatives
    roc = positives @ (lower_negatives + .5 * negatives) / (p * n)
    cum_p = np.cumsum(positives[::-1])
    cum_total = cum_p + np.cumsum(negatives[::-1])
    precision = np.divide(cum_p, cum_total, out=np.zeros_like(cum_p), where=cum_total > 0)
    return float(roc), float(precision @ positives[::-1] / p)


def operating_point(labels, scores, cutoff):
    predicted = scores > cutoff
    tn, fp = [int(np.count_nonzero((labels == 0) & (predicted == decision))) for decision in (0, 1)]
    fn, tp = [int(np.count_nonzero((labels == 1) & (predicted == decision))) for decision in (0, 1)]
    errors = fp + fn
    return {"confusion": [[tn, fp], [fn, tp]], "fpr": fp / (tn + fp), "recall": tp / (tp + fn),
            "precision": tp / (tp + fp) if tp + fp else None,
            "macro_f1": ((2 * tn / (2 * tn + errors)) + (2 * tp / (2 * tp + errors))) / 2,
            "threshold_ties": int(np.count_nonzero(scores == cutoff))}


def intervals(groups, labels, scores, cutoff, alternate=None, calibration=None, repeats=1000):
    """Weighted counting retains bootstrap multiplicities and pairs method draws."""
    names, inverse = np.unique(groups, return_inverse=True)
    if calibration is not None:
        cal_groups, cal_values = calibration
        cal_names, cal_inverse = np.unique(cal_groups, return_inverse=True)
        cal_members = [np.flatnonzero(cal_inverse == idx) for idx in range(len(cal_names))]
    rng = np.random.default_rng(137)
    samples = []
    for _ in range(repeats):
        weight = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))[inverse]
        if not (weight @ labels and weight @ (1 - labels)):
            continue
        roc, ap = ranking(labels, scores, weight)
        point = cutoff
        if calibration is not None:
            draw = rng.integers(len(cal_names), size=len(cal_names))
            point = cutoff_for(cal_values[np.concatenate([cal_members[i] for i in draw])])
        yes = scores > point
        values = [roc, ap, weight @ (yes & (labels == 0)) / (weight @ (1 - labels)),
                  weight @ (yes & (labels == 1)) / (weight @ labels)]
        if alternate is not None:
            values.append(roc - ranking(labels, alternate, weight)[0])
        samples.append(values)
    keys = ("auroc", "ap", "fpr", "recall", "paired_auroc_delta")
    return dict(zip(keys, np.quantile(samples, [.025, .975], axis=0).T.tolist()))


def cutoff_for(values):
    return float(np.sort(values)[int(np.ceil(.95 * (len(values) - 1)))])


def assess(table, calibration, alternate=None):
    groups, labels, scores = score_arrays(table)
    cal_groups, cal_labels, cal_scores = score_arrays(calibration)
    if set(groups) & set(cal_groups):
        raise ValueError("Calibration and evaluation group overlap")
    cutoff = cutoff_for(cal_scores[cal_labels == 0])
    model = LogisticRegression(C=1).fit(cal_scores[:, None], cal_labels)
    probability = model.predict_proba(scores[:, None])[:, 1]
    prevalence = float(cal_labels.mean())
    bins = np.minimum((10 * probability).astype(int), 9)
    ece = sum((bins == idx).sum() * abs((labels - probability)[bins == idx].mean())
              for idx in set(bins)) / len(labels)
    selectivity = {}
    for level in (.5, .55, .6, .7, .9):
        accept = np.maximum(probability, 1 - probability) >= level
        selectivity[str(level)] = [float(accept.mean()),
                                   float(np.mean((probability[accept] >= .5) != labels[accept]))
                                   if accept.any() else None]
    comparisons = {}
    by_spec = {}
    for row, label, value in zip(table, labels, scores):
        spec = row["sample_id"].rsplit(":", 1)[0]
        key = (row["source_group"], spec)
        match = by_spec.setdefault(key, {})
        if label in match:
            raise ValueError("Duplicate label within an edit spec")
        match[label] = value
    aligned = [pair for pair in by_spec.values() if set(pair) == {0, 1}]
    pair_win = float(np.mean([(pair[1] > pair[0]) + .5 * (pair[1] == pair[0])
                              for pair in aligned])) if aligned else None
    if alternate is not None:
        score_arrays(alternate)
        by_id = {row["sample_id"]: row for row in alternate}
        if set(by_id) != {row["sample_id"] for row in table}:
            raise ValueError("Paired tables have different sample IDs")
        matched = [by_id[row["sample_id"]] for row in table]
        if any((row["source_group"], row["label"]) != (other["source_group"], other["label"])
               for row, other in zip(table, matched)):
            raise ValueError("Paired method source identity or label mismatch")
        alternate = np.array([float(row["logit"]) for row in matched])
        comparisons["paired_auroc_delta"] = ranking(labels, scores)[0] - ranking(labels, alternate)[0]
    roc, ap = ranking(labels, scores)
    return {"documents": len(labels), "groups": len(set(groups)), "auroc": roc, "ap": ap,
            "paired_spec_count": len(aligned), "within_pair_win_with_half_credit_for_ties": pair_win,
            "cutoff": cutoff, "operating_point": operating_point(labels, scores, cutoff),
            "fixed_calibration_95": intervals(groups, labels, scores, cutoff, alternate),
            "resampled_calibration_cutoff_95": {key: interval for key, interval in intervals(
                groups, labels, scores, cutoff, calibration=(cal_groups[cal_labels == 0],
                                                           cal_scores[cal_labels == 0])).items()
                if key in ("fpr", "recall")},
            "brier": float(np.mean((labels - probability) ** 2)), "constant_brier": float(np.mean((labels - prevalence) ** 2)),
            "ece": ece, "constant_ece": abs(float(labels.mean()) - prevalence),
            "risk_coverage": selectivity,
            "constant_risk_coverage": {str(level): [1.0, float(np.mean((prevalence >= .5) != labels))]
                                       if max(prevalence, 1 - prevalence) >= level else [0.0, None]
                                       for level in (.5, .55, .6, .7, .9)}, **comparisons}


def check_manifest(table, manifest, role):
    expected = {row["sample_id"]: (row["source_group"], row["source_dataset"],
                                 int(row["label"] == "manipulated"))
                for row in manifest if row["split"] == role}
    actual = {row["sample_id"]: (row["source_group"], row["corpus"], int(row["label"])) for row in table}
    if actual != expected or len(actual) != len(table):
        raise ValueError(f"Prediction coverage or labels differ from manifest for {role}")


def run(output):
    study = ARTIFACTS / "receipt-study-v1"
    frozen = ARTIFACTS / "receipt-retrospective-v1"
    xfund = ARTIFACTS / "receipt-xfund-v1"
    manifest_path = ARTIFACTS / "receipt-protocol-v1-contained/manifest.csv"
    manifest = rows(manifest_path)
    roles_by_group = {}
    for row in manifest:
        group, role = row["source_group"], row["split"]
        if roles_by_group.setdefault(group, role) != role or row["source_dataset"] == "xfund":
            raise ValueError("Receipt partitions cross source groups or contain forms")
    freeze = json.loads((frozen / "freeze.json").read_text())
    matrix = json.loads((study / "matrix.json").read_text())
    report = {"development": {}, "historical": {}, "exports": {}, "generator_comparison": {},
              "trufor_historical_panel": {}}
    for recipe, summary in matrix.items():
        seeds = []
        for item in summary["seeds"]:
            path = study / item["run"]
            fit = json.loads((path / "fit.json").read_text())
            mode = "full" if "-fraction-" in path.name else recipe
            fraction = float(recipe.removeprefix("full-")) if mode != recipe else 1.
            if (file_sha256(path / "model.pt") != fit["checkpoint_sha256"]
                    or file_sha256(manifest_path) != fit["training_manifest_sha256"]
                    or fit["mode"] != mode or fit["fraction"] != fraction or fit["seed"] != item["seed"]):
                raise ValueError(f"Fit provenance mismatch: {path.name}")
            scores, cal = rows(path / "validation.csv"), rows(path / "calibration.csv")
            check_manifest(scores, manifest, "validation")
            check_manifest(cal, manifest, "calibration")
            group, labels, logits = score_arrays(scores)
            roc, ap = ranking(labels, logits)
            if not np.allclose([roc, ap], [item["auroc"], item["ap"]], atol=1e-12, rtol=0):
                raise ValueError(f"Point metrics disagree: {path.name}")
            seed_result = {"seed": fit["seed"], "auroc": roc, "ap": ap}
            if recipe == "residual":
                other = rows(study / f"full-{fit['seed']}/validation.csv")
                seed_result["residual_vs_rgb"] = assess(scores, cal, other)
            seeds.append(seed_result)
        aucs = [seed["auroc"] for seed in seeds]
        report["development"][recipe] = {"seeds": seeds, "mean_auroc": float(np.mean(aucs)),
                                         "seed_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None}
    main_recipes = {name: entry for name, entry in report["development"].items()
                    if len(entry["seeds"]) == 3}
    selection = max(main_recipes, key=lambda name: (
        main_recipes[name]["mean_auroc"],
        float(np.mean([seed["ap"] for seed in main_recipes[name]["seeds"]]))))
    if selection != freeze["configuration"] or freeze["seed"] != 42:
        raise ValueError("Freeze differs from the development ranking and prespecified seed")
    report["selection"] = {"recipe": selection, "seed": freeze["seed"]}
    chosen = study / f"{freeze['configuration']}-{freeze['seed']}"
    hashes = {"checkpoint": chosen / "model.pt", "calibration": chosen / "calibration.csv",
              "manifest": manifest_path, "matrix": study / "matrix.json"}
    if any(file_sha256(path) != freeze[f"{key}_sha256"] for key, path in hashes.items()):
        raise ValueError("Frozen model inputs differ")
    xfund_lock = json.loads((xfund / "freeze-before-labels.json").read_text())
    if any(xfund_lock[key] != freeze[key] for key in freeze if key != "evaluation_status"):
        raise ValueError("XFUND uses a different freeze")
    cal = rows(chosen / "calibration.csv")
    for role, file, source, split in (("test", frozen / "test.csv", manifest, "test"),
            ("shifted_test", frozen / "shifted_test.csv", manifest, "shifted_test"),
            ("xfund", xfund / "scores.csv", rows(xfund / "manifest.csv"), "test")):
        predictions = rows(file)
        check_manifest(predictions, source, split)
        result = assess(predictions, cal)
        if result["cutoff"] != freeze["operating_points"]["0.05"]:
            raise ValueError("Frozen threshold differs")
        report["historical"][role] = {**result, "scores_sha256": file_sha256(file)}
    generator = ARTIFACTS / "receipt-generator-evaluation-v1"
    matches = rows(generator / "matched-pairs.csv")
    if (len({row["spec_id"] for row in matches}) != len(matches)
            or any(row["v1_generator"] == row["v2_generator"]
                   or roles_by_group.get(row["source_group"]) != row["historical_role"]
                   or row["historical_role"] not in ("test", "shifted_test") for row in matches)):
        raise ValueError("Matched releases contain an ineligible or duplicated edit spec")
    panel = ARTIFACTS / "trufor-native-panel-v1"
    panel_manifest = rows(panel / "manifest.csv")
    panel_cal = rows(panel / "trufor-calibration.csv")
    check_manifest(panel_cal, panel_manifest, "calibration")
    for role in ("validation", "test", "shifted_test"):
        scores = rows(panel / f"trufor-{role}.csv")
        check_manifest(scores, panel_manifest, role)
        report["trufor_historical_panel"][role] = assess(
            scores, panel_cal, rows(panel / f"residual-{role}.csv"))
        if role == "validation":
            continue
        v1 = rows(generator / f"{role}-v1.csv")
        v2 = rows(generator / f"{role}-v2.csv")
        for version, scores in (("v1", v1), ("v2", v2)):
            check_manifest(scores, rows(generator / "manifests" / f"{version}.csv"), role)
        report["generator_comparison"][role] = {"v1": assess(v1, cal), "v2": assess(v2, cal, v1)}
    distill = ARTIFACTS / "receipt-distill-v1"
    report["distillation"] = {}
    for mode in ("student", "distilled"):
        folder = distill / mode
        check_manifest(rows(folder / "validation.csv"), manifest, "validation")
        fit = json.loads((folder / "fit.json").read_text())
        if file_sha256(folder / "model.pt") != fit["checkpoint_sha256"]:
            raise ValueError("Student checkpoint differs")
        report["distillation"][mode] = assess(rows(folder / "validation.csv"), rows(folder / "calibration.csv"),
            rows(distill / "student/validation.csv") if mode == "distilled" else None)
    for name in ("receipt-share-v1", "receipt-share-distill-v1"):
        folder = ARTIFACTS / name
        index = json.loads((folder / "index.json").read_text())
        for item in index:
            table = folder / item["file"]
            if file_sha256(table) != item["table_sha256"] or len(rows(table)) != item["rows"]:
                raise ValueError("Pseudonymized export index mismatch")
        report["exports"][name] = {"tables": len(index), "rows": sum(item["rows"] for item in index)}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
