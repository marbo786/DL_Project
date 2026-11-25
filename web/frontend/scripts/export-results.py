import csv
import json
from hashlib import sha256
from math import exp
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "web/frontend/data"
SOURCES = {
    "development": "artifacts/receipt-study-v1/residual-42/metrics.json",
    "development_scores": "artifacts/receipt-study-v1/residual-42/validation.csv",
    "frozen_policy": "artifacts/receipt-retrospective-v1/freeze.json",
    "id": "artifacts/receipt-retrospective-v1/test.json",
    "shifted": "artifacts/receipt-retrospective-v1/shifted_test.json",
    "pareto": "artifacts/receipt-pareto-v1/summary.json",
    "stress": "artifacts/receipt-residual-stress-v1/summary.json",
    "model": "artifacts/receipt-study-v1/residual-42/fit.json",
    "model_card": "docs/MODEL_CARD.md",
    "targets": "docs/REQUIREMENTS.md",
}


def read(key):
    return json.loads((ROOT / SOURCES[key]).read_text(encoding="utf-8"))


def cohort(key, name, published=None):
    data = read(key)
    values = (round(data["auroc"], 4), round(data["classification_at_probability_half"]["macro_f1"], 4),
              round(data["reliability"]["ece"], 4))
    if published is not None and values != published:
        raise ValueError(f"{key} aggregate differs from the published model card")
    return dict(name=name, auroc=values[0], macro_f1=values[1], ece=values[2], documents=data["documents"])


def review_points(development):
    scores = ROOT / SOURCES["development_scores"]
    if sha256(scores.read_bytes()).hexdigest() != development["evaluation_sha256"]:
        raise ValueError("Development score digest differs from the saved evaluation")
    frozen = read("frozen_policy")
    slope, intercept = (frozen["calibration"][key] for key in ("slope", "intercept"))
    threshold = development["operating_points"]["0.05"]["logit_threshold"]
    if threshold != frozen["operating_points"]["0.05"]:
        raise ValueError("Development and frozen review cutoffs differ")
    with scores.open(newline="", encoding="utf-8") as stream:
        rows = [(float(row["logit"]), int(row["label"])) for row in csv.DictReader(stream)]
    if len(rows) != development["documents"]:
        raise ValueError("Development document count differs")

    def confidence(logit):
        probability = 1 / (1 + exp(-max(-60., min(60., slope * logit + intercept))))
        return max(probability, 1 - probability)

    scored = [(confidence(logit), logit > threshold, label) for logit, label in rows]
    points = []
    for point in development["risk_coverage"]:
        accepted = [row for row in scored if row[0] >= point["confidence"]]
        if abs(len(accepted) / len(scored) - point["coverage"]) > 1e-12:
            raise ValueError("Confidence coverage differs from the saved risk curve")
        review = [label for _, flag, label in accepted if flag]
        points.append({**{key: point[key] for key in ("confidence", "coverage", "accepted_error")},
                       "review_count": len(review), "edits_caught": sum(review), "false_flags": len(review) - sum(review)})
    genuine, edited = development["operating_points"]["0.05"]["confusion"]
    if (points[0]["false_flags"], points[0]["edits_caught"]) != (genuine[1], edited[1]):
        raise ValueError("Ungated counts differ from the saved operating point")
    return points


def main():
    development = read("development")
    pareto = read("pareto")
    stress = read("stress")
    shifted = read("shifted")
    targets = (ROOT / SOURCES["targets"]).read_text(encoding="utf-8")
    policy = (ROOT / SOURCES["model_card"]).read_text(encoding="utf-8")
    parameters = int(read("model")["parameters"])
    if not all(value in targets for value in ("90%", "0.75", "100 ms", "50% coverage", "10%", "105.15", "1% assumed prevalence")):
        raise ValueError("Review the published requirements before exporting the limits")
    if not all(value in policy for value in ("prioritized zero", "six genuine and six edited", "2,660")):
        raise ValueError("Review the published routing replay before exporting the limits")
    development_gate = next(row for row in development["risk_coverage"] if row["confidence"] == .55)
    shifted_gate = next(row for row in shifted["risk_coverage"] if row["confidence"] == .55)
    data = {
        "sources": {key: {"path": value, "sha256": sha256((ROOT / value).read_bytes()).hexdigest()}
                    for key, value in SOURCES.items()},
        "cohorts": [cohort("id", "CORD / SROIE", (.5134, .5063, .0029)),
                    cohort("shifted", "WildReceipt", (.5015, .4863, .0106))],
        "reliability": development["reliability"]["bins"],
        "constant_reliability": development["constant_reliability"]["bins"],
        "risk": review_points(development),
        "development_documents": development["documents"],
        "pareto": [{key: row[key] for key in ("checkpoint_mib", "median_ms", "p95_ms", "recall")} | {"id": id}
                   for id, row in pareto["results"].items()],
        "review_budget": pareto["review_budget_documents"],
        "stress": [dict(id=id, **{key: row[key] for key in ("auroc", "recall", "fpr")})
                   for id, row in stress["stress"].items()],
        "stress_pairs": stress["pairs"],
        "limits": {
            "priority_recall": {"observed": 0, "target": .9},
            "shifted_f1": {"observed": round(shifted["classification_at_probability_half"]["macro_f1"], 4), "target": .75},
            "coverage": {"observed": development_gate["coverage"], "target": .5},
            "accepted_error": {"observed": development_gate["accepted_error"], "target": .1},
            "queue_load": {"observed": 105.15, "target": 100},
            "shifted_ece": {"observed": round(shifted["reliability"]["ece"], 4), "target": .1},
            "median_ms": {"observed": round(pareto["results"]["residual-fp32"]["median_ms"], 2),
                          "target": pareto["target_median_ms"]},
            "parameters": {"observed": parameters, "target": 25_000_000},
        },
        "published_shift": {
            "coverage": round(shifted_gate["coverage"], 4),
            "accepted_error": shifted_gate["accepted_error"],
            "binary_fpr": round(shifted["operating_points"]["0.05"]["fpr"], 3),
            "binary_recall": round(shifted["operating_points"]["0.05"]["recall"], 3),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
