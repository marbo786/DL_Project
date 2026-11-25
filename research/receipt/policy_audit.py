"""Recompute document routes from frozen score tables and quantify random review."""
import argparse
import csv
import json
from collections import Counter
from math import exp
from pathlib import Path

from scipy.stats import hypergeom

from research.manifest import file_sha256
from research.receipt.demo import route_receipt


def audit(artifacts):
    historical = artifacts / "receipt-retrospective-v1"
    selected = artifacts / "receipt-study-v1/residual-42"
    forms = artifacts / "receipt-xfund-v1"
    lock = json.loads((historical / "freeze.json").read_text())
    mapping = lock["calibration"]
    threshold = lock["operating_points"]["0.05"]
    roles = {"development": (selected / "validation.csv", selected / "metrics.json"),
             "historical_id": (historical / "test.csv", historical / "test.json"),
             "historical_shift": (historical / "shifted_test.csv", historical / "shifted_test.json"),
             "held_back_forms": (forms / "scores.csv", forms / "metrics.json")}
    results = {}
    for role, (scores, metrics) in roles.items():
        report = json.loads(metrics.read_text())
        if file_sha256(scores) != report["evaluation_sha256"]:
            raise ValueError(f"Digest mismatch for {role}")
        counts = {name: Counter() for name in ("probability_half", "nominal_five_percent", "demo")}
        with scores.open(newline="") as stream:
            for r in csv.DictReader(stream):
                logit = float(r["logit"])
                p = 1 / (1 + exp(-max(-60., min(60., mapping["slope"] * logit + mapping["intercept"]))))
                route, _ = route_receipt(logit, p, threshold)
                label = int(r["label"])
                counts["probability_half"]["flag" if p >= .5 else "other", label] += 1
                counts["nominal_five_percent"]["flag" if logit > threshold else "other", label] += 1
                counts["demo"][route, label] += 1
        results[role] = {}
        for policy, outcomes in counts.items():
            names = ("Prioritize review", "Normal queue", "Inconclusive") if policy == "demo" else ("flag", "other")
            results[role][policy] = {name: [outcomes[name, 0], outcomes[name, 1]] for name in names}
        for policy, evidence in (("probability_half", report["classification_at_probability_half"]),
                                 ("nominal_five_percent", report["operating_points"]["0.05"])):
            data = results[role][policy]
            if list(map(list, zip(data["other"], data["flag"]))) != evidence["confusion"]:
                raise ValueError(f"Confusion mismatch for {role}/{policy}")
    timing = artifacts / "receipt-pareto-v1/scores-and-timings.csv"
    with timing.open(newline="") as stream:
        records = [r for r in csv.DictReader(stream) if r["model"] == "residual-fp32"]
    ranked = sorted(records, key=lambda r: (-float(r["logit"]), r["sample_id"]))
    n, positives, budget = len(ranked), sum(r["edited"] == "True" for r in ranked), len(ranked) // 10
    found = sum(r["edited"] == "True" for r in ranked[:budget])
    random = hypergeom(n, positives, budget)
    return {"columns": ["genuine", "edited"], "counts": results, "random_review": {
            "documents": n, "edits": positives, "slots": budget, "found": found,
            "random_expected": random.mean(), "random_central_95": list(map(int, random.interval(.95))),
            "random_probability_at_least_observed": random.sf(found - 1)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    print(json.dumps(audit(parser.parse_args().artifacts), indent=2))
