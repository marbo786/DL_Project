"""Audit and evaluate a versioned private inventory without threshold fitting."""

import argparse
import json
import platform
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import __version__ as pillow_version
from sklearn.metrics import roc_auc_score

from research.phase_f_compare import POLICY, POLICY_PATH, PRIMARY, THRESHOLDS, compare, decode, verify_freeze
from research.phase_f_dataset import PHYSICAL, audit, excluded_sources, load_mask, read_hashed
from research.reference_compare import project


def rate(flags):
    return sum(flags) / len(flags) if flags else None


def coverage(rows):
    return rate([row["score"] is not None for row in rows])


def timing(rows):
    values = [row["ms"] for row in rows]
    return dict(zip(("median_ms", "p95_ms"), np.percentile(values, [50, 95]).tolist())) if values else None


def classification(rows):
    positive = [r for r in rows if r["scenario"] == "localized_edit"]
    negative = [r for r in rows if r["scenario"] in {"recapture", "simulated_unchanged"}]
    scored = [r for r in positive + negative if r["score"] is not None]
    labels = [r["scenario"] == "localized_edit" for r in scored]
    visible_iou = [r["iou"] for r in positive if r["iou"] is not None]
    return {
        "localized_edits": len(positive), "unchanged_controls": len(negative),
        "score_coverage_edits": coverage(positive), "score_coverage_recaptures": coverage(negative),
        "accepted_only_auroc": (
            roc_auc_score(labels, [r["score"] for r in scored]) if len(set(labels)) == 2 else None
        ),
        "frozen_operating_points": {
            target: {
                "threshold": cutoff,
                "recall_all_edits": rate([r["score"] is not None and r["score"] > cutoff for r in positive]),
                "fpr_all_recaptures": rate([r["score"] is not None and r["score"] > cutoff for r in negative]),
                "recall_accepted": rate([r["score"] > cutoff for r in positive if r["score"] is not None]),
                "fpr_accepted": rate([r["score"] > cutoff for r in negative if r["score"] is not None]),
            }
            for target, cutoff in THRESHOLDS.items()
        },
        "localization_evaluable": len(visible_iou),
        "localization_overlap_iou": rate(visible_iou),
        "localization_all_edits_iou": rate([r["iou"] or 0.0 for r in positive]),
    }


def summarize(rows):
    strata = {}
    for mode in ("physical", "born_digital", "simulated"):
        subset = [r for r in rows if r["mode"] == mode]
        strata[mode] = {"classification": classification(subset), "scenarios": {}}
        for scenario in ("recapture", "localized_edit", "legitimate_revision", "wrong_reference", "simulated_unchanged"):
            group = [r for r in subset if r["scenario"] == scenario]
            scored = [r for r in group if r["score"] is not None]
            strata[mode]["scenarios"][scenario] = {
                "pairs": len(group), "coverage": coverage(group),
                "candidate_source_groups": len({r["candidate_group"] for r in group}),
                "abstention": rate([r["score"] is None for r in group]),
                "difference_flagged_all_pairs": rate([r["score"] > PRIMARY if r["score"] is not None else False for r in group]),
                "difference_flagged_accepted": rate([r["score"] > PRIMARY for r in scored]),
                "latency": timing(group),
            }
    return {"strata": strata, "latency": timing(rows), "explicit_mismatch_detection": "unsupported"}


def evaluate(data, images, folder):
    verify_freeze()
    rows = []
    for pair in data["pairs"]:
        ref, can = (images[pair[key]] for key in ("reference", "candidate"))
        ref_bytes, can_bytes = read_hashed(folder, ref), read_hashed(folder, can)
        started = perf_counter()
        reference, candidate = decode(ref_bytes), decode(can_bytes)
        result = compare(reference, candidate)
        ms = (perf_counter() - started) * 1000
        iou, visible = None, None
        if "mask" in pair and result.alignment is not None and result.alignment.accepted:
            mask = load_mask(folder, pair["mask"], candidate.size)
            height, width = result.mask.shape
            raster = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA) > 0
            aligned = project(raster.astype(np.uint8), result.alignment.matrix, (height, width), nearest=True) > 0
            target = aligned & result.alignment.valid
            visible = float(target.sum() / aligned.sum()) if aligned.any() else 0.0
            if target.any():
                iou = float((target & result.mask).sum() / (target | result.mask).sum())
        acquisitions = {ref["acquisition"], can["acquisition"]}
        mode = "simulated" if "simulated" in acquisitions else "physical" if acquisitions <= PHYSICAL else "born_digital"
        rows.append({"pair": pair["id"], "reference_group": ref["source_group"], "candidate_group": can["source_group"],
                     "scenario": pair["scenario"], "mode": mode, "score": result.score, "reason": result.reason,
                     "decision": result.decision, "iou": iou, "visible_target_fraction": visible, "ms": ms})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "evaluate"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--aiforge-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "evaluate" and (args.output is None or args.output.exists()):
        parser.error("evaluate requires a new --output file")
    folder = args.data_root.resolve()
    manifest_digest = sha256(args.manifest.read_bytes()).hexdigest()
    data, images, counts = audit(args.manifest, folder, excluded_sources(args.aiforge_manifest))
    report = {"dataset_version": data["dataset_version"], "manifest_sha256": manifest_digest,
              "policy_version": POLICY["version"], "policy_sha256": sha256(POLICY_PATH.read_bytes()).hexdigest(),
              "scenario_counts": counts}
    if args.command == "evaluate":
        rows = evaluate(data, images, folder)
        report.update(metrics=summarize(rows), rows=rows, runtime={"python": platform.python_version(),
                      "platform": platform.platform(), "opencv": cv2.__version__, "numpy": np.__version__,
                      "pillow": pillow_version, "opencv_threads": cv2.getNumThreads()})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
