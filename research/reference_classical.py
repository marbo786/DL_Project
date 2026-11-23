"""Exact, nuisance, and difficult reference-comparison baselines."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from research.reference_compare import difference, register, regions, score_map
from research.reference_data import PINNED_MANIFEST, REGIMES, pairs, project
from research.reference_metrics import calibration_split, pixel_metrics, summary


CLASSICAL = (("raw_rgb", False, "rgb", .055), ("raw_gray", False, "gray", .10),
             ("registered_rgb", True, "rgb", .055), ("registered_gray", True, "gray", .10),
             ("registered_ssim", True, "ssim", .13))


def grid_error(estimate, actual, shape):
    height, width = shape
    points = np.float32([[[x * width, y * height]] for y in (.15, .5, .85) for x in (.15, .5, .85)])
    recovered = cv2.perspectiveTransform(points, estimate @ actual)
    return float(np.median(np.linalg.norm(recovered - points, axis=2)))


def gallery(destination, item, aligned, score, predicted):
    heat = cv2.applyColorMap(np.uint8(np.clip(score * 3, 0, 1) * 255), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = aligned.image.copy()
    overlay[predicted] = .45 * overlay[predicted] + .55 * np.array([255, 40, 20])
    panels = [item.reference, item.candidate, aligned.image, heat, overlay]
    caption = f"trusted | candidate | aligned | difference | overlay    accepted {aligned.accepted}, inliers {aligned.inliers}"
    view = np.concatenate(panels, axis=1)
    view = cv2.copyMakeBorder(view, 25, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(view, caption, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255))
    cv2.imwrite(str(destination), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))


def wire_benchmark(args):
    timings = []
    for _, group in pairs(args.manifest, args.asset_root, "validation", args.pairs, ("transformed",)):
        for item in group:
            wire = [cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1]
                    for image in (item.reference, item.candidate)]
            started = perf_counter()
            reference, candidate = [cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for data in wire]
            aligned = register(reference, candidate)
            heat = difference(reference, aligned.image, aligned.valid, "ssim")
            score_map(heat, aligned.valid)
            regions(heat, aligned.valid, .13)
            timings.append(1000 * (perf_counter() - started))
    return {"receipts": len(timings), "median_ms": float(np.median(timings)), "p95_ms": float(np.quantile(timings, .95))}


def run(args):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(42)
    if sha256(args.manifest.read_bytes()).hexdigest() != PINNED_MANIFEST:
        raise ValueError("Unexpected manifest checksum")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.wire_bench:
        (args.output / "summary.json").write_text(json.dumps(wire_benchmark(args), indent=2), encoding="utf-8")
        return
    records, alignment_audit = defaultdict(list), defaultdict(list)
    methods = tuple(method for method in CLASSICAL if not args.raw_only or not method[1])
    for index, (regime, group) in enumerate(pairs(args.manifest, args.asset_root, "validation", args.pairs, REGIMES)):
        for item in group:
            shape = item.reference.shape[:2]
            align = None
            if not args.raw_only:
                start = perf_counter()
                align = register(item.reference, item.candidate)
                registration_ms = 1000 * (perf_counter() - start)
                alignment_audit[regime].append({
                    "accepted": align.accepted, "milliseconds": registration_ms,
                    "raw_grid_error": grid_error(np.eye(3), item.transform, shape),
                    "estimated_grid_error": grid_error(align.matrix, item.transform, shape),
                    "reference_overlap": float(align.valid.mean()), "inliers": align.inliers,
                    "reprojection_px": align.reprojection_px, "class": item.label,
                })
            for name, registered, kind, threshold in methods:
                start = perf_counter()
                candidate = align.image if registered else item.candidate
                valid = align.valid if registered else np.ones(shape, bool)
                diff = difference(item.reference, candidate, valid, kind)
                score = score_map(diff, valid)
                predicted = regions(diff, valid, threshold)
                elapsed = 1000 * (perf_counter() - start) + (registration_ms if registered else 0.)
                transform = align.matrix if registered else np.eye(3)
                evaluation_valid = valid & (project(item.valid, transform, shape, nearest=True) > 0)
                target = project(item.mask, transform, shape, nearest=True)
                records[(regime, name)].append({"source": item.source, "label": item.label, "score": score,
                    "accepted": align.accepted if registered else True, "milliseconds": elapsed,
                    "mean_residual": float(diff[valid].mean()), **pixel_metrics(predicted, target, evaluation_valid)})
                if (name == "registered_ssim" and index // 3 < 3 and item.label == 1) or (
                        name == "registered_ssim" and not align.accepted and not (args.output / f"{regime}-rejected.png").exists()):
                    suffix = f"{index // 3}-edited" if align.accepted else "rejected"
                    gallery(args.output / f"{regime}-{suffix}.png", item, align, diff, predicted)
        if (index + 1) % 96 == 0:
            print(f"processed {index + 1} source-regime pairs", flush=True)
    result = {regime: {} for regime in REGIMES}
    for (regime, name), rows in records.items():
        calibration = calibration_split([row["source"] for row in rows])
        result[regime][name] = summary(rows, calibration)
        result[regime][name]["mean_residual"] = float(np.mean([r["mean_residual"] for r in rows]))
    alignment = {}
    for regime, rows in alignment_audit.items():
        keep = [row for row in rows if row["accepted"]]
        median = lambda data, key: float(np.median([row[key] for row in data])) if data else None
        alignment[regime] = {
            "pairs": len(rows), "accepted_fraction": len(keep) / len(rows),
            "accepted_genuine": sum(r["class"] == 0 for r in keep) / (len(rows) / 2),
            "accepted_edited": sum(r["class"] == 1 for r in keep) / (len(rows) / 2),
            "raw_median_grid_error_px": median(rows, "raw_grid_error"),
            "accepted_grid_error_px": median(keep, "estimated_grid_error"),
            "all_median_grid_error_px": median(rows, "estimated_grid_error"),
            "median_registration_ms": median(rows, "milliseconds"), "accepted_reprojection_px": median(keep, "reprojection_px"),
        }
    (args.output / "summary.json").write_text(json.dumps({"methods": result, "alignment": alignment},
                                                        indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "measurements.json").write_text(json.dumps({"/".join(key): rows for key, rows in records.items()}),
                                                              encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "asset-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=228)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--wire-bench", action="store_true")
    run(parser.parse_args())
