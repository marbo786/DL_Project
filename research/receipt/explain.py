"""Audit the frozen receipt tile cue using development-only interventions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from research.data import ManifestDocuments
from research.manifest import file_sha256
from research.patch_data import load_pair, rgb_crop
from research.receipt.demo import LOCK_PATH, LOCK_SHA256, ROOT, RUN_PATH, FrozenReceiptDemo
from research.receipt.model import SIDE, inference_tiles, pooled_logit, tile_logits
from research.receipt.report import plt, save
from research.receipt.stress import pair_indices, write_json


CASES = {
    "A": ("000003626:original", "highlight intervention"),
    "B": ("000003581:original", "background-only pooling"),
    "C": ("000000042:edited", "half-tile grid offset"),
    "D": ("000000383:edited", "top-one pooling"),
    "E": ("000003631:edited", "restore edited pixels"),
}
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 427


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def clipped_boxes(boxes, size):
    return [(x, y, min(right, size[0]), min(bottom, size[1])) for x, y, right, bottom in boxes]


def intersects(first, second):
    return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(first[3], second[3])


def matched_control(image, boxes, top, mask_hits):
    highlight = boxes[top]
    size = image.crop(highlight).size
    gray = image.convert("L")

    def appearance(box):
        pixels = np.asarray(gray.crop(box), dtype=np.float64)
        return np.array([pixels.mean(), pixels.std()])

    candidates = [box for box, hit in zip(boxes, mask_hits)
                  if not hit and not intersects(box, highlight) and (box[2] - box[0], box[3] - box[1]) == size]
    if not candidates:
        return None
    target = appearance(highlight)
    return min(candidates, key=lambda box: (float(np.linalg.norm(appearance(box) - target)), box))


def neutralize(image, box):
    result = image.copy()
    result.paste((127, 127, 127), box)
    return result


def offset_boxes(size):
    def starts(length):
        end = max(0, length - SIDE)
        return sorted({0, end, *range(SIDE // 2, end, SIDE)})

    return [(x, y, x + SIDE, y + SIDE) for y in starts(size[1]) for x in starts(size[0])]


def predict_tiles(demo, image):
    boxes, inputs = inference_tiles(image, demo.model.mode)
    scores = tile_logits(demo.model, inputs, demo.batch_size)
    return clipped_boxes(boxes, image.size), scores


def deletion(demo, image, boxes, scores, mask_hits):
    top = int(scores.argmax())
    control = matched_control(image, boxes, top, mask_hits)
    if control is None:
        return None
    baseline = pooled_logit(scores)
    remove_top = demo.score(neutralize(image, boxes[top])).logit
    remove_control = demo.score(neutralize(image, control)).logit
    return {"control_box": control, "top_removed_logit": remove_top, "control_removed_logit": remove_control,
            "top_drop": baseline - remove_top, "control_drop": baseline - remove_control,
            "paired_drop_advantage": remove_control - remove_top}


def restricted(scores, hit):
    mask = torch.tensor(hit, dtype=torch.bool)
    return {"edit_tiles": int(mask.sum()), "background_tiles": int((~mask).sum()),
            "edit_logit": pooled_logit(scores[mask]) if mask.any() else None,
            "background_logit": pooled_logit(scores[~mask]) if (~mask).any() else None}


def pair_audit(demo, documents, pair, reference, faithfulness):
    original, edited, mask = load_pair(documents, pair)
    boxes, original_scores = predict_tiles(demo, original)
    _, edited_scores = predict_tiles(demo, edited)
    hit = [mask.crop(box).getbbox() is not None for box in boxes]
    evidence = []
    for index, image, scores in zip(pair, (original, edited), (original_scores, edited_scores)):
        row = documents.rows[index]
        expected = reference[row["sample_id"]]
        top = boxes[int(scores.argmax())]
        logit = pooled_logit(scores)
        if abs(logit - float(expected["logit"])) > 1e-5 or top != tuple(map(int, expected["bbox"].split())):
            raise ValueError(f"Frozen development parity failed for {row['sample_id']}")
        record = {"sample_id": row["sample_id"], "source_group": row["source_group"], "label": row["label"],
                  "document_size": image.size, "logit": logit, "top_logit": float(scores.max()), "tile": top,
                  **restricted(scores, hit)}
        if faithfulness or row["sample_id"] == CASES["A"][0]:
            record["deletion"] = deletion(demo, image, boxes, scores, hit)
        evidence.append((record, image))
    return evidence, original, mask


def case_probe(demo, case, item, image, original, mask):
    if case == "A":
        return item["deletion"]
    if case == "B":
        return {"background_only_logit": item["background_logit"], "edit_site_logit": item["edit_logit"]}
    if case == "C":
        boxes = offset_boxes(image.size)
        inputs = torch.stack([rgb_crop(image, box) for box in boxes])
        scores = tile_logits(demo.model, inputs, demo.batch_size)
        tile = clipped_boxes(boxes, image.size)[int(scores.argmax())]
        return {"offset_grid_logit": pooled_logit(scores), "offset_grid_tile": tile,
                "offset_tile_mask_intersection_px": mask.crop(tile).histogram()[255]}
    if case == "D":
        return {"top_one_logit": item["top_logit"], "top_three_logit": item["logit"]}
    restored = Image.composite(original, image, mask)
    return {"mask_restored_logit": demo.score(restored).logit, "original_logit": demo.score(original).logit,
            "restored_equals_original_pixels": bool(np.array_equal(restored, original))}


def private_panel(path, item, image, mask):
    from matplotlib.patches import Rectangle

    tile = item["tile"]
    truth = mask.getbbox()
    regions = [tile] if truth is None else [tile, truth]
    crop = (max(0, min(box[0] for box in regions) - 20), max(0, min(box[1] for box in regions) - 20),
            min(image.width, max(box[2] for box in regions) + 20), min(image.height, max(box[3] for box in regions) + 20))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    axes[0].imshow(image.crop(crop))
    axes[1].imshow(mask.crop(crop), cmap="gray", vmin=0, vmax=255)
    axes[2].imshow(image.crop(crop))
    axes[2].add_patch(Rectangle((tile[0] - crop[0], tile[1] - crop[1]), tile[2] - tile[0], tile[3] - tile[1],
                                fill=False, edgecolor="#e79b29", linewidth=2))
    for axis, title in zip(axes, ("Input region", "True edit mask", "Highest-scoring tile")):
        axis.set_title(title)
        axis.set_axis_off()
    fig.suptitle(f"Case {item['case']} | logit {item['logit']:.3f} | cutoff {item['cutoff']:.3f}")
    save(fig, path)


def geometry_panel(path, cases):
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, len(cases), figsize=(10, 3.55))
    for axis, item in zip(axes, cases):
        width, height = item["document_size"]
        axis.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f5f8fa", edgecolor="#607182"))
        if item["mask_bbox"]:
            x1, y1, x2, y2 = item["mask_bbox"]
            axis.add_patch(Rectangle((x1 / width, y1 / height), (x2 - x1) / width, (y2 - y1) / height,
                                     facecolor="#aad9d2", edgecolor="#087f73", linewidth=1.4))
        x1, y1, x2, y2 = item["tile"]
        axis.add_patch(Rectangle((x1 / width, y1 / height), (x2 - x1) / width, (y2 - y1) / height,
                                 fill=False, edgecolor="#b66b09", linewidth=1.4))
        axis.set(xlim=(-.06, 1.06), ylim=(1.05, -.05), title=f"Case {item['case']}\nLogit {item['logit']:.2f}")
        axis.set_axis_off()
    fig.suptitle(f"Normalized receipt coordinates | frozen logit cutoff {cases[0]['cutoff']:.3f}")
    fig.text(.30, .04, "Orange outline: predicted tile", color="#a45c00", ha="center")
    fig.text(.69, .04, "Teal rectangle: edit-mask bounds", color="#087f73", ha="center")
    fig.tight_layout(rect=(0, .09, 1, .95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def faithfulness_summary(records):
    usable = [row for row in records if row["deletion"] is not None]
    top = np.array([row["deletion"]["top_drop"] for row in usable])
    control = np.array([row["deletion"]["control_drop"] for row in usable])
    groups = np.array([row["source_group"] for row in usable])
    bins = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mean_deltas = [np.mean((top - control)[np.concatenate([bins[i] for i in rng.integers(len(bins), size=len(bins))])])
                   for _ in range(BOOTSTRAPS)]
    return {"documents": len(records), "matched_documents": len(usable), "groups": len(bins),
            "mean_top_drop": float(top.mean()), "mean_control_drop": float(control.mean()),
            "mean_paired_advantage": float((top - control).mean()),
            "paired_advantage_95": np.quantile(mean_deltas, (.025, .975)).tolist(),
            "top_wins": int((top > control).sum()), "control_wins": int((top < control).sum()),
            "ties": int((top == control).sum())}


def run(args):
    torch.set_num_threads(2)
    demo = FrozenReceiptDemo()
    frozen = read_json(ROOT / LOCK_PATH)
    manifest_sha = file_sha256(args.manifest)
    if manifest_sha != frozen["manifest_sha256"]:
        raise ValueError("The explanation requires the manifest from the frozen study")
    documents = ManifestDocuments(args.manifest, args.root, "validation")
    groups, pairs = pair_indices(documents)
    paired_count = len(pairs)
    lookup = {row["sample_id"]: i for i, row in enumerate(documents.rows)}
    case_by_id = {identifier: (letter, hypothesis) for letter, (identifier, hypothesis) in CASES.items()}
    for identifier, _ in CASES.values():
        spec = identifier.split(":")[0]
        pair = (lookup[f"{spec}:original"], lookup[f"{spec}:edited"])
        if pair not in pairs:
            pairs.append(pair)
    saved = ROOT / RUN_PATH / "validation.csv"
    if file_sha256(saved) != read_json(ROOT / RUN_PATH / "metrics.json")["evaluation_sha256"]:
        raise ValueError("Saved development scores failed digest verification")
    with saved.open(newline="", encoding="utf-8") as stream:
        reference = {row["sample_id"]: row for row in csv.DictReader(stream)}

    args.output.mkdir(parents=True)
    private = args.output / "private"
    private.mkdir()
    lock_path = args.output / "lock.json"
    write_json(lock_path, {"run_id": args.output.name, "manifest_sha256": manifest_sha,
                          "demo_lock_sha256": LOCK_SHA256, "checkpoint_sha256": demo.checkpoint_sha256,
                          "logit_cutoff": demo.review_cutoff, "development_groups": groups, "cases": CASES,
                          "reference_scores_sha256": file_sha256(saved), "neutral_fill": [127, 127, 127],
                          "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_repeats": BOOTSTRAPS,
                          "threads": 2, "tile_batch": demo.batch_size, "torch": torch.__version__})

    records, backgrounds, cases = [], [], []
    for index, pair in enumerate(pairs):
        is_cohort = index < paired_count
        evidence, original, mask = pair_audit(demo, documents, pair, reference, is_cohort)
        if is_cohort:
            genuine, edited = [item for item, _ in evidence]
            backgrounds.append({"genuine": genuine, "edited": edited})
            records.extend([genuine, edited])
        for item, image in evidence:
            if item["sample_id"] in case_by_id:
                case, hypothesis = case_by_id[item["sample_id"]]
                truth = mask if item["label"] == "manipulated" else Image.new("L", image.size)
                case_item = {**item, "case": case, "hypothesis": hypothesis, "cutoff": demo.review_cutoff,
                             "mask_bbox": truth.getbbox(), "tile_mask_intersection_px": truth.crop(item["tile"]).histogram()[255],
                             "probe": case_probe(demo, case, item, image, original, mask)}
                cases.append(case_item)
                private_panel(private / f"case-{case}.png", case_item, image, truth)
        if (index + 1) % 6 == 0:
            print(f"Audited {index + 1}/{len(pairs)} development pairs", flush=True)
    cases.sort(key=lambda item: item["case"])
    if len(cases) != len(CASES):
        raise ValueError("Missing failure case")
    geometry_panel(args.output / "case-geometry.png", cases)
    for data, name in ((records, "faithfulness"), (backgrounds, "background-control"), (cases, "cases")):
        write_json(args.output / f"{name}.json", data)
    background_summary = {}
    for label in ("edit_logit", "background_logit", "logit"):
        compared = [(r["genuine"][label], r["edited"][label]) for r in backgrounds]
        deltas = [edited - original for original, edited in compared if original is not None and edited is not None]
        background_summary[label] = {"pairs": len(deltas), "edited_higher": sum(d > 0 for d in deltas),
                                     "exact_ties": sum(d == 0 for d in deltas), "mean_delta": float(np.mean(deltas)),
                                     "mean_absolute_delta": float(np.mean(np.abs(deltas)))}
    write_json(args.output / "summary.json", {"run_id": args.output.name, "lock_sha256": file_sha256(lock_path),
                                             "faithfulness": faithfulness_summary(records), "background_control": background_summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
