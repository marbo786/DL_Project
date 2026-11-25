"""Run pinned TruFor document inference without storing private rasters."""
import argparse
import csv
from hashlib import sha1
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps
import psutil
import torch
from torch.nn import functional as F

from research.manifest import file_sha256, resolve_image
from research.patch_data import tile_boxes
from research.receipt.independent_metrics import assess, check_manifest
from research.receipt.trufor import input_name

ARTIFACTS = Path("artifacts")
UPSTREAM = ARTIFACTS / "trufor-upstream"
WEIGHTS = Path("weights/trufor.pth.tar")
RECEIPTS = ARTIFACTS / "receipt-protocol-v1-contained/manifest.csv"
PANEL = ARTIFACTS / "trufor-native-panel-v1"
FORMS = ARTIFACTS / "receipt-xfund-v1/manifest.csv"
FIELDS = ("sample_id", "source_group", "corpus", "label", "logit", "iou", "hit", "bbox",
          "milliseconds", "condition", "native_mask_iou_at_0_5", "peak_working_set_bytes")


def bounded_attention(layer, tokens, height, width, original):
    batch, count, channels = tokens.shape
    heads = layer.num_heads
    reduction = layer.sr_ratio
    key_count = (height // reduction) * (width // reduction)
    if count * key_count * heads <= 12_000_000:
        return original(layer, tokens, height, width)
    if layer.training:
        raise ValueError("TruFor bounded attention is inference-only")
    head_size = channels // heads
    query = layer.q(tokens).reshape(batch, count, heads, head_size).permute(0, 2, 1, 3)
    if reduction > 1:
        grid = tokens.permute(0, 2, 1).reshape(batch, channels, height, width)
        tokens = layer.norm(layer.sr(grid).reshape(batch, channels, -1).permute(0, 2, 1))
    key, value = layer.kv(tokens).reshape(batch, -1, 2, heads, head_size).permute(2, 0, 3, 1, 4)
    attended = F.scaled_dot_product_attention(query, key, value, scale=layer.scale)
    return layer.proj_drop(layer.proj(attended.transpose(1, 2).reshape(batch, count, channels)))


def model_from_official_files():
    provenance = json.loads((UPSTREAM / "provenance.json").read_text(encoding="utf-8"))
    if file_sha256(WEIGHTS) != provenance["checkpoint_sha256"]:
        raise ValueError("Official checkpoint SHA-256 mismatch")
    for filename, expected in provenance["source_git_blobs"].items():
        data = (UPSTREAM / "src" / filename).read_bytes()
        normalized = data.replace(b"\r\n", b"\n")
        hashes = {sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
                  for raw in (data, normalized)}
        if expected not in hashes:
            raise ValueError(f"Pinned source mismatch: {filename}")
    sys.path[:0] = [str((UPSTREAM / folder).resolve()) for folder in ("src", "deps")]
    from config import _C as config
    from models.cmx.builder_np_conf import myEncoderDecoder
    from models.cmx.encoders.dual_segformer import Attention

    original = Attention.forward

    def forward(layer, tokens, height, width):
        return bounded_attention(layer, tokens, height, width, original)

    Attention.forward = forward

    config.merge_from_file(str(UPSTREAM / "src/trufor.yaml"))
    config.freeze()
    allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
               *[type(np.dtype(t)) for t in (np.float16, np.float32, np.float64,
                                              np.int32, np.int64, np.bool_)]]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(WEIGHTS, weights_only=True, map_location="cpu")
    model = myEncoderDecoder(cfg=config)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.eval(), provenance


@torch.inference_mode()
def infer(model, image):
    rgb = torch.tensor(np.asarray(image).transpose(2, 0, 1), dtype=torch.float32)[None] / 256.0
    pred, confidence, detection, _ = model(rgb)
    heatmap = pred[0].softmax(dim=0)[1].numpy()
    whole_score = detection.sigmoid().item()
    if (heatmap.shape != (image.height, image.width) or confidence.shape[-2:] != heatmap.shape
            or not np.isfinite(heatmap).all() or not torch.isfinite(confidence).all()
            or not 0 <= whole_score <= 1):
        raise ValueError("Invalid native-resolution semantic output")
    return whole_score, heatmap


def score_row(model, row, root, process, panel_reference=None):
    start = perf_counter()
    with Image.open(resolve_image(root, row["image_path"])) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    document_score, heatmap = infer(model, image)
    if panel_reference is not None:
        with np.load(panel_reference, allow_pickle=False) as original:
            if (abs(document_score - float(original["score"].item())) > 2e-5
                    or not np.allclose(heatmap, original["map"], rtol=1e-4, atol=2e-5)):
                raise ValueError("Official panel score or heatmap parity failed")
    boxes = [(x, y, min(x2, image.width), min(y2, image.height))
             for x, y, x2, y2 in tile_boxes(image.size, 112, 112)]
    x, y, x2, y2 = max(boxes, key=lambda box: heatmap[box[1]:box[3], box[0]:box[2]].mean())
    elapsed = (perf_counter() - start) * 1000
    label = row["label"] == "manipulated"
    iou, hit, native_iou = "", "", ""
    if label:
        with Image.open(resolve_image(root, row["mask_path"])) as source:
            mask = np.asarray(ImageOps.exif_transpose(source).convert("L")) > 0
        if not mask.any() or mask.shape != heatmap.shape:
            raise ValueError("Invalid localization target")
        intersection = mask[y:y2, x:x2].sum()
        iou = intersection / (mask.sum() + (x2 - x) * (y2 - y) - intersection)
        hit = int(intersection > 0)
        detected = heatmap >= .5
        native_iou = float((mask & detected).sum() / (mask | detected).sum())
    memory = process.memory_info()
    return (row["sample_id"], row["source_group"], row["source_dataset"], int(label), document_score,
            iou, hit, f"{x} {y} {x2} {y2}", elapsed, "clean", native_iou,
            getattr(memory, "peak_wset", memory.rss))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate_completed(output):
    calibration = output / "trufor-calibration.csv"
    if not calibration.with_suffix(".provenance.json").is_file():
        return
    receipts = read_rows(RECEIPTS)
    calibration_rows = read_rows(calibration)
    check_manifest(calibration_rows, receipts, "calibration")
    historical = ARTIFACTS / "receipt-retrospective-v1"
    alternatives = {
        "validation": ARTIFACTS / "receipt-study-v1/residual-42/validation.csv",
        "test": historical / "test.csv", "shifted_test": historical / "shifted_test.csv",
        "xfund": ARTIFACTS / "receipt-xfund-v1/scores.csv",
    }
    for role, comparator in alternatives.items():
        predictions = output / f"trufor-{role}.csv"
        target = predictions.with_suffix(".metrics.json")
        manifest = read_rows(FORMS) if role == "xfund" else receipts
        if target.exists() or not predictions.with_suffix(".provenance.json").is_file():
            continue
        scores = read_rows(predictions)
        check_manifest(scores, manifest, "test" if role == "xfund" else role)
        result = assess(scores, calibration_rows, read_rows(comparator))
        edits = [row for row in scores if row["label"] == "1"]
        result.update({
            "mean_tile_iou": float(np.mean([float(row["iou"]) for row in edits])),
            "tile_hit_rate": float(np.mean([int(row["hit"]) for row in edits])),
            "mean_native_mask_iou_at_0_5": float(np.mean([float(row["native_mask_iou_at_0_5"]) for row in edits])),
            "inference_time_ms_p50_p95": np.quantile([float(row["milliseconds"]) for row in scores], [.5, .95]).tolist(),
            "peak_working_set_bytes": max(int(row["peak_working_set_bytes"]) for row in scores),
        })
        with target.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--roles", nargs="+", required=True,
                        choices=("panel", "calibration", "validation", "test", "shifted_test", "xfund"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    model, provenance = model_from_official_files()
    process = psutil.Process()
    for role in args.roles:
        path = PANEL / "manifest.csv" if role == "panel" else FORMS if role == "xfund" else RECEIPTS
        root = (Path("AIForge-Doc-v2.0") if role == "xfund" else Path("data/aiforge/assets")).resolve()
        rows = [row for row in read_rows(path) if role in ("panel", "xfund") or row["split"] == role]
        if not rows:
            raise ValueError(f"No rows for {role}")
        if role == "panel":
            historical = {r["sample_id"]: float(r["logit"]) for panel_role in
                          ("calibration", "validation", "test", "shifted_test")
                          for r in read_rows(PANEL / f"trufor-{panel_role}.csv")}
        out = args.output / f"trufor-{role}.csv"
        completed = read_rows(out) if out.exists() else []
        known_ids = {row["sample_id"] for row in completed}
        expected_ids = {row["sample_id"] for row in rows}
        if (len(known_ids) != len(completed) or known_ids - expected_ids
                or any(None in row or None in row.values() for row in completed)):
            raise ValueError("Invalid partial output")
        with out.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            if not completed:
                writer.writerow(FIELDS)
            for index, row in enumerate(rows, start=1):
                if row["sample_id"] in known_ids:
                    continue
                reference = PANEL / "predictions" / (input_name(row) + ".npz") if role == "panel" else None
                result = score_row(model, row, root, process, reference)
                if role == "panel" and abs(result[4] - historical[row["sample_id"]]) > 2e-5:
                    raise ValueError("Official panel document score parity failed")
                writer.writerow(result)
                stream.flush()
                if index <= 2 or index % 50 == 0 or index == len(rows):
                    print(f"{role}: {index}/{len(rows)} ({result[8]:.0f} ms last)", flush=True)
        (args.output / f"trufor-{role}.provenance.json").write_text(json.dumps({
            "manifest_sha256": file_sha256(path), "scores_sha256": file_sha256(out),
            "source_commit": provenance["source_commit"], "checkpoint_sha256": provenance["checkpoint_sha256"],
            "preprocessing": "EXIF transpose and RGB / 256 at native dimensions",
            "document_score": "official sigmoid detection head",
            "attention": "original small attention, CPU scaled dot-product attention for larger tensors",
            "torch": torch.__version__, "threads": args.threads, "documents": len(rows),
        }, indent=2) + "\n", encoding="utf-8")
        evaluate_completed(args.output)


if __name__ == "__main__":
    main()
