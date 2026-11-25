"""Compare FP32 with FX static INT8 for an explicitly research-only detector."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from research.data import ManifestDocuments
from research.receipt_model import SIDE, load_detector, inference_tiles, predict
from research.manifest import file_sha256


def quantize(model, calibration, engine):
    torch.backends.quantized.engine = engine
    example = torch.zeros((1, 3, SIDE, SIDE), dtype=torch.uint8)
    prepared = prepare_fx(model, get_default_qconfig_mapping(engine), (example,))
    with torch.inference_mode():
        for batch in calibration:
            prepared(batch)
    converted = convert_fx(prepared)
    operators = [type(layer).__module__ + "." + type(layer).__name__ for layer in converted.modules()
                 if ".quantized" in type(layer).__module__]
    if not any("Conv" in name for name in operators):
        raise RuntimeError("Quantized graph contains no INT8 convolutions")
    converted.mode = model.mode
    return converted, operators


def ranked_paths(documents, limit):
    return sorted(documents.rows, key=lambda row: sha256(row["sample_id"].encode()).digest())[:limit]


def load_image(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def serialized_bytes(model):
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def process(model, path, batch_size):
    image = load_image(path)
    score, box = predict(model, image, batch_size)
    return score, json.dumps({"score": score, "bbox": box})


def run(args):
    torch.set_num_threads(args.threads)
    calibration = ManifestDocuments(args.manifest, args.root, "calibration")
    validation = ManifestDocuments(args.manifest, args.root, "validation")
    if {row["source_group"] for row in calibration.rows} & {row["source_group"] for row in validation.rows}:
        raise ValueError("Calibration and development sources overlap")
    fp32 = load_detector(args.checkpoint)
    calibration_rows = ranked_paths(calibration, args.observer_documents)
    calibration_batches = (
        inference_tiles(load_image(args.root / row["image_path"]), fp32.mode)[1][::4]
        for row in calibration_rows
    )
    int8, operators = quantize(fp32, calibration_batches, args.engine)
    paths = [args.root / row["image_path"] for row in validation.rows]
    labels = [row["label"] == "manipulated" for row in validation.rows]
    scores = {key: [] for key in ("fp32", "int8")}
    timings = {key: [] for key in scores}
    models = {"fp32": fp32, "int8": int8}
    for model in models.values():
        for path in paths[:args.warmup]:
            process(model, path, args.batch_size)
    for trial in range(args.repetitions):
        trial_paths = paths if trial == 0 else paths[:args.timing_documents]
        for index, path in enumerate(trial_paths):
            order = tuple(models) if (index + trial) % 2 == 0 else tuple(reversed(models))
            for name in order:
                start = perf_counter()
                logit, _ = process(models[name], path, args.batch_size)
                if index < args.timing_documents:
                    timings[name].append((perf_counter() - start) * 1000)
                if trial == 0:
                    scores[name].append(logit)
    results = {
        key: {"state_dict_bytes": serialized_bytes(model),
              "median_ms": float(np.median(timings[key])), "p95_ms": float(np.quantile(timings[key], .95)),
              "auroc": float(roc_auc_score(labels, scores[key])), "ap": float(average_precision_score(labels, scores[key]))}
        for key, model in models.items()
    }
    report = {"engine": args.engine, "quantized_modules": operators, "documents": len(paths),
              "parameters_before_packing": sum(parameter.numel() for parameter in fp32.parameters()),
              "batch_size": args.batch_size, "threads": args.threads, "warmup_documents": args.warmup,
              "repetitions": args.repetitions, "timing_documents": min(len(paths), args.timing_documents),
              "observer_documents": len(calibration_rows), "results": results,
              "torch": torch.__version__, "platform": platform.platform(), "peak_memory": None,
              "manifest_sha256": file_sha256(args.manifest),
              "checkpoint_sha256": file_sha256(args.checkpoint / "model.pt")}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "root", "checkpoint", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--engine", choices=("x86", "fbgemm", "onednn", "qnnpack"), default="onednn")
    cli.add_argument("--batch-size", type=int, default=64)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--warmup", type=int, default=8)
    cli.add_argument("--repetitions", type=int, default=2)
    cli.add_argument("--timing-documents", type=int, default=32)
    cli.add_argument("--observer-documents", type=int, default=16)
    run(cli.parse_args())
