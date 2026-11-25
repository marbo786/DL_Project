"""Measure isolated CPU working sets for validated FP32 and INT8 TorchScript exports."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import psutil
import torch

from research.data import ManifestDocuments
from research.manifest import file_sha256
from research.receipt_efficiency import load_image, process, quantize, ranked_paths
from research.receipt_model import inference_tiles, load_detector


def worker(args):
    torch.set_num_threads(args.threads)
    torch.backends.quantized.engine = "onednn"
    model = torch.jit.load(str(args.worker), map_location="cpu").eval()
    model.mode = args.mode
    documents = ManifestDocuments(args.manifest, args.root, "validation")
    paths = [args.root / row["image_path"] for row in ranked_paths(documents, 32)]
    for path in paths[:8]:
        process(model, path, 64)
    elapsed = []
    for _ in range(2):
        for path in paths:
            start = perf_counter()
            process(model, path, 64)
            elapsed.append((perf_counter() - start) * 1000)
    info = psutil.Process().memory_info()
    print(json.dumps({"documents": len(paths), "warmup_documents": 8, "repetitions": 2, "tile_batch": 64,
                      "threads": args.threads, "median_ms": float(np.median(elapsed)),
                      "p95_ms": float(np.quantile(elapsed, .95)),
                      "peak_working_set_bytes": getattr(info, "peak_wset", None),
                      "script_bytes": args.worker.stat().st_size, "script_sha256": file_sha256(args.worker)}))


def run(args):
    torch.set_num_threads(args.threads)
    fp32 = load_detector(args.checkpoint)
    calibration = ManifestDocuments(args.manifest, args.root, "calibration")
    rows = ranked_paths(calibration, 16)
    batches = [inference_tiles(load_image(args.root / row["image_path"]), fp32.mode)[1][::4] for row in rows]
    int8, operators = quantize(fp32, batches, "onednn")
    args.output.mkdir(parents=True)
    results = {}
    for name, model in (("fp32", fp32), ("int8", int8)):
        traced = torch.jit.trace(model, batches[0][:8])
        destination = args.output / f"{name}.torchscript"
        traced.save(str(destination))
        loaded = torch.jit.load(str(destination), map_location="cpu").eval()
        with torch.inference_mode():
            for item in batches[:3]:
                torch.testing.assert_close(loaded(item), model(item), rtol=1e-5, atol=1e-6)
        child = subprocess.run(
            [sys.executable, "-m", "research.receipt_memory", "--worker", str(destination),
             "--mode", fp32.mode, "--manifest", str(args.manifest), "--root", str(args.root),
             "--threads", str(args.threads)], check=True, capture_output=True, text=True,
        )
        results[name] = json.loads(child.stdout)
    report = {"torch": torch.__version__, "psutil": psutil.__version__, "engine": "onednn",
              "checkpoint_sha256": file_sha256(args.checkpoint / "model.pt"), "observer_documents": len(rows),
              "quantized_modules": operators, "export_parity_batches_per_model": 3, "results": results}
    (args.output / "runtime.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("manifest", "root"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("checkpoint", "output", "worker"):
        parser.add_argument("--" + field, type=Path)
    parser.add_argument("--mode")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        if not args.mode:
            parser.error("worker requires the exported detector mode")
        worker(args)
    else:
        if not args.checkpoint or not args.output:
            parser.error("comparison requires a checkpoint and fresh output directory")
        run(args)
