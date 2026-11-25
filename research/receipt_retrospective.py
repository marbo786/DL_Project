"""Lock a development-selected recipe before scoring previously studied holdouts."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from research.manifest import file_sha256
from research.receipt_metrics import evaluate
from research.receipt_model import MODES, score
from research.receipt_protocol import verify_groups


def run(args):
    verify_groups(args.manifest)
    torch.set_num_threads(args.threads)
    matrix_path = args.study / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    modes = [name for name in MODES if name in matrix]
    if len(modes) != len(MODES) or any(len(matrix[mode]["seeds"]) != 3 for mode in modes):
        raise ValueError("Complete the preregistered matrix before selecting a recipe")
    mode = max(modes, key=lambda key: (matrix[key]["mean_auroc"], matrix[key]["mean_ap"]))
    chosen = args.study / f"{mode}-42"
    metadata = json.loads((chosen / "fit.json").read_text(encoding="utf-8"))
    report = json.loads((chosen / "metrics.json").read_text(encoding="utf-8"))
    calibration = chosen / "calibration.csv"
    if (metadata["training_manifest_sha256"] != file_sha256(args.manifest)
            or report["calibration_sha256"] != file_sha256(calibration)):
        raise ValueError("Historical probe inputs differ from the fitted protocol")
    args.output.mkdir(parents=True)
    freeze = {"evaluation_status": "retrospective_historical", "configuration": mode, "seed": 42,
              "checkpoint_sha256": metadata["checkpoint_sha256"], "matrix_sha256": file_sha256(matrix_path),
              "manifest_sha256": metadata["training_manifest_sha256"],
              "calibration_sha256": report["calibration_sha256"], "calibration": report["calibration"],
              "operating_points": {key: point["logit_threshold"] for key, point in report["operating_points"].items()}}
    (args.output / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    for split in ("test", "shifted_test"):
        table = args.output / f"{split}.csv"
        score(SimpleNamespace(manifest=args.manifest, root=args.root, checkpoint=chosen, batch_size=args.batch_size,
                              split=split, output=table, condition="clean", limit_sources=None))
        result = evaluate(calibration, table, args.output / f"{split}.json")
        print(f"{split}: retrospective AUROC={result['auroc']:.4f}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "root", "study", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--batch-size", type=int, default=64)
    run(cli.parse_args())
