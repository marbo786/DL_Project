"""Export a trusted PyTorch checkpoint to a small, pickle-free NumPy artifact."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from api.network import ReceiptNet
from api.signal import Predictor, pixels
from research.data import specimen


def arrays(model):
    result = {}
    for name in ("first", "second", "last"):
        conv, norm, _ = getattr(model, name)
        weight, scale = conv.weight.detach(), norm.weight.detach()
        factor = scale * torch.rsqrt(norm.running_var + norm.eps)
        result[f"{name}_weight"] = (weight.flatten(1) * factor[:, None]).numpy()
        result[f"{name}_bias"] = (norm.bias.detach() - norm.running_mean * factor).numpy()
    for name, layer in (("map", model.map), ("global", model.global_head)):
        result[f"{name}_weight"] = layer.weight.detach().flatten(1).numpy()
        result[f"{name}_bias"] = layer.bias.detach().numpy()
    return result


def export(checkpoint, destination):
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ReceiptNet(**saved["config"]).eval()
    model.load_state_dict(saved["weights"])
    metadata = {key: value for key, value in saved.items() if key != "weights"}
    np.savez_compressed(destination, metadata=json.dumps(metadata), **arrays(model))

    artifact = Predictor(destination)
    errors = {"score_logit": 0.0, "heatmap_logit": 0.0}
    elapsed = []
    with torch.inference_mode():
        for index in range(80):
            image, _, _ = specimen(40000 + index // 2, index % 2, shift=index % 3 == 0)
            image = pixels(image)
            expected_score, expected_heatmap = model(torch.from_numpy(image)[None, None])
            now = perf_counter()
            score, heatmap = artifact.evaluate(image)
            if index >= 10:
                elapsed.append((perf_counter() - now) * 1000)
            errors["score_logit"] = max(errors["score_logit"], abs(score - expected_score.item()))
            errors["heatmap_logit"] = max(errors["heatmap_logit"], float(np.abs(heatmap - expected_heatmap.numpy()[0, 0]).max()))
    if max(errors.values()) > .0001:
        raise RuntimeError(f"Export parity failed: {errors}")
    report = {
        "parity_examples": 80, "max_abs_error": errors, "export_size_bytes": destination.stat().st_size,
        "batch_1_cpu_median_ms": float(np.median(elapsed)), "batch_1_cpu_p95_ms": float(np.percentile(elapsed, 95)),
        "timing_scope": "NumPy forward pass only, after 10 warm-up images",
    }
    destination.with_name("export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.destination)
