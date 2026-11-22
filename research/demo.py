"""Select a clearly labeled fictional demo pair outside the evaluation splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from api.signal import Predictor, digest, pixels, sigmoid
from research.data import perturb, specimen


def choose(predictor):
    config = predictor.meta
    cutoff = config["confidence_threshold"]
    best = None
    for source in range(50000, 50096):
        clean, _, _ = specimen(source, 0)
        edited, mask, _ = specimen(source, 1)
        scores = []
        for image in (clean, edited):
            logit, _ = predictor.evaluate(pixels(image))
            scores.append(float(sigmoid(logit / config["temperature"] + config["bias"])))
        margin = min(1 - scores[0], scores[1])
        if best is None or margin > best[0]:
            best = (margin, source, clean, edited, mask)
        if margin >= cutoff:
            return margin, source, clean, edited, mask
    return best


def render(release, public):
    path = release / "model.npz"
    predictor = Predictor(path)
    if predictor.meta["scope"] != "synthetic":
        raise ValueError("The curated demo requires the fictional benchmark model.")
    margin, source, clean, edited, mask = choose(predictor)
    masks = {}
    for name, image in {"original": clean, "edited": edited, "blurred": perturb(clean, "blur")}.items():
        image.save(public / f"{name}.png", optimize=True)
        masks[digest(pixels(image))] = name != "blurred"
    left, top, right, bottom = mask.getbbox()
    width, height = mask.size
    page = {"run_id": "curated-synthetic-demo", "source_id": source, "edit_region": {
        "x": left / width, "y": top / height, "width": (right - left) / width, "height": (bottom - top) / height,
    }}
    (public / "specimen.json").write_text(json.dumps(page, indent=2) + "\n", encoding="utf-8")

    predictor.meta["specimen_digests"] = masks
    np.savez_compressed(path, metadata=json.dumps(predictor.meta), **predictor.parameters)
    checkpoint = release / "model.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved["specimen_digests"] = masks
    torch.save(saved, checkpoint)
    export = release / "export.json"
    checks = json.loads(export.read_text(encoding="utf-8"))
    checks["export_size_bytes"] = path.stat().st_size
    export.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print("Curated demo source:", source, "minimum class confidence:", round(margin, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("public", type=Path)
    args = parser.parse_args()
    render(args.release, args.public)
