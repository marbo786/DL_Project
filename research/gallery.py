"""Render the known targets and model responses for shifted benchmark failures."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from api.signal import Predictor, pixels, sigmoid
from research.data import SYNTHETIC_SPLITS, specimen


def render(release, public):
    report = json.loads((release / "robustness.json").read_text(encoding="utf-8"))
    if report["scope"] != "synthetic":
        raise ValueError("Only the fictional failure cases can be published by this tool.")
    predictor = Predictor(release / "model.npz")
    for case in report["shifted_failure_cases"]:
        index = case["shifted_test_index"]
        base = SYNTHETIC_SPLITS["shifted_test"][0]
        original, mask, _ = specimen(base + index // 2, index % 2, shift=True)
        _, logits = predictor.evaluate(pixels(original))
        values = sigmoid(logits)
        overlay = Image.new("RGBA", original.size, (230, 115, 46, 0))
        alpha = Image.fromarray(np.uint8(np.clip((values - .25) * 210, 0, 140)))
        overlay.putalpha(alpha.resize(original.size, Image.Resampling.BILINEAR))
        combined = Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB")
        bounds = mask.getbbox()
        if bounds:
            ImageDraw.Draw(combined).rectangle(bounds, outline=(0, 113, 157), width=3)
        combined.save(public / f"failure-{index}.png", optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("public", type=Path)
    args = parser.parse_args()
    render(args.release, args.public)
