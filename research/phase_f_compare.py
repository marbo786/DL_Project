"""Frozen Phase F registered SSIM adapter, shared by evaluation and Studio."""

import json
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from research import reference_compare

POLICY_PATH = Path(__file__).with_name("phase_f_policy.json")
POLICY = json.loads(POLICY_PATH.read_bytes())
THRESHOLDS = POLICY["score_thresholds_strict_greater_than"]
PRIMARY = THRESHOLDS[POLICY["primary_nominal_fpr"]]
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000


def verify_freeze():
    for name, expected in POLICY["source_lf_sha256"].items():
        source = POLICY_PATH.with_name(name).read_bytes().replace(b"\r\n", b"\n")
        if sha256(source).hexdigest() != expected:
            raise RuntimeError(f"{name} differs from the frozen revision")


def decode(encoded):
    if not 0 < len(encoded) <= MAX_BYTES:
        raise ValueError("Use a nonempty image no larger than 8 MiB.")
    try:
        with Image.open(BytesIO(encoded), formats=("JPEG", "PNG", "WEBP")) as image:
            if image.width * image.height > MAX_PIXELS or getattr(image, "is_animated", False):
                raise ValueError("Use a single-frame image with at most 16 megapixels.")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError("Use a readable JPEG, PNG or WebP image.") from error


@dataclass
class Result:
    reason: str | None
    reference: np.ndarray | None = None
    alignment: reference_compare.Alignment | None = None
    heatmap: np.ndarray | None = None
    mask: np.ndarray | None = None
    score: float | None = None

    @property
    def decision(self):
        if self.reason:
            return "inconclusive"
        return "difference_detected" if self.score > PRIMARY else "no_difference_flagged"


def compare(reference, candidate):
    width, height = reference.size
    scale = POLICY["long_side"] / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if min(*size, *reference.size, *candidate.size) < 24:
        return Result("unsupported_dimensions")
    ref = cv2.resize(np.asarray(reference), size, interpolation=cv2.INTER_AREA)
    can = cv2.resize(np.asarray(candidate), size, interpolation=cv2.INTER_AREA)
    alignment = reference_compare.register(ref, can)
    if not alignment.accepted:
        return Result("registration_rejected", alignment=alignment)
    heatmap = reference_compare.difference(ref, alignment.image, alignment.valid, "ssim")
    mask = reference_compare.regions(heatmap, alignment.valid, POLICY["pixel_threshold_strict_greater_than"])
    score = reference_compare.score_map(heatmap, alignment.valid)
    return Result(None, ref, alignment, heatmap, mask, score)
