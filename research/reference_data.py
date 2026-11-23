"""Source-paired receipt data and reproducible acquisition perturbations."""

from dataclasses import dataclass
from hashlib import sha256

import cv2
import numpy as np

from research.data import ManifestDocuments
from research.diagnose import select_pairs
from research.patch_data import load_pair


PINNED_MANIFEST = "3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80"
REGIMES = ("exact", "transformed", "difficult")


@dataclass
class Comparison:
    reference: np.ndarray
    candidate: np.ndarray
    label: int
    mask: np.ndarray
    valid: np.ndarray
    transform: np.ndarray
    source: str
    corpus: str


def random_stream(source, label, regime, seed):
    token = sha256(f"{seed}|{source}|{label}|{regime}".encode()).digest()
    return np.random.default_rng(int.from_bytes(token[:8], "big"))


def project(image, matrix, shape, nearest=False, reflect=False):
    return cv2.warpPerspective(image, matrix, (shape[1], shape[0]),
                               flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101 if reflect else cv2.BORDER_CONSTANT)


def nuisance(image, edit_geometry, rng, regime):
    shape = image.shape[:2]
    height, width = shape
    if regime == "exact":
        return image.copy(), edit_geometry.copy(), np.ones(shape, np.uint8), np.eye(3)
    strong = regime == "difficult"
    angle = 6 if strong else 2.5
    scale = (.92, 1.13) if strong else (.97, 1.045)
    shift = .065 if strong else .023
    perspective = .036 if strong else .010
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    mask_count = int(edit_geometry.sum())
    for _ in range(20):
        affine = cv2.getRotationMatrix2D((width / 2, height / 2), rng.uniform(-angle, angle), rng.uniform(*scale))
        affine[:, 2] += rng.uniform(-shift, shift, 2) * [width, height]
        perturbed = corners @ affine[:, :2].T + affine[:, 2]
        perturbed += rng.uniform(-perspective, perspective, corners.shape) * [width, height]
        matrix = cv2.getPerspectiveTransform(corners, np.float32(perturbed))
        target = project(edit_geometry, matrix, shape, nearest=True)
        if target.sum() >= max(1, .65 * mask_count):
            break
    else:
        matrix, target = np.eye(3), edit_geometry.copy()
    valid = project(np.ones(shape, np.uint8), matrix, shape, nearest=True)
    changed = project(image, matrix, shape, reflect=True).astype(np.float32)
    contrast = rng.uniform(.7, 1.25) if strong else rng.uniform(.87, 1.13)
    brightness = rng.uniform(-32, 28) if strong else rng.uniform(-15, 15)
    changed = (changed - 127.5) * contrast + 127.5 + brightness
    changed *= rng.uniform(.92 if strong else .97, 1.08 if strong else 1.03, 3)
    sigma = rng.uniform(.15, 1.6 if strong else .7)
    changed = cv2.GaussianBlur(changed, (0, 0), sigma)
    changed += rng.normal(0, rng.uniform(0, 5 if strong else 2.5), changed.shape)
    encoded = np.clip(changed, 0, 255).astype(np.uint8)
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(42 if strong else 65, 96))])
    jpegged = cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return jpegged, target, valid, matrix


def pairs(manifest, root, split, count, regimes=REGIMES, seed=42, long_side=512):
    if split not in ("train", "validation"):
        raise ValueError("Reference experiments only accept development partitions")
    data = ManifestDocuments(manifest, root, split)
    for pair in select_pairs(data, count, 42):
        original, edited, mask_image = load_pair(data, pair)
        row = data.rows[pair[0]]
        source = row["source_document_id"]
        width, height = original.size
        size = (max(1, round(width * long_side / max(width, height))),
                max(1, round(height * long_side / max(width, height))))
        reference = cv2.resize(np.asarray(original), size, interpolation=cv2.INTER_AREA)
        altered = cv2.resize(np.asarray(edited), size, interpolation=cv2.INTER_AREA)
        # Area-to-binary projection retains edits that are narrower than a pixel.
        geometry = (cv2.resize(np.asarray(mask_image, dtype=np.float32), size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        if not geometry.any():
            raise ValueError("Edit disappeared during common-raster conversion")
        for regime in regimes:
            items = []
            for label, image in enumerate((reference, altered)):
                changed, target, valid, matrix = nuisance(image, geometry, random_stream(source, label, regime, seed), regime)
                items.append(Comparison(reference, changed, label, target if label else np.zeros_like(target),
                                        valid, matrix, source, row["source_dataset"]))
            yield regime, items
