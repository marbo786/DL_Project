"""Dependency-light preprocessing and inference for the exported receipt model."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = (112, 160)
RESAMPLE = Image.Resampling.BILINEAR


def pixels(image):
    raw = np.asarray(image.convert("L").resize(SIZE, RESAMPLE), dtype=np.float32)
    return raw / 255.0 - 0.5


def digest(matrix):
    return sha256(matrix.tobytes()).hexdigest()


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -60, 60)))


def spatial_block(image, weight, bias):
    channels = image.shape[-1]
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(0, 1))[::2, ::2]
    features = windows.reshape(*windows.shape[:2], channels * 9)
    convolved = features @ weight.T + bias
    return convolved * sigmoid(convolved)


class Signal:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as artifact:
            self.meta = json.loads(str(artifact["metadata"].item()))
            self.parameters = {key: artifact[key] for key in artifact.files if key != "metadata"}
        self.version = self.meta["version"]

    def evaluate(self, gray):
        config = self.meta["config"]
        param = self.parameters
        image = gray[..., None]
        if config.get("residual", True):
            padded = np.pad(gray, 1)
            window = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
            counts = np.full(gray.shape, 9, dtype=np.float32)
            counts[[0, -1], :] = 6
            counts[:, [0, -1]] = 6
            counts[np.ix_([0, -1], [0, -1])] = 4
            residual = (gray - window.sum(axis=(-1, -2)) / counts) * 3
            image = np.stack((gray, residual), axis=-1)
        for name in ("first", "second"):
            image = spatial_block(image, param[f"{name}_weight"], param[f"{name}_bias"])
        mask = (image @ param["map_weight"].T + param["map_bias"])[..., 0]
        global_score = 0.0
        if config.get("context", True) or not config.get("local", True):
            context = spatial_block(image, param["last_weight"], param["last_bias"])
            global_score = (context.mean(axis=(0, 1)) @ param["global_weight"].T + param["global_bias"]).item()
        if not config.get("local", True):
            return global_score, mask
        flat = mask.ravel()
        local_score = np.partition(flat, -12)[-12:].mean() if config.get("topk", True) else flat.mean()
        return float(local_score + global_score), mask


def regions(heatmap, threshold=.5):
    foreground = heatmap >= threshold
    cells = {tuple(point) for point in np.argwhere(foreground)}
    h, w = heatmap.shape
    boxes = []
    while cells:
        point = min(cells)
        cells.remove(point)
        remaining = [point]
        component = [point]
        while remaining:
            y, x = remaining.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in cells:
                    cells.remove(neighbor)
                    remaining.append(neighbor)
                    component.append(neighbor)
        if len(component) < 2:
            continue
        rows, cols = zip(*component)
        boxes.append((
            max(float(heatmap[y, x]) for y, x in component),
            {"x": min(cols) / w, "y": min(rows) / h, "width": (max(cols) - min(cols) + 1) / w,
             "height": (max(rows) - min(rows) + 1) / h},
        ))
    return [box for _, box in sorted(boxes, key=lambda pair: pair[0], reverse=True)[:5]]


class Predictor(Signal):
    def predict(self, image):
        matrix = pixels(image)
        scope = self.meta["scope"]
        specimen = self.meta["specimen_digests"].get(digest(matrix))
        if scope == "synthetic" and specimen is None:
            return {
                "decision": "inconclusive",
                "warnings": ["This checkpoint has only been validated on fictional specimens. Decision and heatmap are disabled for other documents."],
            }
        score, grid = self.evaluate(matrix)
        probability = float(sigmoid(score / self.meta["temperature"] + self.meta.get("bias", 0.0)))
        mask = sigmoid(grid)
        rejected = not specimen and scope == "synthetic"
        rejected |= max(probability, 1 - probability) < self.meta["confidence_threshold"]
        suggested = probability >= .5
        decision = "inconclusive" if rejected else "suspicious" if suggested else "no_strong_evidence"
        warnings = []
        if scope == "synthetic":
            warnings.append("This score is calibrated on fictional receipts only.")
        if rejected:
            warnings.append("The image is outside the trusted operating range or the model confidence is too low.")
        return {
            "decision": decision,
            "predicted_class": None if rejected else "manipulated" if suggested else "genuine",
            "calibrated_probability": None if scope == "synthetic" and not specimen else probability,
            "uncertainty": None if scope == "synthetic" and not specimen else float(2 * min(probability, 1 - probability)),
            "heatmap": {"width": grid.shape[1], "height": grid.shape[0], "values": mask.ravel().tolist()},
            "regions": regions(mask) if suggested else [],
            "warnings": warnings,
        }


def create_predictor(path=None):
    base = Path(__file__).resolve().parents[1]
    return Predictor(path or base / "release" / "model.npz")
