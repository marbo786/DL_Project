"""Native RGB receipt bags and checkpoint-addressed frozen feature caches."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from research.diagnose import select_pairs
from research.patch_data import rgb_crop, tile_boxes
from research.patch_probe import PatchNet
from research.phase_c import SIDE, STRIDE, load_rgb


class MobileFeatures(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.features = network.features
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def encode(self, tiles):
        return self.features((tiles.float() / 255 - self.mean) / self.std).mean((2, 3))


def make_encoder(name, checkpoint, device, initial_seed):
    if name == "patchnet":
        model = PatchNet()
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        dim, preprocessing = 64, "native-rgb-edge-pad-patchnet-v1"
    else:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        torch.manual_seed(initial_seed)
        if name == "mobile-pretrained":
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            network = mobilenet_v3_small(weights=None)
            network.load_state_dict(weights.get_state_dict(progress=False, check_hash=True))
        else:
            network = mobilenet_v3_small(weights=None)
        model = MobileFeatures(network)
        dim, preprocessing = 576, "native-rgb-edge-pad-imagenet-normalize-v1"

    digest = sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    model.requires_grad_(False).eval().to(device)
    return model, {"encoder": name, "state_sha256": digest.hexdigest(),
                   "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest() if name == "patchnet" else None,
                   "preprocessing": preprocessing, "side": SIDE, "stride": STRIDE, "feature_dim": dim,
                   "encoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "encoder_state_bytes": sum(array.numel() * array.element_size() for array in model.state_dict().values())}


@dataclass
class Bag:
    features: torch.Tensor
    boxes: np.ndarray
    logits: np.ndarray
    size: tuple
    seconds: float
    label: int
    source: str
    corpus: str
    path: Path


@torch.no_grad()
def extract_tiles(model, image, device, batch_size=32):
    boxes = tile_boxes(image.size, SIDE, STRIDE)
    features, logits = [], []
    for start in range(0, len(boxes), batch_size):
        crops = torch.stack([rgb_crop(image, box) for box in boxes[start:start + batch_size]]).to(device)
        embedding = model.encode(crops)
        if isinstance(model, PatchNet):
            logits.append(model.head(embedding).flatten().cpu().numpy())
        features.append(embedding.cpu().numpy())
    return np.asarray(boxes), np.concatenate(features), np.concatenate(logits) if logits else np.empty(0)


def load_bag(model, documents, index, identity, manifest_sha, directory, device):
    """Document pixels and dimensions are the only inputs to feature extraction."""
    row = documents.rows[index]
    if row["split"] not in {"train", "validation"}:
        raise ValueError("Feature extraction is restricted to development partitions")
    label = int(row["label"] == "manipulated")
    metadata = {**identity, "manifest_sha256": manifest_sha, "split": row["split"],
                "source": row["source_document_id"], "corpus": row["source_dataset"], "label": label}
    key = sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = directory / f"{key}.npz"
    if not path.exists():
        start = perf_counter()
        image = load_rgb(documents, index)
        boxes, features, logits = extract_tiles(model, image, device)
        seconds = perf_counter() - start
        pending = path.with_suffix(".tmp")
        with pending.open("wb") as stream:
            np.savez(stream, metadata=json.dumps(metadata), features=features, boxes=boxes,
                     logits=logits, size=image.size, seconds=seconds)
        pending.replace(path)
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(str(saved["metadata"])) != metadata:
            raise ValueError("Cached bag provenance mismatch")
        return Bag(torch.from_numpy(saved["features"]), saved["boxes"], saved["logits"],
                   tuple(saved["size"]), float(saved["seconds"]), label, metadata["source"],
                   metadata["corpus"], path)


def build_bags(model, documents, count, identity, manifest_sha, directory, device):
    directory.mkdir(parents=True, exist_ok=True)
    pairs = []
    for position, pair in enumerate(select_pairs(documents, count, 42), 1):
        loaded = tuple(load_bag(model, documents, index, identity, manifest_sha, directory, device) for index in pair)
        if loaded[0].label != 0 or loaded[1].label != 1 or loaded[0].source != loaded[1].source:
            raise ValueError("Invalid source-matched document labels")
        pairs.append(loaded)
        if position % 32 == 0 or position == count:
            print(f"{documents.rows[0]['split']}: cached {position}/{count} source pairs", flush=True)
    return pairs
