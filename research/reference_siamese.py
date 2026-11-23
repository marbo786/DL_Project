"""Compact shared-weight comparison on fixed tensors in a common pixel frame."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from api.network import block
from research.reference_compare import register, regions
from research.reference_data import PINNED_MANIFEST, pairs, project
from research.reference_metrics import calibration_split, pixel_metrics, summary


SIZE = (160, 224)
DEVELOPMENT_REGIMES = ("transformed", "difficult")
CONFIGURATIONS = ("unregistered", "registered", "localized")


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(block(3, 8), block(8, 16))
        self.fuse = nn.Conv2d(32, 16, 3, padding=1)
        self.classify = nn.Linear(32, 1)
        self.localize = nn.Conv2d(16, 1, 1)

    def forward(self, reference, candidate, valid):
        a, b = self.encoder(reference.float() / 255 - .5), self.encoder(candidate.float() / 255 - .5)
        fused = F.silu(self.fuse(torch.cat(((a - b).abs(), a * b), dim=1)))
        valid = F.avg_pool2d(valid[:, None].float(), 4, stride=4) >= .99
        maximum = fused.masked_fill(~valid, -1e4).amax((2, 3))
        mean = (fused * valid).sum((2, 3)) / valid.sum((2, 3)).clamp(min=1)
        score = self.classify(torch.cat((maximum, mean), dim=1)).flatten()
        return score, self.localize(fused)[:, 0], valid[:, 0]


def common_raster(image, nearest=False):
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(image, SIZE, interpolation=interpolation)


def sample_tensors(item, aligned=None):
    shape = item.reference.shape[:2]
    candidate = item.candidate if aligned is None else aligned.image
    valid = np.ones(shape, dtype=np.uint8) if aligned is None else aligned.valid.astype(np.uint8)
    actual = item.valid if aligned is None else project(item.valid, aligned.matrix, shape, nearest=True)
    target = item.mask if aligned is None else project(item.mask, aligned.matrix, shape, nearest=True)
    return (common_raster(item.reference).transpose(2, 0, 1),
            common_raster(candidate).transpose(2, 0, 1),
            common_raster(valid, nearest=True),
            common_raster(target, nearest=True),
            common_raster(actual, nearest=True))


def build_cache(args, split, count):
    path = args.cache / f"v1-{split}-{count}.npz"
    if path.exists():
        return path
    if split not in ("train", "validation"):
        raise ValueError("Final partitions are outside the experiment")
    images, labels, groups, accepted, milliseconds = [], [], [], [], []
    for index, (regime, group) in enumerate(pairs(args.manifest, args.asset_root, split, count, DEVELOPMENT_REGIMES)):
        for item in group:
            start = perf_counter()
            raw = sample_tensors(item)
            raw_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            alignment = register(item.reference, item.candidate)
            aligned = sample_tensors(item, alignment)
            registered_ms = (perf_counter() - start) * 1000
            images.append((raw, aligned))
            labels.append(item.label)
            groups.append((regime, item.source))
            accepted.append(alignment.accepted)
            milliseconds.append((raw_ms, registered_ms))
        if (index + 1) % 96 == 0:
            print(f"{split}: cached {index + 1} source-regime pairs", flush=True)
    arrays = {f"view{view}_{field}": np.stack([item[view][field] for item in images])
              for view in range(2) for field in range(5)}
    pending = path.with_suffix(".tmp")
    with pending.open("wb") as stream:
        np.savez(stream, **arrays, labels=np.asarray(labels), groups=np.asarray(groups),
                 accepted=np.asarray(accepted), milliseconds=np.asarray(milliseconds))
    pending.replace(path)
    return path


def read_cache(path, view):
    with np.load(path, allow_pickle=False) as source:
        arrays = [torch.from_numpy(source[f"view{view}_{field}"]) for field in range(5)]
        labels = torch.from_numpy(source["labels"]).float()
        return TensorDataset(*arrays, labels), source["groups"], source["accepted"], source["milliseconds"][:, view]


def train_model(data, seed, epochs, device, supervised):
    torch.manual_seed(seed)
    model = SiameseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loader = DataLoader(data, batch_size=12, shuffle=True, generator=torch.Generator().manual_seed(seed))
    curve = []
    for epoch in range(epochs):
        model.train()
        total = 0.
        for reference, candidate, valid, target, actual, labels in loader:
            reference, candidate, valid, target, actual, labels = [
                tensor.to(device) for tensor in (reference, candidate, valid, target, actual, labels)]
            score, heat, support = model(reference, candidate, valid)
            loss = F.binary_cross_entropy_with_logits(score, labels)
            if supervised:
                target_grid = F.max_pool2d(target[:, None].float(), 4, stride=4)[:, 0]
                usable = support & (F.avg_pool2d(actual[:, None].float(), 4, stride=4)[:, 0] >= .99)
                pixel = F.binary_cross_entropy_with_logits(heat, target_grid, pos_weight=heat.new_tensor(8), reduction="none")
                loss += .35 * (pixel * usable).sum() / usable.sum().clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        curve.append({"epoch": epoch + 1, "loss": total / len(data)})
        print(f"epoch {epoch + 1}: training loss {curve[-1]['loss']:.5f}", flush=True)
    return model.eval(), curve


@torch.inference_mode()
def evaluate(model, data, groups, accepted, preprocessing, device, registered, supervised):
    records = defaultdict(list)
    for index, (reference, candidate, valid, target, actual, label) in enumerate(data):
        start = perf_counter()
        score, spatial, _ = model(reference[None].to(device), candidate[None].to(device), valid[None].to(device))
        score_value = float(score)
        probability = spatial[0].sigmoid().cpu().numpy()
        heatmap = cv2.resize(probability.astype(np.float32), SIZE, interpolation=cv2.INTER_LINEAR)
        foreground = regions(heatmap, valid.numpy().astype(bool), .5) if supervised else np.zeros_like(target, dtype=bool)
        elapsed = (perf_counter() - start) * 1000 + preprocessing[index]
        regime, source = groups[index]
        pair_valid = valid.numpy().astype(bool) & actual.numpy().astype(bool)
        records[regime].append({"source": source, "score": score_value, "label": int(label),
                                "accepted": bool(accepted[index]) if registered else True, "milliseconds": elapsed,
                                **pixel_metrics(foreground, target.numpy(), pair_valid)})
    result = {}
    for regime, rows in records.items():
        result[regime] = summary(rows, calibration_split([row["source"] for row in rows]))
        if not supervised:
            result[regime]["pixel_iou"] = None
            result[regime]["localization_f1"] = None
    return result


def run(args):
    if sha256(args.manifest.read_bytes()).hexdigest() != PINNED_MANIFEST:
        raise ValueError("Unexpected manifest checksum")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(42)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_file = build_cache(args, "train", args.train_pairs)
    val_file = build_cache(args, "validation", args.val_pairs)
    results = {}
    for name in args.configs:
        view = int(name != "unregistered")
        training, train_groups, *_ = read_cache(train_file, view)
        validation, val_groups, accepted, preprocessing = read_cache(val_file, view)
        if set(train_groups[:, 1]) & set(val_groups[:, 1]):
            raise ValueError("A source crosses development partitions")
        for seed in args.seeds:
            start = perf_counter()
            print(f"{name}, seed {seed}", flush=True)
            model, curve = train_model(training, seed, args.epochs, device, name == "localized")
            elapsed = perf_counter() - start
            key = f"{name}-{seed}"
            results[key] = {"training_seconds": elapsed, "curve": curve,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "validation": evaluate(model, validation, val_groups, accepted, preprocessing,
                                                   device, bool(view), name == "localized")}
            torch.save(model.state_dict(), args.output / f"{key}.pt")
            (args.output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "asset-root", "cache", "output"):
        p.add_argument("--" + option, type=Path, required=True)
    p.add_argument("--configs", nargs="+", choices=CONFIGURATIONS, default=CONFIGURATIONS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--train-pairs", type=int, default=128)
    p.add_argument("--val-pairs", type=int, default=228)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--threads", type=int, default=5)
    a = p.parse_args()
    if min(a.train_pairs, a.val_pairs, a.epochs, a.threads) < 1:
        p.error("budgets must be positive")
    run(a)
