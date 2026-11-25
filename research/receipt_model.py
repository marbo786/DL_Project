"""Train and score an ImageNet-initialized receipt detector with image-only inference."""

import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small

from research.data import ManifestDocuments
from research.manifest import file_sha256
from research.patch_data import load_pair, rgb_crop, tile_boxes
from research.receipt_protocol import verify_groups

SIDE = 112
MODES = ("frozen", "partial", "full", "residual", "whole")
CONDITIONS = ("clean", "jpeg-1", "jpeg-2", "blur-1", "blur-2", "resize-1", "resize-2",
              "brightness-1", "brightness-2", "noise-1", "noise-2", "combined")


class Detector(nn.Module):
    def __init__(self, mode, weights=None, *, trainable_tail=None):
        super().__init__()
        self.mode = mode
        backbone = mobilenet_v3_small()
        if weights is not None:
            backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.features = backbone.features
        self.head = nn.Linear(576, 1)
        self.frozen_blocks = len(self.features) if mode == "frozen" else len(self.features) - 3 if mode == "partial" else 0
        if trainable_tail is not None:
            self.frozen_blocks = len(self.features) - trainable_tail
        for block in self.features[:self.frozen_blocks]:
            block.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        for block in self.features[:self.frozen_blocks]:
            block.eval()
        return self

    def forward(self, images):
        pixels = images.float() / 255
        if self.mode == "residual":
            smooth = F.avg_pool2d(pixels, 3, stride=1, padding=1, count_include_pad=False)
            pixels = (pixels + .5 * (pixels - smooth)).clamp(0, 1)
        features = self.features(((pixels - self.mean) / self.std).contiguous(memory_format=torch.channels_last))
        return self.head(features.mean(dim=(2, 3))).flatten()


def whole_tensor(image):
    return rgb_crop(image.resize((SIDE, SIDE), Image.Resampling.BILINEAR), (0, 0, SIDE, SIDE))


def inference_tiles(image, mode):
    if mode == "whole":
        return [(0, 0, image.width, image.height)], whole_tensor(image)[None]
    boxes = tile_boxes(image.size, SIDE, SIDE)
    return boxes, torch.stack([rgb_crop(image, box) for box in boxes])


def training_data(documents, fraction, mode):
    by_spec = {}
    for index, row in enumerate(documents.rows):
        slots = by_spec.setdefault(row["sample_id"].split(":")[0], {})
        label = row["label"] == "manipulated"
        if label in slots:
            raise ValueError("Duplicate class within matched spec")
        slots[label] = index
    groups = sorted({row["source_group"] for row in documents.rows},
                    key=lambda group: sha256(f"receipt-learning-v1:{group}".encode()).digest())
    selected = set(groups[:max(1, int(len(groups) * fraction))])
    tensors, targets = [], []
    trained = set()
    for indices in by_spec.values():
        original_index, edited_index = indices[False], indices[True]
        group = documents.rows[original_index]["source_group"]
        if group != documents.rows[edited_index]["source_group"]:
            raise ValueError("Matched samples disagree on source grouping")
        if group not in selected:
            continue
        original, edit, mask = load_pair(documents, (original_index, edited_index))
        if mode == "whole":
            tensors.extend((whole_tensor(original), whole_tensor(edit)))
            targets.extend((0., 1.))
            trained.add(group)
            continue
        overlaps = [(mask.crop(box).histogram()[255], box) for box in tile_boxes(mask.size, SIDE, SIDE)]
        area, positive = max(overlaps)
        if area < .005 * SIDE ** 2:
            continue
        tensors.extend((rgb_crop(original, positive), rgb_crop(edit, positive)))
        targets.extend((0., 1.))
        negatives = [box for area, box in overlaps if area == 0]
        if negatives:
            gray = original.convert("L")

            def edge_density(box):
                pixels = np.asarray(gray.crop(box), dtype=np.float32)
                return np.abs(np.diff(pixels, axis=1)).mean() + np.abs(np.diff(pixels, axis=0)).mean()

            negative = max(negatives, key=edge_density)
            tensors.append(rgb_crop(edit, negative))
            targets.append(0.)
        trained.add(group)
    return torch.stack(tensors), torch.tensor(targets), len(selected), len(trained)


def train(args, prepared=None):
    torch.manual_seed(args.seed)
    if prepared is None:
        verify_groups(args.manifest)
        data = ManifestDocuments(args.manifest, args.root, "train")
        prepared = training_data(data, args.fraction, args.mode)
    x, y, selected, used = prepared
    trainable_tail = getattr(args, "trainable_tail", None)
    weighted_bce = getattr(args, "weighted_bce", True)
    brightness_gains = getattr(args, "brightness_gains", ())
    if brightness_gains and len(brightness_gains) != args.epochs:
        raise ValueError("Provide one brightness gain per epoch")
    model = Detector(args.mode, args.weights, trainable_tail=trainable_tail).to(memory_format=torch.channels_last)
    weight_decay = getattr(args, "weight_decay", .01)
    step = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": .001},
        {"params": model.features.parameters(), "lr": .00015},
    ], weight_decay=weight_decay)
    positive_weight = (y == 0).sum() / (y == 1).sum() if weighted_bce else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    weights_before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    losses = []
    start = perf_counter()
    model.train()
    for epoch in range(args.epochs):
        inputs = x
        if brightness_gains:
            inputs = x.float().mul(brightness_gains[epoch]).round().clamp(0, 255).to(torch.uint8)
        loss_sum = 0.
        for indices in torch.randperm(len(y)).split(args.batch_size):
            step.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs[indices]), y[indices])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            step.step()
            loss_sum += loss.item() * len(indices)
        losses.append(loss_sum / len(y))
        print(f"{args.mode} seed={args.seed} epoch={epoch + 1} loss={losses[-1]:.5f}", flush=True)
    changed = [name for name, value in model.named_parameters()
               if name in weights_before and not torch.equal(value, weights_before[name])]
    if not changed or (model.frozen_blocks < len(model.features) and not any(name.startswith("features.") for name in changed)):
        raise ValueError("Expected a task-specific parameter update")
    args.output.mkdir(parents=True)
    torch.save(model.state_dict(), args.output / "model.pt")
    metadata = {
        "mode": args.mode, "seed": args.seed, "fraction": args.fraction, "selected_groups": selected,
        "trained_groups": used, "training_examples": len(y), "positive_examples": int(y.sum()),
        "batch_size": args.batch_size, "threads": args.threads, "losses": losses,
        "checkpoint_rule": "final epoch", "adamw_weight_decay": weight_decay, "trainable_tail": trainable_tail,
        "weighted_bce": weighted_bce, "brightness_gains": list(brightness_gains),
        "parameters": sum(value.numel() for value in model.parameters()),
        "updated_parameter_tensors": len(changed), "fit_seconds": perf_counter() - start,
        "training_manifest_sha256": file_sha256(args.manifest), "initial_weights_sha256": file_sha256(args.weights),
        "checkpoint_sha256": file_sha256(args.output / "model.pt"),
        "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
    }
    (args.output / "fit.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def tile_logits(model, tiles, batch_size):
    return torch.cat([model(batch) for batch in tiles.split(batch_size)])


def pooled_logit(scores):
    return scores.topk(min(3, len(scores))).values.mean().item()


def predict(model, image, batch_size, *, geometry=None):
    boxes, tiles = inference_tiles(image, geometry or model.mode)
    scores = tile_logits(model, tiles, batch_size)
    score = pooled_logit(scores)
    x0, y0, x1, y1 = boxes[int(scores.argmax())]
    return score, (x0, y0, min(x1, image.width), min(y1, image.height))


def load_detector(path):
    fit = json.loads((path / "fit.json").read_text(encoding="utf-8"))
    if file_sha256(path / "model.pt") != fit["checkpoint_sha256"]:
        raise ValueError("Checkpoint digest does not match fit metadata")
    if fit["mode"] in {"student", "distilled"}:
        from research.patch_probe import PatchNet

        model = PatchNet()
        model.mode = "patch"
    else:
        model = Detector(fit["mode"], trainable_tail=fit.get("trainable_tail")).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    return model.eval()


def disturb(image, condition, source_group):
    kind, _, severity = condition.partition("-")
    if kind == "clean":
        return image
    if kind == "combined":
        for stage in ("resize-2", "blur-1", "brightness-1", "noise-1", "jpeg-2"):
            image = disturb(image, stage, source_group)
        return image
    strong = severity == "2"
    if kind == "jpeg":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=42 if strong else 78)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB")
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(1.6 if strong else .7))
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(.65 if strong else .85)
    if kind == "resize":
        factor = .5 if strong else .75
        size = tuple(max(1, round(length * factor)) for length in image.size)
        return image.resize(size, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.BILINEAR)
    digest = sha256(source_group.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    pixels = np.asarray(image, dtype=np.float32)
    pixels += rng.normal(0., 12. if strong else 4., pixels.shape)
    return Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))


def score(args):
    model = load_detector(args.checkpoint)
    documents = ManifestDocuments(args.manifest, args.root, args.split)
    groups = sorted({row["source_group"] for row in documents.rows},
                    key=lambda group: sha256(f"receipt-stress-v1:{group}".encode()).digest())
    selected = set(groups[:args.limit_sources])
    destination = args.output
    destination.parent.mkdir(exist_ok=True, parents=True)
    with destination.open("x", newline="", encoding="utf-8") as stream:
        fields = ("sample_id", "source_group", "corpus", "label", "logit", "iou", "hit", "bbox", "milliseconds", "condition")
        writer = csv.writer(stream)
        writer.writerow(fields)
        for row in documents.rows:
            if row["source_group"] not in selected:
                continue
            start = perf_counter()
            with Image.open(args.root / row["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            image = disturb(image, args.condition, row["source_group"])
            logit, box = predict(model, image, args.batch_size, geometry=getattr(args, "geometry", None))
            milliseconds = (perf_counter() - start) * 1000
            label = row["label"] == "manipulated"
            iou, hit = "", ""
            if label:
                with Image.open(args.root / row["mask_path"]) as source:
                    mask = ImageOps.exif_transpose(source).convert("L")
                if mask.size != image.size:
                    raise ValueError("Image/mask raster mismatch")
                intersection = mask.crop(box).histogram()[255]
                area = (box[2] - box[0]) * (box[3] - box[1])
                iou = intersection / (area + mask.histogram()[255] - intersection)
                hit = int(intersection > 0)
            writer.writerow((row["sample_id"], row["source_group"], row["source_dataset"], int(label),
                             logit, iou, hit, " ".join(map(str, box)), milliseconds, args.condition))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="action", required=True)
    fit_parser = subcommands.add_parser("train")
    fit_parser.add_argument("--mode", choices=MODES, required=True)
    fit_parser.add_argument("--seed", type=int, default=42)
    fit_parser.add_argument("--epochs", type=int, default=3)
    fit_parser.add_argument("--fraction", type=float, default=1.)
    fit_parser.add_argument("--weights", type=Path, required=True)
    score_parser = subcommands.add_parser("score")
    score_parser.add_argument("--checkpoint", type=Path, required=True)
    score_parser.add_argument("--split", choices=("calibration", "validation", "test", "shifted_test"), required=True)
    score_parser.add_argument("--limit-sources", type=int)
    score_parser.add_argument("--condition", choices=CONDITIONS, default="clean")
    for command in (fit_parser, score_parser):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=64)
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 2 or args.threads < 1:
        parser.error("batch size must be at least two and threads must be positive")
    if args.action == "train" and (not 0 < args.fraction <= 1 or args.epochs < 1):
        parser.error("fraction must be in (0, 1] and epochs must be positive")
    if args.action == "score" and args.limit_sources is not None and args.limit_sources < 1:
        parser.error("source limit must be positive")
    torch.set_num_threads(args.threads)
    {"train": train, "score": score}[args.action](args)
