"""Fictional paired specimens and manifest-backed image datasets."""

import csv
from functools import lru_cache
import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from torch.utils.data import Dataset

from api.signal import SIZE, pixels

SYNTHETIC_SPLITS = {"train": (1000, 144), "validation": (10000, 40),
                    "test": (24000, 60), "shifted_test": (35000, 60)}


def features(image, size=SIZE):
    return torch.from_numpy(pixels(image, size)[None])


def example_tensors(image, mask, label, size=SIZE):
    target = np.asarray(mask.resize(size, Image.Resampling.NEAREST), dtype=np.float32) / 255.0
    return features(image, size), torch.from_numpy(target[None]), float(label)


@lru_cache(maxsize=16)
def font(size):
    location = Path(__file__).with_name("fonts") / "DejaVuSansMono.ttf"
    return ImageFont.truetype(location, size) if location.is_file() else ImageFont.load_default(size=size)


def specimen(source, variant, shift=False):
    """Generate one fictional document, paired with a mask and binary label."""
    rng = random.Random(source)
    width, height = 336, 480
    paper = rng.randint(216, 247)
    ink = rng.randint(22, 70)
    texture = np.random.default_rng(source).normal(0, 2.6, (height, width))
    image = Image.fromarray(np.clip(paper + texture, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    margin = rng.randint(22, 45)
    title = rng.choice(("NORTH PIER", "MAPLE STALL", "PAPER TRAIL", "FIELD NOTES"))
    draw.text((margin, 24), title, fill=ink, font=font(31))
    draw.text((margin, 66), "FICTIONAL RESEARCH SPECIMEN", fill=ink, font=font(15))
    draw.line((margin, 100, width - margin, 100), fill=ink, width=1)
    positions = []
    for row in range(7):
        y = 125 + row * 39 + rng.randint(-5, 5)
        draw.text((margin, y), rng.choice(("Field book", "Envelope", "Paper clips", "Index cards")), fill=ink, font=font(17))
        x = width - margin - 66
        draw.text((x, y), f"{rng.uniform(1.0, 25.0):5.2f}", fill=ink, font=font(17))
        positions.append((x, y))
    draw.line((margin, 411, width - margin, 411), fill=ink, width=1)
    draw.text((margin, 429), "DEMO / NO TRANSACTION", fill=ink, font=font(15))
    mask = Image.new("L", image.size)
    edit = random.Random(source + 9_978_371)
    x, y = edit.choice(positions)
    box = (x - 7, y - 4, x + 61, y + 23)
    if variant:
        left, top, right, bottom = box
        if shift:
            patch = image.crop(box).filter(ImageFilter.GaussianBlur(edit.uniform(1.3, 2.5)))
            patch = ImageEnhance.Contrast(patch).enhance(0.74)
        else:
            delta = edit.choice((-1, 1)) * edit.randint(15, 36)
            background = np.clip(paper + delta + texture[top:bottom, left:right] * .35, 0, 255)
            patch = Image.fromarray(background.astype(np.uint8))
            ImageDraw.Draw(patch).text((7, 4), f"{edit.uniform(1, 25):5.2f}", fill=ink, font=font(17))
        image.paste(patch, box)
        ImageDraw.Draw(mask).rectangle((left, top, right - 1, bottom - 1), fill=255)
    return image.convert("RGB"), mask, int(variant)


def perturb(image, kind):
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(2.4))
    if kind == "low_light":
        return ImageEnhance.Brightness(image).enhance(0.58)
    if kind == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=23)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    if kind == "occlusion":
        image = image.copy()
        w, h = image.size
        ImageDraw.Draw(image).rectangle((int(w * .63), int(h * .30), int(w * .86), int(h * .48)), fill="#9e9e9e")
        return image
    return image


class Specimens(Dataset):
    def __init__(self, start, count, shift=False, stress=None):
        self.start, self.count, self.shift, self.stress = start, count, shift, stress

    def __len__(self):
        return self.count * 2

    def __getitem__(self, index):
        source = self.start + index // 2
        image, mask, label = specimen(source, index % 2, self.shift)
        return example_tensors(perturb(image, self.stress), mask, label)


class ManifestDocuments(Dataset):
    def __init__(self, manifest, root, split, stress=None, size=SIZE):
        with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
            self.rows = [row for row in csv.DictReader(stream) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"manifest contains no {split} rows")
        self.root, self.stress, self.size = Path(root), stress, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, mask, label = self.load(index)
        return example_tensors(perturb(image, self.stress), mask, label, self.size)

    def load(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image_path"]) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        label = row["label"] == "manipulated"
        if row["mask_path"]:
            with Image.open(self.root / row["mask_path"]) as target:
                mask = ImageOps.exif_transpose(target).convert("L")
            if mask.size != image.size:
                raise ValueError("source and mask have different oriented dimensions")
        elif label:
            raise ValueError("pixel masks are required for manipulated training and evaluation samples")
        else:
            mask = Image.new("L", image.size)
        return image, mask, label
