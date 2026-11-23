"""Matched native receipt crops and deterministic, image-only tiles."""

from collections import defaultdict
from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from research.diagnose import select_pairs


def check_source_splits(*datasets):
    seen = set()
    for dataset in datasets:
        sources = {row["source_document_id"] for row in dataset.rows}
        if seen & sources:
            raise ValueError("A source appears in more than one partition")
        seen.update(sources)


def axis_starts(length, side, stride):
    end = max(0, length - side)
    starts = list(range(0, end + 1, stride))
    if starts[-1] != end:
        starts.append(end)
    return starts


def tile_boxes(size, side=224, stride=112):
    width, height = size
    return [(x, y, x + side, y + side)
            for y in axis_starts(height, side, stride)
            for x in axis_starts(width, side, stride)]


def oracle_boxes(mask, side):
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("Edited image has an empty mask")
    left, top, right, bottom = bounds
    width, height = mask.size

    def region_starts(low, high, length):
        extent = high - low
        if extent <= side:
            return [min(max((low + high - side) // 2, 0), max(length - side, 0))]
        return [low + start for start in axis_starts(extent, side, side)]

    return [(x, y, x + side, y + side)
            for y in region_starts(top, bottom, height) for x in region_starts(left, right, width)
            if mask.crop((x, y, x + side, y + side)).getbbox()]


def rgb_crop(image, box, transform="rgb"):
    area = image.crop((box[0], box[1], min(box[2], image.width), min(box[3], image.height)))
    pixels = np.asarray(area, dtype=np.uint8)
    height, width = box[3] - box[1], box[2] - box[0]
    if area.size != (width, height):
        pixels = np.pad(pixels, ((0, height - area.height), (0, width - area.width), (0, 0)), mode="edge")
    if transform == "gray":
        gray = np.asarray(Image.fromarray(pixels).convert("L"))
        pixels = np.repeat(gray[:, :, None], 3, axis=2)
    elif transform == "jpeg":
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, "JPEG", quality=85, subsampling=2)
        stream.seek(0)
        with Image.open(stream) as encoded:
            pixels = np.asarray(encoded.convert("RGB"))
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


def load_pair(dataset, pair):
    original, _, genuine = dataset.load(pair[0])
    edited, mask, manipulated = dataset.load(pair[1])
    if genuine or not manipulated or original.size != edited.size or mask.size != edited.size:
        raise ValueError("Invalid oriented receipt pair")
    colors = mask.getcolors(2)
    if colors is None or {value for _, value in colors} - {0, 255} or not mask.getbbox():
        raise ValueError("Expected a nonempty binary edit mask")
    return original, edited, mask


def supervised_tiles(mask, side, stride, threshold, limit=2):
    ranked = []
    for box in tile_boxes(mask.size, side, stride):
        overlap = mask.crop(box).histogram()[255]
        if overlap >= threshold * side * side:
            ranked.append((overlap, box))
    return [box for _, box in sorted(ranked, key=lambda row: (-row[0], row[1]))[:limit]]


class MatchedCrops(Dataset):
    """Lazy RGB crop decoding. At most two decoded pairs are retained by the loader."""

    def __init__(self, documents, count, seed, side, stride=112, threshold=None, transform="rgb"):
        self.documents = documents
        self.pairs = select_pairs(documents, count, seed)
        self.side, self.transform = side, transform
        self.entries, self.audit = [], defaultdict(list)
        self.load = lru_cache(maxsize=2)(self._load)
        for index, pair in enumerate(self.pairs):
            original, edited, mask = self.load(index)
            boxes = (oracle_boxes(mask, side) if threshold is None
                     else supervised_tiles(mask, side, stride, threshold))
            bounds = mask.getbbox()
            self.audit["edit_width"].append(bounds[2] - bounds[0])
            self.audit["edit_height"].append(bounds[3] - bounds[1])
            self.audit["boxes_by_source"].append(len(boxes))
            for box in boxes:
                self.entries.append((index, box))
                region = np.asarray(mask.crop(box)) > 0
                before, after = (np.asarray(image.crop(box)) for image in (original, edited))
                changed = np.any(before != after, axis=2)
                self.audit["mask_fraction"].append(float(region.mean()))
                self.audit["changed_inside"].append(int((changed & region).sum()))
                self.audit["changed_outside"].append(int((changed & ~region).sum()))
        self.load.cache_clear()

    def _load(self, index):
        return load_pair(self.documents, self.pairs[index])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        source, box = self.entries[index]
        original, edited, _ = self.load(source)
        return torch.stack([rgb_crop(image, box, self.transform) for image in (original, edited)])
