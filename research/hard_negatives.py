"""Training-only source-matched background sampling for fixed receipt tiles."""

from functools import lru_cache
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from research.diagnose import select_pairs
from research.patch_data import load_pair, rgb_crop, tile_boxes
from research.patch_probe import document_scores, quantiles


def select_background(candidates, random_count, hard_count, seed, scores):
    if hard_count and scores is None:
        raise ValueError("Hard selection requires training predictions")
    if scores is not None and len(scores) != len(candidates):
        raise ValueError("Scores must align with candidates")
    order = list(range(len(candidates)))
    rng = random.Random(seed)
    rng.shuffle(order)
    hard_size = min(hard_count, max(0, len(order) - random_count))
    hardest = sorted(order, key=lambda index: -scores[index])[:hard_size] if hard_size else []
    reserved = set(hardest)
    random_tiles = [index for index in order if index not in reserved][:random_count]
    return [(candidates[index], "random") for index in random_tiles] + [
        (candidates[index], "hard") for index in hardest
    ]


def require_training(documents):
    if not documents.rows or any(row["split"] != "train" for row in documents.rows):
        raise ValueError("Negative mining and sampling require the training partition")


class BackgroundTiles(Dataset):
    def __init__(self, documents, count, seed, side, stride, threshold, random_ratio, hard_ratio=0,
                 model=None, device=None, tile_batch=32):
        require_training(documents)
        if hard_ratio and (model is None or device is None):
            raise ValueError("Mining requires a training model and device")
        self.docs, self.side = documents, side
        self.pairs = select_pairs(documents, count, 42)
        self.load = lru_cache(maxsize=2)(self._load)
        self.samples, self.counts = [], []
        candidate_scores, hard_scores = [], []
        if model is not None:
            model.eval()
        for index, pair in enumerate(self.pairs):
            authentic, altered, mask = self.load(index)
            boxes = tile_boxes(mask.size, side, stride)
            overlap = [mask.crop(box).histogram()[255] for box in boxes]
            ranked = sorted(zip(overlap, boxes), key=lambda item: (-item[0], item[1]))
            positives = [box for area, box in ranked if area >= threshold * side ** 2][:2]
            candidates = [box for area, box in zip(overlap, boxes) if area == 0]
            scores = None
            if hard_ratio and candidates and positives:
                _, a = document_scores(model, authentic, side, stride, "rgb", device, tile_batch)
                _, b = document_scores(model, altered, side, stride, "rgb", device, tile_batch)
                scores = np.maximum(a, b)[np.asarray(overlap) == 0]
                candidate_scores.extend(scores.tolist())
            backgrounds = select_background(candidates, random_ratio * len(positives), hard_ratio * len(positives),
                                            seed + index, scores) if candidates and positives else []
            if scores is not None:
                lookup = dict(zip(candidates, scores))
                hard_scores.extend(float(lookup[box]) for box, kind in backgrounds if kind == "hard")
            self.samples.extend((index, box, backgrounds[step::len(positives)])
                                for step, box in enumerate(positives))
            self.counts.append({"source": documents.rows[pair[0]]["source_document_id"],
                                "positive": len(positives), "matched": len(positives),
                                "random": 2 * sum(kind == "random" for _, kind in backgrounds),
                                "hard": 2 * sum(kind == "hard" for _, kind in backgrounds),
                                "background_candidate_locations": len(candidates)})
            if hard_ratio and (index + 1) % 32 == 0:
                print(f"mined training sources: {index + 1}/{count}", flush=True)
        self.load.cache_clear()
        self.mining = {"candidate_logit_quantiles": quantiles(candidate_scores),
                       "selected_hard_logit_quantiles": quantiles(hard_scores)}

    def _load(self, index):
        return load_pair(self.docs, self.pairs[index])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source, box, backgrounds = self.samples[index]
        authentic, altered, _ = self.load(source)
        crops = [rgb_crop(authentic, box), rgb_crop(altered, box)]
        labels = [0., 1.]
        for background, _ in backgrounds:
            crops.extend([rgb_crop(authentic, background), rgb_crop(altered, background)])
            labels.extend([0., 0.])
        return torch.stack(crops), torch.tensor(labels)
