"""Permutation-invariant attention over receipt tiles or geometric groups."""

import numpy as np
import torch
from torch import nn


def normalized_centers(boxes, size):
    upper = np.minimum(boxes[:, 2:], size)
    centers = (boxes[:, :2] + upper) / 2
    return centers / np.asarray(size)


def spatial_groups(boxes, size, cell=168):
    # Convert normalized centroids back to pixel spacing so the physical
    # neighborhood size is consistent across long and short receipts.
    centers = normalized_centers(boxes, size) * np.asarray(size)
    cells = np.floor(centers / cell).astype(np.int64)
    _, groups = np.unique(cells, axis=0, return_inverse=True)
    return torch.from_numpy(groups)


class BagAttention(nn.Module):
    def __init__(self, dimension, hidden=24):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden), nn.Tanh())
        self.weight = nn.Linear(hidden, 1, bias=False)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, features, mask=None):
        if mask is None:
            mask = torch.ones(features.shape[:-1], dtype=torch.bool, device=features.device)
        if not mask.any(dim=-1).all():
            raise ValueError("Every bag needs at least one valid instance")
        values = self.project(features.masked_fill(~mask.unsqueeze(-1), 0.))
        logits = self.weight(values).squeeze(-1).masked_fill(~mask, -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        pooled = (values * attention.unsqueeze(-1)).sum(dim=-2)
        return self.classifier(pooled).squeeze(-1), attention

    def grouped(self, features, groups):
        count = int(groups.max()) + 1
        sums = features.new_zeros(count, features.shape[-1]).index_add(0, groups, features)
        sizes = torch.bincount(groups, minlength=count)
        pooled = sums / sizes[:, None].clamp_min(1)
        logit, attention = self(pooled, sizes > 0)
        return logit, attention[groups] / sizes[groups]
