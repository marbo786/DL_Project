"""Compact whole-image, local evidence, and student models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(outputs),
        nn.SiLU(),
    )


class ReceiptNet(nn.Module):
    def __init__(self, width=16, local=True, residual=True, context=True, topk=True):
        super().__init__()
        self.local = local
        self.residual = residual
        self.context = context
        self.topk = topk
        self.first = block(2 if residual else 1, width)
        self.second = block(width, width * 2)
        self.last = block(width * 2, width * 3)
        self.map = nn.Conv2d(width * 2, 1, 1)
        self.global_head = nn.Linear(width * 3, 1)

    def forward(self, image):
        if self.residual:
            smooth = F.avg_pool2d(image, 3, stride=1, padding=1, count_include_pad=False)
            image = torch.cat((image, (image - smooth) * 3), dim=1)
        spatial = self.second(self.first(image))
        logits = self.map(spatial)
        global_logits = self.global_head(self.last(spatial).mean(dim=(2, 3))).flatten()
        if not self.local:
            return global_logits, logits
        flattened = logits.flatten(1)
        if self.topk:
            local_logits = flattened.topk(12, dim=1).values.mean(dim=1)
        else:
            local_logits = flattened.mean(dim=1)
        if self.context:
            local_logits = local_logits + global_logits
        return local_logits, logits
