"""Compare the same PatchNet student with and without soft teacher targets."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from research.data import ManifestDocuments
from research.manifest import file_sha256
from research.patch_probe import PatchNet
from research.receipt.metrics import evaluate
from research.receipt.model import load_detector, score, training_data
from research.receipt.protocol import verify_groups


def run(args):
    torch.set_num_threads(args.threads)
    verify_groups(args.manifest)
    teacher_meta = json.loads((args.teacher / "fit.json").read_text(encoding="utf-8"))
    manifest_digest = file_sha256(args.manifest)
    if teacher_meta["training_manifest_sha256"] != manifest_digest or teacher_meta["mode"] == "whole":
        raise ValueError("Teacher must be a patch model fit on this protocol")
    documents = ManifestDocuments(args.manifest, args.root, "train")
    x, y, sources, _ = training_data(documents, 1., "full")
    teacher = load_detector(args.teacher)
    temperature = 2.
    with torch.inference_mode():
        teacher_scores = torch.cat([teacher(batch) for batch in x.split(args.batch_size)])
    soft = torch.sigmoid(teacher_scores / temperature)
    args.output.mkdir(parents=True)
    for mode in ("student", "distilled"):
        torch.manual_seed(args.seed)
        model = PatchNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        weight = (y == 0).sum() / (y == 1).sum()
        losses = []
        for _ in range(args.epochs):
            total = 0.
            for selection in torch.randperm(len(y)).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[selection])
                loss = F.binary_cross_entropy_with_logits(logits, y[selection], pos_weight=weight)
                if mode == "distilled":
                    soft_loss = F.binary_cross_entropy_with_logits(logits / temperature, soft[selection])
                    loss = .5 * loss + .5 * temperature ** 2 * soft_loss
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(selection)
            losses.append(total / len(y))
        folder = args.output / mode
        folder.mkdir()
        torch.save(model.state_dict(), folder / "model.pt")
        fit = {"mode": mode, "seed": args.seed, "losses": losses, "sources": sources,
               "parameters": sum(value.numel() for value in model.parameters()),
               "checkpoint_sha256": file_sha256(folder / "model.pt"), "training_manifest_sha256": manifest_digest,
               "teacher_sha256": teacher_meta["checkpoint_sha256"] if mode == "distilled" else None,
               "temperature": temperature if mode == "distilled" else None}
        (folder / "fit.json").write_text(json.dumps(fit, indent=2) + "\n", encoding="utf-8")
        for split in ("calibration", "validation"):
            score(SimpleNamespace(manifest=args.manifest, root=args.root, checkpoint=folder,
                                  batch_size=args.batch_size, split=split, output=folder / f"{split}.csv",
                                  condition="clean", limit_sources=None))
        compare = args.output / "student" / "validation.csv" if mode == "distilled" else None
        evaluate(folder / "calibration.csv", folder / "validation.csv", folder / "metrics.json", compare=compare)
        print(f"Measured {mode}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "root", "teacher", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--epochs", type=int, default=3)
    cli.add_argument("--batch-size", type=int, default=64)
    run(cli.parse_args())
