"""Development-only matched-tile ranking pilot and whole-image baseline."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from research import train as legacy
from research.receipt import ablations as audit
from research.receipt import model as receipt_model
from research.data import ManifestDocuments
from research.manifest import file_sha256
from research.patch_probe import PatchNet
from research.receipt.metrics import evaluate
from research.receipt.protocol import verify_groups


def fit_student(folder, triplets, seed, pair_weight, manifest):
    torch.manual_seed(seed)
    model = PatchNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    labels = torch.tensor([0., 1., 0.])
    losses = []
    for _ in range(3):
        running = 0.
        for batch in torch.randperm(len(triplets)).split(21):
            scores = model(triplets[batch].flatten(0, 1)).reshape(-1, 3)
            loss = F.binary_cross_entropy_with_logits(scores, labels.expand_as(scores), pos_weight=labels.new_tensor(2.))
            if pair_weight:
                margin = scores[:, 1:2] - scores[:, [0, 2]]
                loss = loss + pair_weight * F.softplus(-margin).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
        losses.append(running / len(triplets))
    folder.mkdir()
    torch.save(model.state_dict(), folder / "model.pt")
    audit.write_json(folder / "fit.json", {"mode": "student", "seed": seed, "pair_weight": pair_weight,
                     "losses": losses, "training_manifest_sha256": file_sha256(manifest),
                     "checkpoint_sha256": file_sha256(folder / "model.pt")})


def run(args):
    manifest, root, output = args.manifest, args.root, args.output
    verify_groups(manifest)
    rows = audit.read_rows(manifest)
    roles = ("train", "calibration", "validation")
    if {r["split"] for r in rows} != set(roles):
        raise ValueError("Supply a scoped development-only manifest")
    output.mkdir(parents=True)
    audit.write_json(output / "lock.json", {"manifest": file_sha256(manifest), "seed": 42, "roles": roles,
                     "baseline": "legacy whole-image CNN, eight epochs, final epoch",
                     "student": "PatchNet, three epochs, 21 triplets per step, AdamW 0.001",
                     "paired_loss_weights": [0, 1], "selection": "exploratory development only"})
    torch.set_num_threads(2)
    docs = {role: ManifestDocuments(manifest, root, role) for role in roles}
    cached = {role: legacy.cache(data) for role, data in docs.items()}
    baseline, curve = legacy.train("baseline", {"width": 16, "local": False, "residual": False}, cached["train"],
                                   cached["validation"], 8, 42, checkpoint_rule="final epoch")
    folder = output / "baseline"
    folder.mkdir()
    torch.save(baseline.state_dict(), folder / "model.pt")
    audit.write_json(folder / "fit.json", {"seed": 42, "curve": curve, "training_manifest_sha256": file_sha256(manifest),
                     "checkpoint_sha256": file_sha256(folder / "model.pt")})
    for role in roles[1:]:
        logits = legacy.collect(baseline, cached[role])[0]
        audit.write_rows(folder / f"{role}.csv", [{"sample_id": r["sample_id"], "source_group": r["source_group"],
                        "label": int(r["label"] == "manipulated"), "logit": float(logit), "hit": "", "iou": "", "milliseconds": ""}
                        for r, logit in zip(docs[role].rows, logits)])
    evaluate(folder / "calibration.csv", folder / "validation.csv", folder / "metrics.json",
             compare=args.reference / "validation.csv")
    del cached
    x, y, _, _ = receipt_model.training_data(docs["train"], 1., "full")
    if len(y) % 3 or not torch.equal(y.reshape(-1, 3), torch.tensor([0., 1., 0.]).expand(len(y)//3, 3)):
        raise ValueError("Each training spec must have a matched original, edit and off-edit tile")
    for name, weight in (("control", 0.), ("ranking", 1.)):
        folder = output / name
        fit_student(folder, x.reshape(-1, 3, *x.shape[1:]), 42, weight, manifest)
        for role in roles[1:]:
            receipt_model.score(SimpleNamespace(checkpoint=folder, manifest=manifest, root=root, split=role,
                                batch_size=64, condition="clean", limit_sources=None, output=folder / f"{role}.csv"))
        evaluate(folder / "calibration.csv", folder / "validation.csv", folder / "metrics.json",
                 compare=output / "control/validation.csv" if weight else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("root", "manifest", "reference", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    run(parser.parse_args())

