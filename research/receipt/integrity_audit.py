"""Verify receipt checkpoints and exercise the training and export paths."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.data import ManifestDocuments
from research.manifest import file_sha256
from research.receipt.efficiency import load_image, quantize, ranked_paths
from research.receipt.model import Detector, inference_tiles, load_detector, predict, training_data

ART = Path("artifacts")
ROOT = Path("data/aiforge/assets")
MANIFEST = ART / "receipt-protocol-v1-contained/manifest.csv"
WEIGHTS = ART / "torch-cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"
STUDY = ART / "receipt-study-v1"


def loss_for(labels):
    return nn.BCEWithLogitsLoss(pos_weight=(labels == 0).sum() / (labels == 1).sum())


def optimizer_for(model):
    # Match the production training configuration independently.
    return torch.optim.AdamW([{"params": model.head.parameters(), "lr": .001},
                              {"params": [p for p in model.features.parameters() if p.requires_grad],
                               "lr": .00015}])


def gradients(documents):
    inputs, labels, *_ = training_data(documents, .015, "full")
    result = {}
    for mode in ("frozen", "partial", "full"):
        torch.manual_seed(42)
        model = Detector(mode, WEIGHTS).to(memory_format=torch.channels_last).train()
        optimizer = optimizer_for(model)
        allowed = {name for name, p in model.named_parameters() if p.requires_grad}
        assigned = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert assigned == {id(p) for p in model.parameters() if p.requires_grad}
        before = {key: value.clone() for key, value in model.state_dict().items()}
        loss_for(labels)(model(inputs), labels).backward()
        nonzero = {name for name, p in model.named_parameters() if p.grad is not None and torch.count_nonzero(p.grad)}
        assert nonzero <= allowed and all(name in nonzero for name in ("head.weight", "head.bias"))
        optimizer.step()
        changed = {key for key, value in model.state_dict().items() if not torch.equal(before[key], value)}
        assert nonzero <= changed
        assert all(key in allowed for key in changed if key in before and key in dict(model.named_parameters()))
        bn_changed = [key for key in changed if key.endswith(("running_mean", "running_var"))]
        assert mode != "frozen" or not bn_changed
        assert all(int(key.split(".")[1]) >= model.frozen_blocks for key in bn_changed)
        result[mode] = {"optimizer_tensors": len(assigned), "nonzero_gradient_tensors": len(nonzero),
                        "changed_encoder_parameters": sum(key.startswith("features.") for key in changed & allowed),
                        "changed_bn_buffers": len(bn_changed), "frozen_encoder_blocks": model.frozen_blocks}
    return result


def checkpoint_audit(prepared):
    result = {}
    for directory in sorted(STUDY.glob("*")):
        if not directory.is_dir():
            continue
        meta = json.loads((directory / "fit.json").read_text())
        weight_digest = file_sha256(WEIGHTS)
        assert meta["initial_weights_sha256"] == weight_digest
        assert meta["training_manifest_sha256"] == file_sha256(MANIFEST)
        initial = Detector(meta["mode"], WEIGHTS)
        state = load_detector(directory).state_dict()
        changed = []
        for name, parameter in initial.features.named_parameters():
            if not torch.equal(parameter, state[f"features.{name}"]):
                assert parameter.requires_grad, name
                changed.append(name)
        assert len(changed) + 2 == meta["updated_parameter_tensors"]
        if meta["fraction"] == 1.:
            inputs, labels, selected, trained = prepared["whole" if meta["mode"] == "whole" else "patch"]
            assert (len(inputs), int(labels.sum()), selected, trained) == (
                meta["training_examples"], meta["positive_examples"], meta["selected_groups"], meta["trained_groups"])
        result[directory.name] = {"changed_encoder_tensors": len(changed), "epochs": len(meta["losses"]),
                                  "trained_groups": meta["trained_groups"], "training_examples": meta["training_examples"],
                                  "loss_first": meta["losses"][0], "loss_last": meta["losses"][-1]}
    return result


def overfit(documents):
    inputs, labels, selected, trained = training_data(documents, .015, "full")
    results = {}
    for name in ("actual_masks", "obvious_edit", "shuffled_labels"):
        torch.manual_seed(42)
        x = inputs.clone()
        y = labels.clone()
        if name == "obvious_edit":
            x[y == 1, 0, 8:48, 8:48] = 255
            x[y == 1, 1:, 8:48, 8:48] = 0
        if name == "shuffled_labels":
            y = y[torch.randperm(len(y))]
        model = Detector("full", WEIGHTS).to(memory_format=torch.channels_last)
        optim = optimizer_for(model)
        criterion = loss_for(y)
        losses = []
        model.train()
        for _ in range(40):
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            score = model(x)
            accuracy = float(((score > 0) == y.bool()).float().mean())
            balanced = float((((score[y == 0] <= 0).float().mean()) +
                              ((score[y == 1] > 0).float().mean())) / 2)
        results[name] = {"groups_selected": selected, "groups_used": trained, "tiles": len(y),
                         "loss_start": losses[0], "loss_end": losses[-1],
                         "eval_mode_training_accuracy": accuracy, "eval_mode_balanced_accuracy": balanced}
    return results


def runtime_audit():
    directory = ART / "receipt-runtime-v1"
    completed = json.loads((directory / "runtime.json").read_text())
    checkpoint = STUDY / "full-42"
    assert file_sha256(checkpoint / "model.pt") == completed["checkpoint_sha256"]
    fp32 = load_detector(checkpoint)
    calibration = ManifestDocuments(MANIFEST, ROOT, "calibration")
    observers = ranked_paths(calibration, completed["observer_documents"])
    eager_int8, _ = quantize(fp32, (
        inference_tiles(load_image(ROOT / row["image_path"]), fp32.mode)[1][::4]
        for row in observers), completed["engine"])
    validation = ManifestDocuments(MANIFEST, ROOT, "validation")
    inputs = [load_image(ROOT / row["image_path"]) for row in ranked_paths(validation, 8)]
    metrics = json.loads((checkpoint / "metrics.json").read_text())["calibration"]
    cutoff = json.loads((checkpoint / "metrics.json").read_text())["operating_points"]["0.05"]["logit_threshold"]
    def probability(logit):
        return 1 / (1 + np.exp(-metrics["slope"] * logit - metrics["intercept"]))
    results = {}
    for kind, eager in (("fp32", fp32), ("int8", eager_int8)):
        model_path = directory / f"{kind}.torchscript"
        assert file_sha256(model_path) == completed["results"][kind]["script_sha256"]
        scripted = torch.jit.load(str(model_path), map_location="cpu").eval()
        class Export:
            mode = "full"
            def __call__(self, images):
                return scripted(images)
        example_results = []
        for image in inputs:
            tiles = inference_tiles(image, "full")[1][:64]
            with torch.inference_mode():
                tile_delta = float((eager(tiles) - scripted(tiles)).abs().max())
            a, box_a = predict(eager, image, 64)
            b, box_b = predict(Export(), image, 64)
            example_results.append({"tile_logit_delta": tile_delta, "logit_delta": abs(a - b),
                                    "probability_delta": abs(probability(a) - probability(b)),
                                    "same_decision": bool((a > cutoff) == (b > cutoff)), "same_box": box_a == box_b})
        results[kind] = {"documents": len(inputs), "max_tile_logit_delta": max(r["tile_logit_delta"] for r in example_results),
                         "max_document_logit_delta": max(r["logit_delta"] for r in example_results),
                         "max_probability_delta": max(r["probability_delta"] for r in example_results),
                         "decision_matches": sum(r["same_decision"] for r in example_results),
                         "localization_matches": sum(r["same_box"] for r in example_results)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("area", choices=("training", "export"))
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.area == "training":
        data = ManifestDocuments(MANIFEST, ROOT, "train")
        evidence = {"gradients": gradients(data), "overfit": overfit(data)}
        print(json.dumps(evidence), flush=True)
        prepared = {"patch": training_data(data, 1., "full"), "whole": training_data(data, 1., "whole")}
        evidence["checkpoints"] = checkpoint_audit(prepared)
    else:
        evidence = runtime_audit()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
