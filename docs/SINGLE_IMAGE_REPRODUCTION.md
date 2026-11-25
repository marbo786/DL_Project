# Reproduce the single-image receipt study

## Reproduction commands

Run from the repository root with a compatible Python environment active.

| Task | With Make | Bash or PowerShell |
| --- | --- | --- |
| Install constrained Python dependencies | `make setup` | `python scripts/setup.py` |
| Run tests | `make test` | `python -m pytest` |
| Check Python lint and syntax | `make lint` | `python -m ruff check api research scripts tests web/frontend/scripts` |
| Start the saved-sample studio | `make demo` | `npm --prefix web/frontend run demo` |
| Print the inputs and access instructions | `make reproduce` | `python scripts/reproduce.py` |

Tests marked `artifacts` or `data` skip with the missing paths when their inputs are absent. Run `python -m pytest -q -rs` to see those reasons, or use `python -m pytest -m "not artifacts and not data"` to select tests that require neither. GitHub Actions runs lint, CPU pytest, and the Docker/browser release checks on pushes and pull requests targeting `main`.

Obtain the full paired dataset and pretrained checkpoint through the sources printed by the reproduction command. Supply the original paired ZIP, its extracted package directory, the TorchVision checkpoint, and a fresh output directory. The launcher runs the existing containment audit, training matrix, historical scoring, and XFUND evaluation. It downloads no data. Add `--plan` to inspect the stages before committing compute.

Bash

```bash
bash scripts/reproduce.sh --archive /path/to/paired.zip --package /path/to/AIForge-Doc-v2.0 --weights /path/to/mobilenet_v3_small-047dcff4.pth --output runs/receipt-study --plan
```

PowerShell

```powershell
./scripts/reproduce.ps1 --archive C:/path/to/paired.zip --package C:/path/to/AIForge-Doc-v2.0 --weights C:/path/to/mobilenet_v3_small-047dcff4.pth --output runs/receipt-study --plan
```

Remove `--plan` to execute. The full matrix is a research run with substantial CPU time and disk use. Saved measurements and their availability are indexed in [results](../results/README.md).

The [experiment ledger](SINGLE_IMAGE_RESULTS.md) records measured results and dependencies. Keep restricted inputs, source crosswalks, and private case reviews in the ignored `data/` and `artifacts/` trees. The [AIForge v2 dataset card](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2) describes the noncommercial share-alike terms and underlying source-dataset terms. Use the official v2.0.1 package and its paired authentic assets. The [AIForge v1 dataset card](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v1) covers v1 access and terms separately.

## Local CPU environment

The measured host uses Windows 11 x64, AMD Ryzen 7 5800U, Python 3.10.10, PyTorch 2.12.0+cpu, torchvision 0.27.0+cpu, NumPy 2.2.6, Pillow 12.0.0, scikit-learn 1.7.2, and Matplotlib 3.10.8. The script produces portable state dictionaries and local prediction tables. Use the ImageNet MobileNetV3-Small V1 checkpoint from the torchvision model distribution. Its SHA-256 is `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.

The measured Windows CPU environment is pinned in [requirements-receipt.txt](../research/requirements-receipt.txt) for Python 3.10 on x86-64. The macOS command selects the corresponding PyPI wheels without the CPU wheel suffix. Package availability depends on platform and architecture. Run from the repository root. Pass new output directories for each study.

Bash

```bash
python3.10 -m venv .venv/receipt
source .venv/receipt/bin/activate
if [ "$(uname -s)" = Darwin ]; then
  python -m pip install -r <(sed -e '/^--extra-index-url /d' -e 's/+cpu$//' research/requirements-receipt.txt)
else
  python -m pip install -r research/requirements-receipt.txt
fi
python -m pip check
export TORCH_HOME="$PWD/artifacts/torch-cache"
python -c "from torchvision.models import MobileNet_V3_Small_Weights as W; W.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)"

# With the official paired archive already obtained
python -m research.aiforge PATH_TO_AIFORGE_V2_ARCHIVE data/aiforge
python -m research.manifest data/aiforge/manifest.csv data/aiforge/assets --deduplicated-output data/aiforge/manifest_deduplicated.csv
python -m research.receipt.protocol data/aiforge/manifest_deduplicated.csv data/aiforge/assets artifacts/receipt-protocol-v1 --package AIForge-Doc-v2.0
python -m research.receipt.candidate_review artifacts/receipt-protocol-v1 data/aiforge/assets artifacts/receipt-protocol-v1-contained

manifestPath='artifacts/receipt-protocol-v1-contained/manifest.csv'
assetsPath='data/aiforge/assets'
weightsPath='artifacts/torch-cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
studyPath='artifacts/receipt-study-v1'
python -m research.receipt.study --manifest $manifestPath --root $assetsPath --weights $weightsPath --output $studyPath --modes full frozen partial residual whole --curve
python -m research.receipt.study --manifest $manifestPath --root $assetsPath --weights $weightsPath --output artifacts/receipt-stress-v1/full-42 --stress-checkpoint "$studyPath/full-42" --stress-sources 24 --threads 3
python -m research.receipt.report --manifest $manifestPath --root $assetsPath --run "$studyPath/full-42" --output artifacts/receipt-figures-v1
python -m research.receipt.retrospective --manifest $manifestPath --root $assetsPath --study $studyPath --output artifacts/receipt-retrospective-v1 --threads 3
python -m research.receipt.forms --package AIForge-Doc-v2.0 --manifest $manifestPath --receipt-root $assetsPath --study $studyPath --lock artifacts/receipt-retrospective-v1/freeze.json --output artifacts/receipt-xfund-v1 --threads 3

python -m research.receipt.efficiency --manifest $manifestPath --root $assetsPath --checkpoint "$studyPath/full-42" --output "$studyPath/efficiency-full-42.json" --threads 2
python -m research.receipt.memory --manifest $manifestPath --root $assetsPath --checkpoint "$studyPath/full-42" --output artifacts/receipt-runtime-v1 --threads 2
python -m research.receipt.distill --manifest $manifestPath --root $assetsPath --teacher "$studyPath/full-42" --output artifacts/receipt-distill-v1 --threads 2
python -m research.receipt.share $studyPath artifacts/receipt-share-v1 --stress artifacts/receipt-stress-v1/full-42 --frozen-evaluation artifacts/receipt-retrospective-v1 --frozen-evaluation artifacts/receipt-xfund-v1
python -m pytest -q tests/test_receipt_research.py
```

PowerShell

```powershell
py -3.10 -m venv .venv/receipt
. .venv/receipt/Scripts/Activate.ps1
python -m pip install -r research/requirements-receipt.txt
python -m pip check
$env:TORCH_HOME = (Join-Path (Get-Location).Path 'artifacts/torch-cache')
python -c "from torchvision.models import MobileNet_V3_Small_Weights as W; W.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)"

# With the official paired archive already obtained
python -m research.aiforge PATH_TO_AIFORGE_V2_ARCHIVE data/aiforge
python -m research.manifest data/aiforge/manifest.csv data/aiforge/assets --deduplicated-output data/aiforge/manifest_deduplicated.csv
python -m research.receipt.protocol data/aiforge/manifest_deduplicated.csv data/aiforge/assets artifacts/receipt-protocol-v1 --package AIForge-Doc-v2.0
python -m research.receipt.candidate_review artifacts/receipt-protocol-v1 data/aiforge/assets artifacts/receipt-protocol-v1-contained

$manifestPath = 'artifacts/receipt-protocol-v1-contained/manifest.csv'
$assetsPath = 'data/aiforge/assets'
$weightsPath = 'artifacts/torch-cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
$studyPath = 'artifacts/receipt-study-v1'
python -m research.receipt.study --manifest $manifestPath --root $assetsPath --weights $weightsPath --output $studyPath --modes full frozen partial residual whole --curve
python -m research.receipt.study --manifest $manifestPath --root $assetsPath --weights $weightsPath --output artifacts/receipt-stress-v1/full-42 --stress-checkpoint "$studyPath/full-42" --stress-sources 24 --threads 3
python -m research.receipt.report --manifest $manifestPath --root $assetsPath --run "$studyPath/full-42" --output artifacts/receipt-figures-v1
python -m research.receipt.retrospective --manifest $manifestPath --root $assetsPath --study $studyPath --output artifacts/receipt-retrospective-v1 --threads 3
python -m research.receipt.forms --package AIForge-Doc-v2.0 --manifest $manifestPath --receipt-root $assetsPath --study $studyPath --lock artifacts/receipt-retrospective-v1/freeze.json --output artifacts/receipt-xfund-v1 --threads 3

python -m research.receipt.efficiency --manifest $manifestPath --root $assetsPath --checkpoint "$studyPath/full-42" --output "$studyPath/efficiency-full-42.json" --threads 2
python -m research.receipt.memory --manifest $manifestPath --root $assetsPath --checkpoint "$studyPath/full-42" --output artifacts/receipt-runtime-v1 --threads 2
python -m research.receipt.distill --manifest $manifestPath --root $assetsPath --teacher "$studyPath/full-42" --output artifacts/receipt-distill-v1 --threads 2
python -m research.receipt.share $studyPath artifacts/receipt-share-v1 --stress artifacts/receipt-stress-v1/full-42 --frozen-evaluation artifacts/receipt-retrospective-v1 --frozen-evaluation artifacts/receipt-xfund-v1
python -m pytest -q tests/test_receipt_research.py
```

Use `torch.hub.get_dir()` to locate the downloaded weights or set `TORCH_HOME` to `artifacts/torch-cache` before fetching. The manifest is versioned by its digest. A study run records the input manifest and weights hashes, environment, training size, configuration, parameter changes, losses, and checkpoint hash in each `fit.json`. Per-document raw scores report native tile coordinates, group identifiers, localization overlap, and timings. `metrics.json` records calibration and score hashes with confidence intervals. The fitted calibration pool is always separate from the scored development groups.

## Pretrained checkpoint provenance

| Field | Citation |
| --- | --- |
| Publisher and weight revision | PyTorch TorchVision, `MobileNet_V3_Small_Weights.IMAGENET1K_V1`. [Model definition](https://github.com/pytorch/vision/blob/78839c2b06c83c6cfb5c4da692ffb331bbd4c4cc/torchvision/models/mobilenetv3.py) |
| Implementation revision | TorchVision `v0.27.0`, commit `78839c2b06c83c6cfb5c4da692ffb331bbd4c4cc`. [Release](https://github.com/pytorch/vision/releases/tag/v0.27.0) |
| Official download | [`mobilenet_v3_small-047dcff4.pth`](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth) |
| Verified full SHA-256 | `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`. Recorded as `initial_weights_sha256` in `results/receipt-study-v1/residual-42/fit.json`. |
| Software license | BSD 3-Clause for the TorchVision software. [License](https://github.com/pytorch/vision/blob/78839c2b06c83c6cfb5c4da692ffb331bbd4c4cc/LICENSE) |
| Weight terms | An explicit license for this exact checkpoint could not be verified. TorchVision's documentation calls out possible separate or dataset-derived terms for pretrained models. ImageNet publishes terms of access for its database. [TorchVision pretrained-weights notice](https://docs.pytorch.org/vision/main/models.html#general-information-on-pre-trained-weights) · [ImageNet terms of access](https://www.image-net.org/download.php) |

The project keeps the downloaded weights and receipt checkpoints local for course research and cites the official download. Public redistribution and deployment of these checkpoints are out of scope while the weight terms are unresolved.

## CPU environment smoke check

Run `receipt-env-smoke-v1` verified the pinned runtime in an isolated Python 3.10.10 virtual environment on Windows. DNS resolution blocked access to both package indexes, so the check used local copies of the installed packages. All pinned runtime versions matched and `pip check` passed. A fresh install from the package indexes remains unverified.

The smoke fit used residual MobileNetV3-Small, seed 42, one epoch, 3% of training groups, batch size 16, and two CPU threads. The fit used 16 groups and 48 tiles, recorded finite loss, and updated 140 parameter tensors. Scoring used four groups each from calibration and development, with eight finite scores per role. The smoke manifest contained only train, calibration, and development roles. Use the numbers to verify that the entry points execute. Model quality is evaluated in the full study.

The [environment record](../results/receipt-env-smoke-v1/environment.json), [fit metadata](../results/receipt-env-smoke-v1/fit/fit.json), and [metrics](../results/receipt-env-smoke-v1/metrics.json) record the smoke check. The environment and fit records include the requirements, manifest, and checkpoint hashes. The manifest and score CSVs remain in the original run directory.

## External baseline

TruFor is the selected external baseline. Use official source commit `ae54475df6f41a491d7615100feb19263dec13f7`. Its CPU inference flag is `-gpu -1`. The upstream Dockerfile uses PyTorch 1.11 and a CUDA 11.3 base image even though the inference entry point supports CPU. The official zip is `TruFor_weights.zip`, published MD5 `7bee48f3476c75616c3c5721ab256ff8`. Verify the archive before extraction and record the extracted checkpoint SHA-256 plus its environment. Upstream grants informational and nonprofit use. [Official inference instructions](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/test_docker/README.md)

```sh
python -m research.receipt.trufor stage --manifest $manifestPath --root $assetsPath --split validation --smoke --output artifacts/trufor-smoke-inputs
# With the pinned official CPU-capable container built, mount that input directory
# as /data, mount a new output directory as /data_out, and pass
# -gpu -1 -in /data/ -out /data_out/
python -m research.receipt.trufor collect --manifest $manifestPath --root $assetsPath --split validation --smoke --predictions artifacts/trufor-smoke-predictions --weights PATH_TO_TRUFOR_CHECKPOINT --output artifacts/trufor-smoke.csv
```

For a full comparison, stage and collect each cohort without `--smoke`. Use the contained receipt manifest for `calibration`, `validation`, `test` and `shifted_test`. Use `artifacts/receipt-xfund-v1/manifest.csv` with the official v2 package root and `--split test` for the held-back forms. Pass the receipt calibration CSV and each scored cohort CSV to `research.receipt.metrics`. The adapter validates score ranges and EXIF-oriented raster geometry, retains the official document score and native 0.5-cutoff pixel IoU, and derives a matching highest tile from the official anomaly map. Include measured processing time only when separately recorded under comparable settings.

For the native-resolution CPU panel, select and stage the same 40 genuine/edit pairs with:

```sh
python -m research.receipt.trufor_panel --manifest $manifestPath --root $assetsPath --output artifacts/trufor-native-panel-v1
```

Each role has 10 source groups. Calibration, validation and the historical in-domain test use 6 CORD and 4 SROIE groups under a 600,000-pixel ceiling. The shifted historical test uses 10 WildReceipt groups under a 250,000-pixel ceiling. Selection uses a fixed hash rank within each stratum and preserves native resolution. Collect all four roles against the panel manifest. For paired comparisons, subset the frozen residual seed-42 score tables to the same sample IDs before passing them to `research.receipt.metrics --compare`.

The local Windows run used the pinned upstream Python inference sources, PyTorch 2.12.0+cpu, torchvision 0.27.0+cpu, YACS 0.1.8 and the required layer helpers from timm 0.6.13. Its bootstrap runner sets four CPU threads, points the upstream config at the verified checkpoint, and uses the restricted PyTorch loader with an allowlist for the NumPy scalar and dtype metadata in the official archive. The extracted checkpoint SHA-256 is `ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844`.

DocTamper's official DTD inference implementation uses additional model assets and its dataset requires nonprofit access credentials. OSTF/DAF uses a legacy OpenMMLab stack, detection-box outputs, and checkpoints distributed through ModelScope or Baidu. The inspected revisions are `2d53389932f72941575adbb595377955d366881c1f` for DocTamper and `ec955ccd85445f4fc231b0b36cf7844a17a46f` for OSTF. [DocTamper source](https://github.com/qcf-568/DocTamper) · [OSTF source](https://github.com/qcf-568/OSTF)

## Independent data

The [independent audit](SINGLE_IMAGE_INDEPENDENT_AUDIT.md) describes the continuation findings. Run these commands from the repository root with the original private run directories present. Use a fresh output directory for a new audit.

Bash

```bash
mkdir -p artifacts/receipt-verification-repeat
python -m research.receipt.independent_metrics artifacts/receipt-verification-repeat/recomputed.json
python -m research.receipt.integrity_audit artifacts/receipt-verification-repeat/training.json training
python -m research.receipt.integrity_audit artifacts/receipt-verification-repeat/export.json export
python -m research.receipt.trufor_full artifacts/receipt-verification-repeat --roles panel calibration validation test shifted_test xfund
```

PowerShell

```powershell
New-Item -ItemType Directory artifacts/receipt-verification-repeat
python -m research.receipt.independent_metrics artifacts/receipt-verification-repeat/recomputed.json
python -m research.receipt.integrity_audit artifacts/receipt-verification-repeat/training.json training
python -m research.receipt.integrity_audit artifacts/receipt-verification-repeat/export.json export
python -m research.receipt.trufor_full artifacts/receipt-verification-repeat --roles panel calibration validation test shifted_test xfund
```

The independent metrics command joins each score table to its designated manifest and uses 1,000 source-group bootstrap draws. The TruFor runner uses the pinned source tree under `artifacts/trufor-upstream/`, the official `weights/trufor.pth.tar`, and native-size RGB inference. The panel is checked against the official score and localization map. Larger attention matrices use PyTorch scaled dot-product attention to reduce temporary memory and can introduce floating-point differences. The runner saves per-image scores and localization summaries and can resume completed rows from the same output directory. It emits matched metrics after a complete calibration role and a complete evaluation role are available.

Generate the v2 side of the planned cross-generator evaluation with the audited receipt assignments and the official v2 source metadata:

```sh
python -m research.receipt.v2_inventory --dataset AIForge-Doc-v2.0 --assignments $manifestPath --output artifacts/receipt-generator-inventory-v1
```

The resulting `v2.csv` lists candidates for the join. The local matched run used the authorized `data/AIForge-Doc-v1` release. It required a unique shared edit spec, consistent corpus and generator metadata, a v1 authentic filename matching the canonical source ID or its corpus-stripped form, and pixel-identical decoded originals. It also required aligned edited images and nonempty binary masks. The v1 generator column in the v2 inventory served as a consistency check against the actual v1 metadata.

The generator join takes two normalized spec manifests with `spec_id,source_document_id,source_dataset,generator,image_path,authentic_path,mask_path` columns. Supply a reviewed crosswalk with `v1_source_id,v2_source_id,evidence` and the contained v2 assignments. The join verifies aligned local images and a decoded-identical authentic counterpart for each accepted spec. Use `--normalized-output` to produce two ready-to-score manifests. Run `research.receipt.model score` against each recorded asset root using the same checkpoint, then `research.receipt.metrics --compare` for a source-group paired comparison within each historical split. Use the historical calibration CSV and record the locked cutoff. Historical assignments produce a retrospective matched comparison. Fresh confirmatory sources require an independent cohort with documented aliases, transforms, and label-access controls. Run `python -m research.receipt.generator_pairs --help`.

The local run files are under `artifacts/receipt-generator-evaluation-v1/`. To reproduce into a fresh sibling directory, copy `run.py` there and execute it from the repository root. The runner validates the frozen checkpoint, calibration, model-selection matrix, and source assignments before scoring.

For actual acquisition data, use the existing [reference protocol and audit CLI](history/REAL_WORLD_VERIFICATION_ANALYSIS.md). Preserve unchanged recaptures, documented edits, legitimate revisions, and wrong references as separate outcome groups. Compare devices and sessions, then report coverage, rejection and wrong-reference acceptance independently of change scores.

## Reproduce the course-paper follow-up

Run the following in the pinned receipt environment after preparing the contained corpus and the original ablation artifacts. Each training command expects a new output directory. The policy replay reads saved predictions. The training and pooling follow-ups use fitting, calibration, or development data only.

Bash

```bash
assets='data/aiforge/assets'
manifest='artifacts/receipt-protocol-v1-contained/manifest.csv'
previous='artifacts/receipt-ablations-v1'
scoped="$previous/scoped-manifest.csv"
weights='artifacts/torch-cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
python -m research.receipt.policy_audit
python -m research.receipt.decay --root $assets --previous $previous --weights $weights --output artifacts/receipt-decay-repeat
python -m research.receipt.matched_pilot --root $assets --manifest $scoped --reference "$previous/residual-42" --output artifacts/receipt-matched-repeat
python -m research.receipt.pooling_audit --manifest $manifest --assets $assets --output artifacts/receipt-pooling-repeat
python -m research.receipt.repeat_timing --manifest $manifest --assets $assets --artifacts artifacts --output artifacts/receipt-timing-repeat
```

PowerShell

```powershell
$assets = 'data/aiforge/assets'
$manifest = 'artifacts/receipt-protocol-v1-contained/manifest.csv'
$previous = 'artifacts/receipt-ablations-v1'
$scoped = "$previous/scoped-manifest.csv"
$weights = 'artifacts/torch-cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
python -m research.receipt.policy_audit
python -m research.receipt.decay --root $assets --previous $previous --weights $weights --output artifacts/receipt-decay-repeat
python -m research.receipt.matched_pilot --root $assets --manifest $scoped --reference "$previous/residual-42" --output artifacts/receipt-matched-repeat
python -m research.receipt.pooling_audit --manifest $manifest --assets $assets --output artifacts/receipt-pooling-repeat
python -m research.receipt.repeat_timing --manifest $manifest --assets $assets --artifacts artifacts --output artifacts/receipt-timing-repeat
```

The whole-image pilot uses a randomly initialized width-16 CNN with document labels, no residual input, eight fixed epochs, and seed 42. Its common development evaluation and paired source-bootstrap difference use the same IDs and groups as the frozen residual model. The matched PatchNet pilot uses two freshly initialized students with identical seed, batches, optimizer, and inference. One adds a pairwise softplus loss on the annotated edit relative to the genuine and off-edit tiles. Neither experiment opens a shifted image.

The course paper's editable source is [ForgeryLens_Technical_Report.tex](ForgeryLens_Technical_Report.tex). With IEEEtran, graphicx, booktabs, and url installed, compile from the docs directory using `latexmk -pdf ForgeryLens_Technical_Report.tex`. Its figures use relative paths under `figures/`. The separate [historical synthetic report](archive/ForgeryLens_Historical_Synthetic_Report.pdf) is reproduced by the earlier release command in the README.

