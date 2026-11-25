# ForgeryLens model failure diagnostics

This engineering record covers the existing AIForge-Doc v2.0.1 checkpoint. Pair inspection used training data. The score audit used validation data and the existing checkpoint with no fitting. The CORD/SROIE and WildReceipt test sets were excluded.

## Pair and mask checks

- All 24 sampled training pairs had matching oriented dimensions, valid labels and nonempty binary edit masks. The sample was seeded with 42 and balanced across CORD and SROIE. In every sampled pair, authentic and edited pixels differed inside the native mask and matched exactly outside it.
- The masks retained foreground on all 24 samples at 112 × 160. Median foreground was 1.04% at native resolution, 1.07% after nearest-neighbor resizing, and 1.88% on the max-pooled 28 × 40 training grid. The median share of resized grayscale difference magnitude inside the resized mask was 96.2%. All 24 pairs also had some change outside the resized mask. The image and annotation use different resampling kernels, which can spread an image difference beyond the nearest-neighbor mask at its edge.
- The validation mask grid marked 1.35% of cells positive on edited examples. At the existing zero-logit cutoff, predicted positive fractions were 1.22% for authentic examples and 1.23% for edited examples. The median within-image grid-logit standard deviation was 0.912. These figures describe spatial activity and predicted area. Validation IoU of 0.017 shows that the positive cells seldom align with the annotations.
- The edited PNG was larger in bytes in 22 of 24 sampled pairs. File size is a candidate shortcut for tools that consume file metadata. The existing network receives decoded grayscale tensors, so this finding does not show that it used file size.

## Validation score audit

| Measure on 456 validation images | Raw | Calibrated |
| --- | ---: | ---: |
| AUROC | 0.50535 | 0.50535 |
| ECE | 0.02667 | 0.00516 |
| Mean edit probability | 0.49596 | 0.50000 |
| Authentic median edit probability | 0.49674 | 0.50011 |
| Edited median edit probability | 0.49784 | 0.50025 |

The fitted temperature reached the grid maximum of 8.0. The affine transform compressed probabilities toward the 50% class prevalence while preserving score ranking in this run. A near-constant prevalence prediction can have low ECE even when it barely separates the classes. Calibration therefore offers no evidence of improved discrimination here.

## Run the diagnostics

Use the project Python environment with `research/requirements.txt` installed. Use an unused output directory for each invocation. Preview contact sheets contain benchmark images and are optional local outputs.

Bash

```bash
manifest='data/aiforge/manifest_deduplicated.csv'
assets='data/aiforge/assets'
python -m research.diagnose inspect --manifest $manifest --asset-root $assets --pairs 24 --previews 8 --output artifacts/phase-a/pairs
python -m research.diagnose calibration --manifest $manifest --asset-root $assets --checkpoint artifacts/aiforge/model.pt --output artifacts/phase-a/scores
python -m research.diagnose overfit --manifest $manifest --asset-root $assets --pairs 24 --epochs 80 --seed 42 --output artifacts/phase-a/memorization
```

PowerShell

```powershell
$manifest = 'data/aiforge/manifest_deduplicated.csv'
$assets = 'data/aiforge/assets'
python -m research.diagnose inspect --manifest $manifest --asset-root $assets --pairs 24 --previews 8 --output artifacts/phase-a/pairs
python -m research.diagnose calibration --manifest $manifest --asset-root $assets --checkpoint artifacts/aiforge/model.pt --output artifacts/phase-a/scores
python -m research.diagnose overfit --manifest $manifest --asset-root $assets --pairs 24 --epochs 80 --seed 42 --output artifacts/phase-a/memorization
```

## Real-receipt memorization

The existing 4,890-parameter local model ran for 80 epochs on 24 authentic/edited training pairs (48 images, seed 42, 112 × 160 inputs). The same 48 images were scored to measure memorization. The held-out sets were outside the diagnostic.

| Epoch 80 metric | Measured value |
| --- | ---: |
| Combined training loss | 0.97955 |
| Training accuracy | 54.17% |
| Training ROC-AUC | 0.56771 |
| Training mask IoU | 0.06961 |

The original recipe failed to memorize this subset. Poor validation results therefore coexist with weak fitting on the sampled training images. The final metrics alone cannot identify loss interference, resolution, optimizer settings, or capacity as the cause.

## Controlled comparisons

The Windows workspace recovered complete logs for the original combined, classification-only, and localization-only controls. The lighter-mask log stopped at epoch 62 and was rerun from the original seed and schedule. The higher-resolution and higher-learning-rate runs were missing and were completed. Every row below reports epoch 80 on the same 24 training pairs (48 images), seed 42. The last four rows are follow-up probes. All 2,617 canonical source groups were checked for partition overlap. The test and shifted-test receipt files were excluded from both fitting and evaluation in this phase.

| Configuration | Accuracy | AUROC | Classification BCE | Weighted mask BCE | Mask IoU | Gradient L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined, original | 54.17% | 0.56771 | 0.68516 | 0.49245 | 0.06961 | 0.72968 |
| Classification only | 52.08% | 0.58333 | 0.68749 | 0.81956 | 0.02993 | 0.29875 |
| Localization only | 50.00% | 0.50868 | 0.83501 | 0.42642 | 0.10699 | 0.25259 |
| Mask loss weight 0.15 | 52.08% | 0.56250 | 0.68649 | 0.61596 | 0.03568 | 0.37109 |
| 224 × 320 inputs | 56.25% | 0.56250 | 0.68600 | 0.42076 | 0.02961 | 0.60774 |
| Learning rate 0.01 | 56.25% | 0.60938 | 0.68485 | 0.43348 | 0.08226 | 0.57215 |
| Width 24 | 66.67% | 0.76215 | 0.64413 | 0.41539 | 0.10340 | 1.02900 |
| Width 32 | 56.25% | 0.63889 | 0.67277 | 0.42320 | 0.10221 | 0.79304 |
| Batch size 8 | 60.42% | 0.65625 | 0.68696 | 0.38973 | 0.07855 | 1.49299 |
| Width 24 + batch size 8 | 75.00% | 0.82813 | 0.61359 | 0.34565 | 0.12943 | 2.05432 |

BCE and gradient L2 are means over the training batches from the final epoch. Mask BCE includes the original positive-cell weight of 18, before applying the objective's mask coefficient. Both losses are measured even when only one drives updates. Classification scores in the localization-only run and mask scores in the classification-only run are descriptive. The recovered localization-only run used a zero-weight classification loss in its graph, which left the otherwise-unused classification branch subject to AdamW weight decay. The corrected diagnostic detaches that loss and selects localization checkpoints by IoU. The 224 × 320 run reports IoU on its native 56 × 80 grid. The diagnostic now also logs IoU projected onto the original 28 × 40 grid for future high-resolution runs, which passed a two-epoch real-pair smoke check.

### Interpretation

- **Capacity and optimization both contributed to underfitting on these 24 pairs.** Width 24 improved AUROC from 0.56771 to 0.76215 with the same input and objective. Width 32 reached 0.63889, so the gain did not increase monotonically with width. Raising the learning rate reached 0.60938. Reducing the batch size reached 0.65625 and increased updates from 160 to 480 at the same epoch budget, while also changing batch statistics. Combining the width and batch changes reached 75.00% accuracy, 0.82813 AUROC, and 0.12943 IoU. The experiment does not isolate a single causal mechanism for the gain.
- **Loss coupling explains some localization degradation.** Mask-only optimization raised IoU from 0.06961 to 0.10699. Dropping the mask loss raised classification AUROC only to 0.58333, while lowering its weight reached 0.56250. The original combined objective's first-batch shared-gradient cosine was negative in 39 of 80 epochs, with median 0.027. The mean final gradient norm was 0.72968. These results show mixed alignment with active gradients and give weak evidence that loss interference is the primary classification bottleneck.
- **A larger whole-image resize did not improve discrimination.** Doubling both dimensions gave 0.56250 AUROC. This test leaves open whether native-resolution crops, color, or different evidence aggregation would help.

## Validation of the width and batch changes

The width screen used 128 training source pairs and all 228 validation pairs (456 images) for 12 epochs, with the original joint loss and seed 42. It compared the original width with width 24, first at batch size 32 and then at batch size 8. Each row reports the validation-selected checkpoint. Training AUROC is reported on the same checkpoint.

| Batch size | Width | Parameters | Train AUROC | Validation AUROC | Validation accuracy | Validation mask IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 4,890 | 0.50330 | 0.50315 | 50.66% | 0.01882 |
| 32 | 24 | 42,314 | 0.51013 | 0.49954 | 49.78% | 0.01884 |
| 8 | 8 | 4,890 | 0.50665 | 0.50079 | 49.78% | 0.02286 |
| 8 | 24 | 42,314 | 0.50214 | 0.50102 | 50.00% | 0.01806 |

The stronger in-sample recipe did not transfer in this short validation screen. Keep the settings available for reproducible research and keep the historical checkpoint as the comparison. The next diagnostic is a train-and-validation patch signal check on native or near-native RGB. Use the edit masks to align authentic and edited training crops, then apply a fixed image-only tiling policy for document-level validation. Compare matched-patch discrimination and pooled document scores before investing in a stronger pretrained baseline.

## Reproduction and verification

The base memorization command is above. For each control, use a fresh output directory and add one of `--map-weight 0`, `--class-weight 0`, `--map-weight 0.15`, `--resolution 2`, or `--learning-rate 0.01`. The follow-up flags were `--width 24`, `--width 32`, `--batch-size 8`, and `--width 24 --batch-size 8`.

```sh
python -m research.diagnose width-screen --manifest data/aiforge/manifest_deduplicated.csv --asset-root data/aiforge/assets --pairs 128 --epochs 12 --seed 42 --batch-size 32 --output artifacts/phase-a/width-screen-v1
python -m research.diagnose width-screen --manifest data/aiforge/manifest_deduplicated.csv --asset-root data/aiforge/assets --pairs 128 --epochs 12 --seed 42 --batch-size 8 --output artifacts/phase-a/width-screen-batch8
```

Manifest SHA-256 `3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80`. Ordered 24-source sample SHA-256 `e4df74dee29c9557e8d1d4f05473b626c8bdd60ded6f28a47bb99773e17a6494`. Complete raw traces and experiment settings are in their respective local directories under `artifacts/phase-a`. The runs used PyTorch 2.12.0+cpu, scikit-learn 1.7.2, and PyMuPDF 1.27.2.3. Some installed API dependencies differ from the repository pins. The 19 project tests passed with the Windows socket-free test runner. AI assistance recovered the logs, executed the missing runs, implemented the research controls, and generated this engineering record from measurements.

The [Phase B patch evidence experiment](PATCH_EVIDENCE_ANALYSIS.md) tested this follow-up. Tight oracle crops showed modest discrimination. Fixed image-only tiling remained near chance for whole receipts.
