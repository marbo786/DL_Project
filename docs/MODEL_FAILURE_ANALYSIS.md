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

```powershell
$manifest = 'data/aiforge/manifest_deduplicated.csv'
$assets = 'data/aiforge/assets'
py -m research.diagnose inspect --manifest $manifest --asset-root $assets --pairs 24 --previews 8 --output artifacts/phase-a/pairs
py -m research.diagnose calibration --manifest $manifest --asset-root $assets --checkpoint artifacts/aiforge/model.pt --output artifacts/phase-a/scores
py -m research.diagnose overfit --manifest $manifest --asset-root $assets --pairs 24 --epochs 80 --seed 42 --output artifacts/phase-a/memorization
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

The controlled follow-up was started with the same training subset, seed, and epoch budget. It included classification-only, localization-only, the original combined loss, a lighter mask loss, doubled input dimensions, and a higher learning rate. The Windows execution environment became unavailable before the logs could be inspected. The status of these runs and the epoch-by-epoch baseline trace remains unverified. The script for the baseline writes its trace to `artifacts/phase-a/memorization/epochs.jsonl`.

## Controlled repeat

Run each row with the original overfit command, keeping 24 pairs, 80 epochs, seed 42, and the same thread count. Give each run a fresh output directory. The flags below change one factor relative to the combined run.

| Control | Additional flags |
| --- | --- |
| Original combined objective | None |
| Classification only | `--map-weight 0` |
| Localization only | `--class-weight 0` |
| Lighter localization loss | `--map-weight 0.15` |
| Higher input resolution | `--resolution 2` |
| Higher learning rate | `--learning-rate 0.01` |

Use epoch-level accuracy and AUROC for classification, and mask IoU for localization. Each trace records both loss components even when only one drives updates. Scores from the untrained classification objective in the localization-only run are descriptive. The higher-resolution run measures IoU on a finer grid, so its raw IoU is tied to that grid. Comparing localization across resolutions requires projecting both predictions to a common evaluation grid.

AI assistance executed the baseline run and documented the measured results. It is an engineering diagnostic for the project record.
