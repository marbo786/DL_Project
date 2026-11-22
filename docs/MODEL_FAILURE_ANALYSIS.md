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

## Open questions for the student team

The real-pair memorization run remains pending. Its per-epoch JSONL logs the training loss, both loss components, gradient norms, AUROC, score quantiles by class, and localization metrics. It reuses the same tiny training subset for fitting and scoring, so the scores measure memorization only. If the trace fails to show clear learning, inspect the classification and localization objectives separately before comparing architectures. If it memorizes the subset, use training and validation to isolate the effect of input resolution before changing the backbone. The pair audit includes file format, mode, and byte size for a shortcut check. A predictive shortcut, loss interference, and insufficient capacity remain unconfirmed.
