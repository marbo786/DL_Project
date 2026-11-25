# Controlled receipt ablations

## Locked protocol

Run `receipt-ablations-v1` uses the contained CORD/SROIE train partition and paired seeds 17, 42, and 71. The local run lock and scoped manifest identify the same 556 fitting groups, 1,755 tiles and 585 positive tiles for every MobileNetV3 training recipe. A role-specific SHA-256 ordering fixes 60 calibration groups with 126 receipts and 80 development groups with 162 receipts before scoring. Both subsets have equal class counts. Results describe these matched subsets. Historical in-domain and shifted test images are outside the scoped manifest.

All MobileNetV3 training rows use the same ImageNet initialization, sampled tile positions and labels, seed-matched shuffle, batch size 64, and three-epoch schedule. AdamW uses head learning rate 0.001, encoder learning rate 0.00015, and weight decay 0.01. The fixed final epoch supplies each checkpoint. Weight decay and the reference's zero augmentation are fixed protocol choices. The zero-augmentation recipe preserves the original pixel signal for comparison. This experiment measures one brightness schedule and does not search for an optimal augmentation or decay setting.

| Comparison with the residual reference | Single factor |
| --- | --- |
| Plain RGB | Disable residual enhancement during training and inference. Reuse the compatible `full` checkpoints and scores from `receipt-study-v1` after checking their digests and provenance. |
| Last three feature blocks | Freeze earlier encoder weights and running statistics. Keep the residual input, optimizer configuration, loss and schedule. |
| Unweighted BCE | Remove the `2:1` positive patch weight. Training examples and batch order stay matched. |
| Symmetric brightness | Apply gains 0.85, 1.15 and 0.95 in training epochs 1, 2 and 3. Within each epoch, apply the identical transform to every original, edited, and off-edit negative tile. Development and calibration images stay clean. |

The whole-image control from the earlier study mixed resize geometry with document supervision and negative sampling. The new **inference-geometry probe** uses the same frozen residual checkpoint for both sides. It compares the native grid and top-three pooling with one 112 × 112 whole-page view through the identical residual model. One view yields one logit, so input geometry and the number of available logits change together by definition. The probe estimates the effect of the inference-view policy for the fixed model.

The **legacy mask-loss correction** compares the same width-eight CNN with map-loss coefficients 0.65 and 0. Both variants use the same images, mask targets, optimizer, seed and eight-epoch schedule. Their checkpoints both come from epoch eight. This avoids the older selection procedure in which IoU contributed to ranking only for runs that used mask loss.

AUROC uses raw development logits. The operating point uses a cutoff derived separately for each run from genuine calibration scores at nominal 5% FPR. Each comparison reports development recall and observed FPR together. The tables report sample standard deviations across seeds and paired changes against the corresponding reference seed. Per-seed AUROC intervals resample source groups within the matched development cohort. Those intervals describe document sampling for that seed.

## Results

| Recipe | AUROC, mean ± SD | Recall, mean ± SD | Observed FPR, mean ± SD |
| --- | ---: | ---: | ---: |
| Residual reference | 0.5134 ± 0.0079 | 6.17% ± 5.38 pp | 5.76% ± 4.67 pp |
| Plain RGB | 0.5139 ± 0.0083 | 5.76% ± 5.57 pp | 4.94% ± 5.66 pp |
| Last three feature blocks | 0.5184 ± 0.0046 | 4.12% ± 2.57 pp | 3.29% ± 1.89 pp |
| Unweighted BCE | 0.5086 ± 0.0096 | 10.70% ± 5.14 pp | 9.47% ± 3.97 pp |
| Symmetric brightness | 0.5108 ± 0.0030 | 5.76% ± 4.99 pp | 4.53% ± 4.67 pp |
| Whole-page inference | 0.4984 ± 0.0005 | 1.23% ± 1.23 pp | 1.23% ± 1.23 pp |
| Legacy mask-loss reference | 0.4973 ± 0.0030 | 4.94% ± 2.47 pp | 4.53% ± 2.57 pp |
| Legacy classification only | 0.5021 ± 0.0037 | 7.00% ± 2.57 pp | 7.00% ± 2.57 pp |

| Change | AUROC delta, seeds 17 / 42 / 71 | Paired AUROC delta, mean ± SD | Paired recall delta, mean ± SD (pp) | Paired FPR delta, mean ± SD (pp) |
| --- | --- | ---: | ---: | ---: |
| Plain RGB | −0.0002 / −0.0011 / +0.0027 | +0.0005 ± 0.0020 | −0.41 ± 2.57 | −0.82 ± 1.43 |
| Last three feature blocks | −0.0085 / +0.0127 / +0.0108 | +0.0050 ± 0.0117 | −2.06 ± 3.77 | −2.47 ± 4.45 |
| Unweighted BCE | −0.0030 / −0.0062 / −0.0052 | −0.0048 ± 0.0016 | +4.53 ± 5.83 | +3.70 ± 4.45 |
| Symmetric brightness | −0.0082 / +0.0015 / −0.0011 | −0.0026 ± 0.0050 | −0.41 ± 1.43 | −1.23 ± 0.00 |
| Whole-page inference | −0.0242 / −0.0075 / −0.0133 | −0.0150 ± 0.0085 | −4.94 ± 4.28 | −4.53 ± 3.56 |
| Legacy classification only | +0.0038 / +0.0053 / +0.0053 | +0.0048 ± 0.0009 | +2.06 ± 4.99 | +2.47 ± 4.94 |

Plain RGB and residual enhancement show little separation on this subset. Freezing all but the last three feature blocks had the highest mean AUROC in this matrix. Its paired effect changed sign across seeds and its mean recall at the calibration-derived cutoff fell. Removing the patch class weight reduced AUROC in all three seeds. The higher recall came with a higher observed FPR. The class weight addresses the 2:1 imbalance between negative and positive training tiles. These results leave the practical trade-off dependent on operating cost. The tested brightness schedule did not improve mean AUROC or recall.

Whole-page inference reduced AUROC in all three seeds for the fixed residual checkpoint. The paired source-bootstrap intervals excluded zero for two of the three seeds. This supports native tiling for this checkpoint on the scoped development set. The corrected legacy mask-loss comparison gave a small mean AUROC increase when removing the map objective. Every per-seed source-bootstrap interval included zero, and mean FPR also rose. Both legacy variants ranked close to chance.

## Reproduce

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m research.receipt_ablations --manifest artifacts/receipt-protocol-v1-contained/manifest.csv --weights artifacts/torch-cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth --root data/aiforge/assets --study artifacts/receipt-study-v1 --output artifacts/receipt-ablations-v1
```

The versioned evidence is under `artifacts/receipt-ablations-v1/`. `lock.json` records the protocol and checkpoint fingerprints. `scoped-manifest.csv` records the cohorts. The per-seed directories keep calibration and development scores, fitted checkpoints for new training runs, and paired metric files. `summary.json` records the across-seed comparisons. `verification.json` records the independent recomputation from all 6,912 calibration and development score rows.

## AdamW decay follow-up

On the same 126 calibration and 162 development documents, removing AdamW weight decay from the residual recipe gave AUROC 0.51374 ± 0.00769 over seeds 17, 42 and 71. Relative to decay 0.01 the paired AUROC difference was +0.00033 ± 0.00042. Recall rose 0.41 percentage points and FPR was unchanged on average. Decay regularizes parameter growth during fine-tuning, but this experiment supplies little evidence that it improves document ranking. The historical checkpoint and policy remain frozen. See the reproduction guide and the per-seed metrics in the receipt-decay-v1 run.
