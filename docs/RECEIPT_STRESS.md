# Frozen receipt detector stress test

## Protocol

Run `receipt-residual-stress-v1` uses the frozen residual MobileNetV3-Small checkpoint, seed 42, with the saved Platt mapping and `receipt-preview-v1` routing rule. The raw priority cutoff is strictly above `1.4312477111806406` and the demo confidence gate is 0.55. A fixed hash ordering selects 24 source groups from the CORD/SROIE development split. The same genuine and edited images are scored clean and under every condition. `lock.json` records the paired IDs, input fingerprints, policy, and perturbation grid before scoring. The historical and final shifted sets are outside this study.

| Stress family | Level 1 | Level 2 |
| --- | ---: | ---: |
| Gaussian blur | 0.7 px radius | 1.6 px radius |
| Low light | 0.85 RGB brightness gain | 0.65 RGB brightness gain |
| JPEG | Quality 78 | Quality 42 |
| Random occlusion | Neutral gray rectangle, 15% of width and height | Neutral gray rectangle, 30% of width and height |
| Edit-targeted occlusion | Same size as level 1 random occlusion | Same size as level 2 random occlusion |

Random placement is deterministic per pair and severity and independent of the edit mask. Edit-targeted placement centers the rectangle on a foreground mask pixel and clamps it inside the image. Each pair receives the same rectangle on both its genuine and edited version. Masks choose the targeted perturbation and measure overlap. The detector receives the perturbed image alone.

AUROC measures raw score ranking. Recall and FPR are at the frozen logit cutoff before abstention. Genuine and edited coverage each measure the fraction receiving a route other than **Inconclusive** under the full demo rule. Each change is paired against clean on the same source groups. The saved intervals use 500 paired source-group bootstrap draws. They describe variation within this development sample and fixed checkpoint.

## Open-set protocol

The open-set probe passes locally available images through the demo's upload decoder, checkpoint, calibration, and routing code. **Prioritize review** and **Normal queue** both count as inappropriate acceptance for an unsupported input. The three screenshot inputs capture the desktop, mobile, and tablet interfaces of the same local Reference Studio. The photograph inputs are `china.jpg` by danielbuechele and `flower.jpg` by vultilion, as credited under CC BY 2.0 in the local scikit-learn image bundle's `README.txt`. The eight unsupported forms are deterministic blank fixtures for unrelated campus workflows. Screen variants share one interface and the forms share a generator, so this is an exploratory test of the listed inputs.

## Paired development results

The locked sample contained 24 genuine and 24 edited receipts. Parentheses show paired changes from the same clean receipts. Recall and FPR changes are in percentage points.

| Condition | AUROC (change) | Edited recall (change) | Genuine FPR (change) | Coverage, genuine / edited |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.5156 | 8.33% | 8.33% | 0% / 0% |
| Blur 0.7 px | 0.5156 (0.0000) | 20.83% (+12.50) | 20.83% (+12.50) | 0% / 0% |
| Blur 1.6 px | 0.5000 (−0.0156) | 25.00% (+16.67) | 25.00% (+16.67) | 0% / 0% |
| Low light 0.85 | 0.5174 (+0.0017) | 8.33% (0.00) | 8.33% (0.00) | 0% / 0% |
| Low light 0.65 | 0.5000 (−0.0156) | 0% (−8.33) | 0% (−8.33) | 0% / 0% |
| JPEG 78 | 0.5043 (−0.0113) | 12.50% (+4.17) | 8.33% (0.00) | 0% / 0% |
| JPEG 42 | 0.4905 (−0.0252) | 8.33% (0.00) | 8.33% (0.00) | 0% / 0% |
| Random occlusion, level 1 | 0.5156 (0.0000) | 8.33% (0.00) | 8.33% (0.00) | 0% / 0% |
| Random occlusion, level 2 | 0.5052 (−0.0104) | 4.17% (−4.17) | 4.17% (−4.17) | 0% / 0% |
| Edit-targeted occlusion, level 1 | 0.4991 (−0.0165) | 8.33% (0.00) | 8.33% (0.00) | 0% / 0% |
| Edit-targeted occlusion, level 2 | 0.4991 (−0.0165) | 4.17% (−4.17) | 4.17% (−4.17) | 0% / 0% |

Strong blur increased both true and false flags from 2 to 6 at the frozen logit cutoff. Strong low light brought both counts to zero. JPEG 42 had the largest point drop in ranking, with a paired AUROC change of −0.0252 and a 95% interval of [−0.1025, +0.0178]. All ten perturbed AUROC intervals included zero. The sample supports describing cutoff sensitivity, with uncertainty around the ranking changes.

Random occlusion intersected 7 and 11 of the 24 edit masks at levels 1 and 2, covering a mean of 15.7% and 26.0% of edit pixels. Targeted occlusion intersected all 24 masks at both levels, covering 67.8% and 80.5% of edit pixels on average. Class-specific routing coverage stayed at zero in every condition, so each paired coverage change was 0 percentage points.

## Open-set results

| Input type | Inconclusive | Inappropriately routed | Inappropriate acceptance |
| --- | ---: | ---: | ---: |
| Screenshots | 3 | 0 | 0/3 (0%) |
| Photos | 2 | 0 | 0/2 (0%) |
| Unsupported forms | 8 | 0 | 0/8 (0%) |
| Total | 13 | 0 | 0/13 (0%) |

All 13 uploads reached scoring and fell below the confidence gate. The same gate deferred all 48 clean receipt inputs, so zero inappropriate routes on this set provides little evidence of open-set discrimination. The probe leaves broader rejection and workflow utility unverified.

## Reproduce

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m research.receipt_stress --manifest artifacts/receipt-protocol-v1-contained/manifest.csv --root data/aiforge/assets --output artifacts/receipt-residual-stress-v1
```

The run requires the existing receipt checkpoint and calibration files, development assets, and `artifacts/phase-h/desktop.png`, `mobile.png`, and `tablet.png`. Use a new output directory when rerunning. Evidence is under `artifacts/receipt-residual-stress-v1/` in `lock.json`, `stress-scores.csv`, `open-set-scores.csv`, `summary.json`, and `verification.json`. Verification independently recomputed the reported point estimates, checked all 528 stress predictions and 13 open-set routes, and matched the clean logits exactly to the saved development scores.
