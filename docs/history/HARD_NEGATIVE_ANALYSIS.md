# ForgeryLens Phase C hard-negative analysis

## Hypothesis and controls

Phase B trained on edited regions and authentic counterparts at matching coordinates. Phase C tests whether seeing realistic off-mask regions from those same receipts reduces false positives and improves whole-receipt discrimination or localization.

The deduplicated AIForge manifest matched SHA-256 `3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80`. The study reused Phase B's 128 training and 64 validation source pairs, selected with seed 42, and checked that the development source groups were disjoint. The test and shifted-test receipts were excluded. All experiments used the existing 12,177-parameter PatchNet, native RGB, 112 × 112 fixed tiles, stride 84, eight matched positive-location pairs per update, and AdamW at 0.001. The historical six-epoch Phase B checkpoint was re-evaluated for the new metrics. Its max AUROC reproduced exactly at 0.519897.

Up to two tiles per source qualified as positives when at least 0.5% of the tile area overlapped the edit mask. An authentic crop at each positive coordinate supplied a matched negative. Background tiles had zero edit-mask pixels. Both views of each background location received a negative label. This keeps patch labels independent of the parent receipt label and excludes ambiguous tiles from background sampling. Crops use the same RGB decoding and edge padding, with no metadata or paired differences as model inputs.

- **Random:** Sample two distinct off-mask locations per positive location and include the authentic and edited view of each. Random draws are seeded and source-local.
- **Mined mixture:** Fork from the random run after epoch three, including its optimizer state. In one training-only mining pass, rank zero-overlap locations by the higher logit of the two views. Replace half the random locations with the highest-scoring locations and retain the other half as random samples. The per-source quotas bound the influence of any one receipt. Selected hard-location logits had median 0.889 and IQR 0.473 to 1.287, versus median -0.913 across all eligible training locations.

| Setup, per epoch | Positive | Matched negative | Random background | Mined background | Total patches | Updates over six epochs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical | 256 | 256 | 0 | 0 | 512 | 192 |
| Random, every epoch | 256 | 256 | 1,024 | 0 | 1,536 | 192 |
| Mined, epochs 4–6 | 256 | 256 | 512 | 512 | 1,536 | 192 |

All new runs used the fixed epoch-six checkpoint. The mined run's first three epochs shared the random recipe. Every seed retained the same number of positive locations and optimizer updates as Phase B. Background exposure tripled the input patches per update. Class-balanced BCE split aggregate weight evenly between positives and negatives. Background crops consumed most of the negative-class weight, and the bigger input batch also changed BatchNorm statistics relative to Phase B. Random versus mined uses equal patch counts and the same loss. Per-source counts are saved with each run. Identical off-mask views can repeat the same pixels by design.

## Whole-receipt classification

Validation tiled each receipt without masks or references. Training-only receipt scores fit the logistic aggregator and set the genuine-score 90th-percentile cutoff. The normalized alternative averages the highest 5% of tile scores, using at least one tile. Metrics are from the same 128 validation receipts. Bootstrap intervals use 1,000 resamples of complete authentic/edited source pairs.

| Run | Max AUROC (95% CI) | AP | Macro F1 | Recall at zero | FPR at zero | Recall at train cutoff | FPR at train cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-42 | 0.520 [0.501, 0.549] | 0.514 | 0.333 | 100.0% | 100.0% | 10.9% | 10.9% |
| random-17 | 0.510 [0.498, 0.526] | 0.504 | 0.396 | 93.8% | 92.2% | 12.5% | 12.5% |
| random-42 | 0.504 [0.486, 0.522] | 0.505 | 0.375 | 96.9% | 95.3% | 12.5% | 12.5% |
| random-71 | 0.501 [0.485, 0.517] | 0.506 | 0.445 | 75.0% | 78.1% | 4.7% | 4.7% |
| hard-42 | 0.515 [0.497, 0.537] | 0.517 | 0.371 | 95.3% | 95.3% | 14.1% | 10.9% |

For seed 42, the paired-bootstrap 95% interval for the change in max AUROC was [−0.046, +0.010] for random negatives and [−0.037, +0.025] for mined negatives. Max logits were identical within 55, 42, and 48 of 64 pairs, respectively (absolute tolerance 1e−6). This helps explain why better within-document rankings had little impact on classification across receipts.

| Seed-42 setup | Max AUROC | Top-3 | Mean | Top 5% | Trained pooling |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline-42 | 0.520 | 0.518 | 0.506 | 0.511 | 0.510 |
| random-42 | 0.504 | 0.502 | 0.507 | 0.508 | 0.508 |
| hard-42 | 0.515 | 0.521 | 0.505 | 0.515 | 0.516 |

The largest hard-negative AUROC was 0.521, a small difference from the historical max AUROC. At the train-derived cutoff the mined model's max recall was 14.1% at 10.9% FPR, versus 10.9% recall at the same FPR for the baseline. None of the aggregation rules produced credible document-level separation.

### False-positive accumulation

There were 25,958 validation tiles with a median of 144 and a range of 24–1,372 per receipt. Table rates use the fixed zero-logit threshold. The tile fraction is computed only over authentic tiles. Document false-positive rates can remain high even when the tile fraction falls.

| Model | Tiles per receipt | Source pairs | Authentic tiles above threshold | Max FPR | Top-5%-mean FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline-42 | 24–143 | 27 | 42.1% | 100.0% | 100.0% |
| baseline-42 | 144 or more | 37 | 64.0% | 100.0% | 100.0% |
| random-42 | 24–143 | 27 | 16.0% | 92.6% | 85.2% |
| random-42 | 144 or more | 37 | 9.9% | 97.3% | 83.8% |
| hard-42 | 24–143 | 27 | 16.3% | 88.9% | 85.2% |
| hard-42 | 144 or more | 37 | 13.3% | 100.0% | 89.2% |

I inspected the local galleries. High-scoring authentic examples commonly showed printed amounts, subtotal or tax text, and dotted rules. Some edited receipts ranked the same unaltered tiles above the edited field. Random negatives moved some failures toward enlarged or clipped characters, dense text and blue print. Mined negatives still ranked handwriting, grainy or clipped text, printed dates, and separators highly. These are observations from six high-ranked cases per category, with no validation-derived samples used for training.

## Localization and cost

Hit@k is the fraction of edited receipts with at least one mask intersection among the top k tiles. Uniform sampling is without replacement and accounts for each receipt's number of tiles and intersecting tiles. The interval for hit@1 is bootstrapped over source pairs.

| Run | Hit@1 (95% CI) | Hit@3 | Hit@5 | Top-tile mean IoU | CPU median / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline-42 | 12.5% (4.7%–21.9%) | 25.0% | 40.6% | 0.009 | 161 / 835 |
| random-17 | 21.9% (12.5%–31.3%) | 51.6% | 65.6% | 0.039 | 140 / 798 |
| random-42 | 31.3% (20.3%–42.2%) | 56.3% | 68.8% | 0.034 | 136 / 837 |
| random-71 | 23.4% (14.1%–34.4%) | 42.2% | 62.5% | 0.028 | 139 / 807 |
| hard-42 | 25.0% (14.1%–35.9%) | 54.7% | 65.6% | 0.034 | 111 / 610 |
| Uniform | 6.8% | 18.6% | 28.4% | — | — |

Best available tile coverage was 44.8% at the median and 51.8% on average. Every one of the three random seeds improved hit@1, hit@3, and hit@5 over Phase B. The gain varied across seeds and did not carry into whole-receipt AUROC. Mining at seed 42 improved localization over Phase B and yielded lower top-k hit rates than random sampling at the same seed. The mining configuration was screened at one seed because it did not outperform the random control on localization or produce clear classification gains.

Latency includes image decoding, fixed crop extraction, batched inference, and the four fixed pooling statistics. Training and evaluation used the Windows CPU and shared compute during some measurements. The random seed-42 run took 136 seconds including training and scoring. The hard-negative continuation took 151 seconds including mining and scoring, plus the shared three-epoch warmup. Weights, score arrays, per-source sampling counts and crop galleries stay under `artifacts/phase-c/` and are excluded from Git.

## Decision and deployment gate

Random off-mask negatives are useful for localizing candidate regions. Whole-receipt discrimination remains the limiting factor. Follow Outcome B with a single controlled multiple-instance learning experiment on fixed tiles from the existing encoder. Compare training-only document supervision with spatially clustered pooling against the current fixed rules. This targets the observed gap between a tile overlapping an edit and that tile carrying enough discriminative weight across receipts. A wider or pretrained backbone is deferred pending that comparison. If document-level supervision still yields chance-level separation, stop tuning PatchNet and compare a frozen pretrained MobileNetV3 feature extractor with a matched random-initialization control.

Before deploying real detection, the system needs repeatable receipt-level validation on the full split, a calibrated operating point with acceptable recall and false positives, and a frozen evaluation on the held-out corpora. Any cross-generator claim requires genuinely different editing generators.

## Reproduction

Use `python -m research.phase_c` with `--manifest data/aiforge/manifest_deduplicated.csv --asset-root data/aiforge/assets` and a fresh `--output` under `artifacts/phase-c`. The modes are `baseline --baseline artifacts/phase-b/fixed-tight-42/checkpoint.pt`, `random --seed 42`, and `hard --initial artifacts/phase-c/random-42/epoch3.pt`. Repeat random with `--seed 17` and `--seed 71`. The historical, random seed-42, and mined runs used `--gallery`. All 28 tests passed with the workstation's socket-free ASGI test runner.

AI assistance implemented, ran and analyzed the engineering experiment. Results are measured from AIForge, with no changes to the public inference path.
