# ForgeryLens native-resolution patch evidence

## Hypothesis and protocol

A localized receipt edit may preserve pixel evidence in a native RGB crop that is obscured by whole-page resizing. Phase B tests first whether the edited region can be recognized with its position supplied, then whether a classifier can find useful evidence when the full receipt is tiled using image dimensions alone.

The AIForge-Doc v2.0.1 deduplicated manifest matched SHA-256 `3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80`. Source IDs were disjoint across the development partitions. Training used the same 128 source pairs for every seed, selected with seed 42. Oracle validation used all 228 validation source pairs. Fixed-tiling validation used an identical, deterministically selected 64-pair screen for both models. The 262 CORD/SROIE test pairs and 1,336 WildReceipt shifted-test pairs were excluded. WildReceipt represents a corpus shift within the same editing generator.

## Matched crop experiment

The crop loader decodes native RGB and aligns each authentic crop with the same coordinates in its edited counterpart. It checks oriented dimensions and binary masks. Small receipts use edge replication. When an edit extends beyond one patch, a deterministic covering grid retains boxes intersecting the mask. The median training edit bounds were 249.5 × 34 pixels. Training crops covered a median of 29.5% mask foreground at 112 pixels. The measured changed-pixel counts outside the native mask were zero across the tight training and oracle validation crops.

The 12,177-parameter CNN uses RGB and a local high-pass residual, four strided convolution blocks, global average and maximum feature pooling, and a linear head trained with BCE. The runs used six epochs, AdamW at 0.001, eight matched pairs per batch, and weights selected on oracle validation AUROC. The CPU runtime was PyTorch 2.12.0+cpu on a 16-logical-core Windows machine with about 7.1 GiB available RAM at startup. CUDA was unavailable. Decoded pairs use a two-pair cache. Materialized RGB crops are bounded to 160 MiB per split by default. Checkpoints contain optimizer state and epoch history for resume.

| Run | Selected epoch | Train AUROC | Train accuracy | Oracle AUROC | AP | Accuracy | Macro F1 | Edit recall | FPR | BCE train / val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| context-17 | 6 | 0.956 | 82.2% | 0.526 | 0.530 | 51.8% | 0.482 | 25.4% | 21.8% | 0.461 / 0.720 |
| context-42 | 4 | 0.918 | 80.9% | 0.520 | 0.520 | 50.9% | 0.481 | 74.0% | 72.2% | 0.584 / 0.696 |
| context-71 | 5 | 0.947 | 86.5% | 0.545 | 0.547 | 53.1% | 0.530 | 58.5% | 52.2% | 0.491 / 0.695 |
| tight-17 | 5 | 0.968 | 89.2% | 0.595 | 0.593 | 57.2% | 0.571 | 55.8% | 41.5% | 0.356 / 0.707 |
| tight-42 | 5 | 0.964 | 89.2% | 0.567 | 0.550 | 55.7% | 0.554 | 47.6% | 36.2% | 0.407 / 0.702 |
| tight-71 | 6 | 0.979 | 92.0% | 0.559 | 0.560 | 53.8% | 0.535 | 45.8% | 38.2% | 0.311 / 0.728 |
| fixed-tight-42 | 6 | 0.978 | 90.2% | 0.551 | 0.547 | 53.9% | 0.535 | 62.9% | 55.1% | 0.369 / 0.711 |

Across three seeds, oracle AUROC averaged **0.574 ± 0.019** for 112-pixel crops and **0.531 ± 0.013** for 224-pixel crops (mean ± sample SD). The 112-pixel model's training AUROC near 0.96 and validation AUROC below 0.60 indicate substantial overfitting. The tighter setup changes both field of view and the number of covering crops, so the comparison cannot isolate context alone. Crop-level metrics weight sources with multiple crops more heavily, and checkpoint selection uses these same validation sources. This is an exploratory oracle result with known edit locations.

### Shortcut checks

At seed 42, a logistic classifier using channel means, standard deviations, and adjacent-pixel difference statistics achieved oracle AUROC **0.547**. Applying identical grayscale conversion or JPEG quality-85 re-encoding to both classes at inference gave **0.566** and **0.568** with the selected RGB model. These controls suggest the modest signal survives those transforms. They do not rule out reconstruction, texture, or other generator-specific artifacts. Identical native crop coordinates and symmetric transforms remove a direct crop-position or pipeline distinction between classes within each pair. File bytes and file format were inspected in Phase A and were never given as inputs here. The median tight RGB probabilities at seed 42 were 0.460 for authentic and 0.492 for edited crops. Full score quantiles and curves are in the run JSON.

## Image-only fixed tiling

The receipt is tiled at 112 × 112 and stride 84 (25% overlap), with explicit final windows at the page edges and no image stretch. The experiment compares (A) an oracle-crop-trained CNN applied to every fixed tile, (B) a fixed-tile-trained CNN evaluated on those same image-only tiles, and (C) a document-level logistic aggregator trained on fixed max, top-three, and mean logits from training receipts. This aggregator uses the fixed-tile CNN as a frozen encoder. The fixed-tile CNN uses up to two training tiles per pair whose edit-mask area exceeds 0.5% of tile area, sorted by overlap. Each selected tile is paired with the identical authentic region. Low-coverage tiles are omitted from patch training and all positions are scored at inference. Training masks determine supervision and sampling from a fixed grid. Validation masks are read after prediction for localization scoring.

| Training setup | Pooling | Receipt AUROC | AP | Edit recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| tight-42 | max | 0.498 | 0.499 | 100.0% | 100.0% |
| tight-42 | top3 | 0.494 | 0.498 | 100.0% | 100.0% |
| tight-42 | mean | 0.506 | 0.520 | 46.9% | 46.9% |
| fixed-tight-42 | max | 0.520 | 0.514 | 100.0% | 100.0% |
| fixed-tight-42 | top3 | 0.518 | 0.511 | 100.0% | 100.0% |
| fixed-tight-42 | mean | 0.506 | 0.521 | 54.7% | 56.3% |
| fixed-tight-42 | trained_pooling | 0.510 | 0.522 | 54.7% | 51.6% |

The 128 validation receipts generated 25,958 tiles. The median was 144 per receipt and the maximum was 1,372. For edited receipts, the best available tile covered a median of 44.8% of the annotated edit. The oracle-trained CNN's top tile hit the mask on 6.3% of edits (mean tile-mask IoU 0.005). The fixed-tile CNN reached 12.5% (IoU 0.009). Uniform tile selection would hit 6.8% on average. The fixed CNN's max-pooled AUROC was 0.525 below 144 tiles and 0.517 at or above 144. Splitting by whether any one tile covered half the edit yielded 0.518 and 0.524. Each stratum is small and exploratory.

Median end-to-end CPU latency was 124 ms for the oracle-trained model and 165 ms for the fixed-tile-trained model, with p95 of 693 and 945 ms. Timing includes image decode, padding, model scoring, and aggregation. Runs shared CPU time and the medians are descriptive. The six oracle training runs took approximately 107–150 seconds each, and the fixed-tile training and evaluation run took 228 seconds.

## Decision

The native crops show a weak, repeatable oracle signal at 112 pixels. This screen found no credible whole-receipt transfer. The fixed-tile model found the edit more often than a uniform tile but ranked complete receipts near chance. Many candidate tiles, split edits, and the absence of ordinary background negatives from patch training leave room for improvement. The next controlled change should add hard negatives from unedited locations on the same source, including off-mask regions of edited receipts, then repeat the same image-only validation screen with a frozen pooling comparison. The results do not justify a pretrained rollout or a change to public inference.

## Reproduction

Use `py -m research.patch_probe` with `--manifest data/aiforge/manifest_deduplicated.csv --asset-root data/aiforge/assets --epochs 6` and a unique `--output` under `artifacts/phase-b/`. Oracle runs used `--side 112` or default 224, with `--seed 17`, `42`, or `71`. For the seed-42 tight checkpoint, `--resume --tiled --tile-stride 84 --tile-pairs 64 --shortcuts` evaluates fixed tiling and the symmetric preprocessing stress controls without additional epochs. The fixed-trained run used `--side 112 --stride 84 --threshold .005 --tiled --tile-pairs 64 --threads 5`. Seed-42 context used five threads and the other oracle runs used four. The run directories hold checkpoints, epoch logs and aggregate JSON. The project test suite passed all 24 tests with the Windows socket-free runner.

AI assistance implemented, executed, and interpreted this engineering experiment. The repository contains code and aggregate findings. Dataset imagery, raw crops, and checkpoints remain local.
