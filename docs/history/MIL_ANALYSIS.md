# ForgeryLens Phase D document-level supervision

## Hypothesis and protocol

The experiment tested whether document-label supervision over fixed receipt tiles could improve single-image forgery detection while retaining the localization evidence from Phase C. The frozen Phase C random-negative PatchNet was the first encoder. The same native 112 × 112 RGB tiles, 84-pixel stride, deterministic edge coverage and edge padding were used throughout. Each variable-length bag contained every tile in its receipt. One authentic or edited label applied to the whole bag.

The pinned manifest SHA-256 was `3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80`. All runs used the same 128 training source pairs. Screening used the same 64 validation pairs as Phase C. The full validation comparisons used all 228 pairs. Sources were checked for disjointness. No final-test or shifted-test receipts were loaded. Document features used one image without mask coordinates, reference images, document metadata or source identifiers. Validation masks were loaded separately for localization scoring only.

PatchNet's frozen 64-dimensional feature vector comes from mean/max pooling of the last convolution maps, before the Phase C tile head. Its original weight file SHA-256 was `b71e98c24f10bf3ab9d4382330ed9eb9d75f30eafda705ffea3a3cb84ce3de04`. The reusable feature cache records the encoder state fingerprint, manifest, preprocessing identifier, native tile geometry, split, source, image label and feature dimension. It stores embeddings and boxes, with no raw receipt pixels. Extraction batches up to 32 tiles, and head training processes one unpadded bag at a time while accumulating gradients across eight source pairs. Optional batch padding masks are tested separately. Encoder parameters remained frozen and feature extraction ran without gradients.

The learned study used 12 epochs of AdamW at 0.001 and fixed epoch-12 heads. The nonspatial head projects standardized-per-tile inputs to 24 hidden units, then applies normalized attention and a binary classifier. The spatial head averages embeddings in 168-pixel centroid cells and applies the same head to those groups. Its group weights are spread equally over the group's member tiles for the localization comparison. The logistic summary control uses fixed pooling statistics, upper score quantiles, and log tile count. The tile-count-only control receives only log count. The shuffled-label control swaps authentic and edited labels within selected training pairs using the run seed. No shuffled labels enter validation.

## PatchNet screening and full validation

The reused fixed max policy reproduced Phase C AUROC of 0.5043945 and hit@3 of 56.25%. All confidence intervals use 1,000 bootstrap draws over source pairs. Recall, FPR and balanced accuracy in these score tables use logit zero. Training-genuine quantile thresholds appear separately below.

| Rule | AUROC (95% CI) | AP | Macro F1 | Recall | FPR | Balanced accuracy | Hit@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max | 0.504 [0.486, 0.522] | 0.505 | 0.375 | 96.9% | 95.3% | 50.8% | 56.3% |
| top3 | 0.502 [0.486, 0.518] | 0.504 | 0.401 | 95.3% | 92.2% | 51.6% | 56.3% |
| mean | 0.507 [0.501, 0.518] | 0.518 | 0.347 | 1.6% | 1.6% | 50.0% | 56.3% |
| top5_percent | 0.508 [0.498, 0.520] | 0.509 | 0.433 | 84.4% | 84.4% | 50.0% | 56.3% |
| count | 0.500 [0.500, 0.500] | 0.500 | 0.333 | 100.0% | 100.0% | 50.0% | — |
| summary | 0.501 [0.482, 0.519] | 0.497 | 0.472 | 29.7% | 31.3% | 49.2% | — |
| attention-42 | 0.517 [0.500, 0.537] | 0.513 | 0.508 | 48.4% | 46.9% | 50.8% | 45.3% |
| spatial-42 | 0.507 [0.492, 0.523] | 0.511 | 0.500 | 48.4% | 48.4% | 50.0% | 29.7% |
| shuffled_labels-42 | 0.508 [0.490, 0.528] | 0.522 | 0.501 | 68.8% | 65.6% | 51.6% | 50.0% |

On all 228 validation pairs, neither attention nor spatial pooling gave useful discrimination. The default score distribution head also stayed near chance.

| Rule | AUROC (95% CI) | AP | Macro F1 | Recall | FPR | Balanced accuracy | Hit@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max | 0.503 [0.495, 0.511] | 0.501 | 0.376 | 95.2% | 94.7% | 50.2% | 46.9% |
| top3 | 0.505 [0.500, 0.510] | 0.503 | 0.398 | 92.5% | 91.7% | 50.4% | 46.9% |
| mean | 0.503 [0.502, 0.506] | 0.507 | 0.354 | 2.6% | 3.1% | 49.8% | 46.9% |
| top5_percent | 0.505 [0.502, 0.509] | 0.505 | 0.432 | 84.6% | 84.6% | 50.0% | 46.9% |
| count | 0.500 [0.500, 0.500] | 0.500 | 0.333 | 100.0% | 100.0% | 50.0% | — |
| summary | 0.508 [0.499, 0.516] | 0.501 | 0.470 | 28.5% | 29.8% | 49.3% | — |
| attention-17 | 0.510 [0.503, 0.519] | 0.509 | 0.498 | 68.4% | 65.8% | 51.3% | 41.7% |
| attention-42 | 0.509 [0.502, 0.516] | 0.509 | 0.509 | 60.5% | 57.9% | 51.3% | 41.7% |
| attention-71 | 0.503 [0.500, 0.506] | 0.508 | 0.492 | 37.3% | 37.3% | 50.0% | 8.8% |
| spatial-17 | 0.505 [0.499, 0.511] | 0.509 | 0.494 | 43.9% | 44.7% | 49.6% | 31.1% |
| spatial-42 | 0.504 [0.498, 0.509] | 0.507 | 0.500 | 48.2% | 48.2% | 50.0% | 30.7% |
| spatial-71 | 0.504 [0.502, 0.508] | 0.507 | 0.499 | 42.1% | 41.7% | 50.2% | 21.9% |
| shuffled_labels-17 | 0.497 [0.484, 0.510] | 0.501 | 0.403 | 91.7% | 90.8% | 50.4% | 42.1% |
| shuffled_labels-42 | 0.507 [0.499, 0.516] | 0.508 | 0.513 | 63.6% | 59.6% | 52.0% | 40.8% |
| shuffled_labels-71 | 0.499 [0.494, 0.504] | 0.503 | 0.494 | 55.3% | 56.1% | 49.6% | 10.5% |

Attention seed 42 improved AUROC relative to fixed max by 0.005 on the full split (paired interval [-0.003, 0.014]). The shuffled-label seed 42 achieved a similar AUROC and hit@3. Fixed PatchNet tiles retained stronger localization than the learned ranks. The D2 end-to-end experiment was gated off because no D1 head showed useful separation.

## Matched MobileNetV3 comparison

MobileNetV3-Small pooled its final feature tensor to a 576-dimensional vector. Both variants used identical native RGB tiles and ImageNet channel normalization, with no resize to 224 or center crop. The pretrained variant used torchvision `IMAGENET1K_V1` with verified download hash. The source checkpoint SHA-256 was `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`. The random control used the same architecture and a fixed encoder initialization seed of 42. Head initialization alone varied across seeds. Both encoders used the same heads, training budget, and input bags.

| Head and seed | Random AUROC | ImageNet AUROC | Paired AUROC difference (95% CI) | Random hit@3 | ImageNet hit@3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention-17 | 0.504 | 0.510 | 0.005 [-0.006, 0.020] | 28.1% | 15.6% |
| attention-42 | 0.502 | 0.516 | 0.014 [0.005, 0.029] | 10.9% | 34.4% |
| attention-71 | 0.503 | 0.514 | 0.012 [0.002, 0.029] | 18.8% | 37.5% |
| spatial-17 | 0.499 | 0.508 | 0.009 [0.001, 0.023] | 15.6% | 10.9% |
| spatial-42 | 0.499 | 0.508 | 0.009 [0.002, 0.023] | 7.8% | 15.6% |
| spatial-71 | 0.499 | 0.510 | 0.011 [0.004, 0.026] | 10.9% | 17.2% |

ImageNet weights consistently increased point estimates on the screen relative to their randomized counterpart. The highest AUROC was 0.516. The same matched comparison was repeated on all 228 validation pairs at seed 42 rather than selecting a model from the small screen.

| Rule | AUROC (95% CI) | AP | Macro F1 | Recall | FPR | Balanced accuracy | Hit@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PatchNet max | 0.503 [0.495, 0.511] | 0.501 | 0.376 | 95.2% | 94.7% | 50.2% | 46.9% |
| PatchNet attention | 0.509 [0.502, 0.516] | 0.509 | 0.509 | 60.5% | 57.9% | 51.3% | 41.7% |
| PatchNet spatial | 0.504 [0.498, 0.509] | 0.507 | 0.500 | 48.2% | 48.2% | 50.0% | 30.7% |
| MobileNet random attention | 0.500 [0.498, 0.501] | 0.500 | 0.333 | 100.0% | 100.0% | 50.0% | 8.3% |
| MobileNet random spatial | 0.500 [0.498, 0.501] | 0.501 | 0.333 | 100.0% | 100.0% | 50.0% | 7.9% |
| MobileNet ImageNet attention | 0.507 [0.505, 0.512] | 0.511 | 0.473 | 75.9% | 74.6% | 50.7% | 34.6% |
| MobileNet ImageNet spatial | 0.503 [0.501, 0.506] | 0.509 | 0.503 | 55.3% | 54.4% | 50.4% | 12.7% |

On full validation the ImageNet attention gain over random initialization was 0.008 with paired interval [0.005, 0.012]. The spatial gain was 0.003 with interval [0.001, 0.006]. The intervals quantify source sampling uncertainty conditional on the models and these partitions. They exclude variation from changing the training corpus. Pretrained attention remained close to chance and below PatchNet attention on the same split.

## Operating points, localization and efficiency

Each cutoff below was set at the 95th, 90th, or 80th percentile of genuine **training** receipt scores for the corresponding model. The table gives achieved validation recall and FPR because the training quantile does not guarantee the target FPR out of sample.

| Full validation, seed 42 | Nominal 5% FPR, recall / FPR | Nominal 10%, recall / FPR | Nominal 20%, recall / FPR |
| --- | ---: | ---: | ---: |
| PatchNet max | 3.5% / 3.5% | 7.0% / 7.0% | 12.7% / 13.2% |
| PatchNet attention | 11.0% / 11.4% | 19.7% / 19.3% | 28.1% / 26.3% |
| MobileNet random attention | 8.3% / 8.3% | 14.0% / 13.6% | 25.9% / 25.9% |
| MobileNet ImageNet attention | 14.0% / 14.0% | 19.3% / 19.7% | 26.3% / 25.0% |

| Full validation, seed 42 | Hit@1 | Hit@3 | Hit@5 | Top-tile IoU | Median / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| PatchNet max | 24.6% | 46.9% | 58.3% | 0.035 | 230 / 2131 |
| PatchNet attention | 23.7% | 41.7% | 60.1% | 0.033 | 231 / 2133 |
| PatchNet spatial | 17.5% | 30.7% | 46.1% | 0.021 | 231 / 2134 |
| MobileNet random attention | 2.6% | 8.3% | 12.3% | 0.001 | 363 / 3080 |
| MobileNet random spatial | 2.6% | 7.9% | 11.4% | 0.003 | 364 / 3083 |
| MobileNet ImageNet attention | 15.8% | 34.6% | 51.8% | 0.016 | 360 / 3139 |
| MobileNet ImageNet spatial | 5.7% | 12.7% | 17.5% | 0.005 | 361 / 3140 |
| Uniform | 5.4% | 14.9% | 23.2% | — | — |

The reported latency reconstructs a full pass from the measured image decode, crop extraction and encoder time at cache creation, plus measured head or fixed-pooling time. Training and extraction ran on Windows CPU with shared load. The cached timing for some full runs includes bags from the prior screen. These figures support rough runtime comparisons. Whole-process peak RSS was unavailable.

| Full run | Encoder parameters | Encoder state MiB | Attention head parameters | Bag cache MiB | Seed-42 attention fit s |
| --- | ---: | ---: | ---: | ---: | ---: |
| PatchNet | 12,177 | 0.05 | 1,737 | 51.8 | 6.8 |
| MobileNet random | 927,008 | 3.58 | 15,049 | 397.3 | 6.3 |
| MobileNet ImageNet | 927,008 | 3.58 | 15,049 | 397.3 | 6.4 |

## Error analysis and decision

On the full split, PatchNet seed-42 attention assigned a lower score to the edited receipt in 98 matched pairs. The worst recorded inversions included both CORD and SROIE and spanned 40–170 tiles. AUROC was 0.505 below the median tile count and 0.510 at or above it. The analogous height-bin AUROCs were 0.505 and 0.510. Full-split medians of the authentic and edited sigmoid scores were 0.524 and 0.533. The ImageNet model's medians were 0.509 and 0.509. These overlapping scores yield low recall when false positives are constrained.

Seed-42 attention used a median effective tile count of 24 for PatchNet and 135 for ImageNet MobileNet, using the inverse sum of squared tile weights. Attention can concentrate on candidate regions and remain near chance as a receipt classifier. The shuffled-label localization results reinforce that attention rankings are descriptive and do not establish an explanation.

**Decision:** Close the PatchNet MIL branch of research at Gate C. The frozen pretrained comparison also failed to reach useful single-image discrimination. The next research question should target task or data formulation. Compare a reference-assisted verification task, where a trusted receipt exists, with the image-only task on the same source-disjoint data. For image-only detection, establish a fresh source- and generator-disjoint benchmark before another architecture sweep. The current dataset supports only corpus-shift evaluation with a shared editing generator. The production synthetic release remains unchanged. A real adapter still requires a useful operating point and frozen independent evaluation.

## Reproduction

Run `python -m research.mil_study` with the deduplicated manifest, receipt asset root, `--checkpoint artifacts/phase-c/random-42/final.pt`, `--cache artifacts/phase-d/cache` and a fresh `--output`. Use `--encoder patchnet`, `mobile-random`, or `mobile-pretrained`. Mobile runs use torchvision 0.27.0+cpu and a writable `TORCH_HOME`, with the official ImageNet weight hash checked. The screen uses `--val-pairs 64`. The full comparison uses `--val-pairs 228` without a 64-pair score reference. Seeds 17, 42 and 71 were used for both MobileNet screens and on the full PatchNet run. Seed 42 was used for the full MobileNet pair. The outputs include checkpointed heads, timing and confidence results, source-level diagnostics, and raw score arrays kept locally. The full test suite passed 32 tests using the Windows socket-free runner.

AI assistance implemented, ran and interpreted these measured engineering experiments. Source imagery, local features, weights, and private error details are excluded from the repository.
