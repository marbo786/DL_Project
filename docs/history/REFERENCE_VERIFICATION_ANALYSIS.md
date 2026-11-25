# ForgeryLens Phase E reference-assisted verification

## Task and trust assumptions

The task is to compare a candidate against a trusted copy of the same receipt and flag localized visual changes. The caller must establish the reference's provenance. A detected change is evidence of a difference. It cannot establish fraudulent intent, reference authenticity, or whether a revised field was legitimate. This is a two-image task. [Phase D image-only results](MIL_ANALYSIS.md) describe a different input protocol and remain a negative finding.

## Data and nuisance protocol

The AIForge-Doc v2.0.1 source manifest matched SHA-256 `3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80`. We used 128 training source pairs for the learned comparison and all 228 validation source pairs for every measured regime. Validation sources were grouped into a fixed 64-pair calibration subset and a disjoint 164-pair evaluation subset. Thresholds and logistic calibration were fitted only on the 64 validation-calibration pairs. Training and validation were checked for source overlap. Neither final test set was accessed.

Each validation source provided an authentic candidate and its edited counterpart against the same original. All images were decoded, EXIF-oriented, and resized to an aspect-preserving common raster with a maximum side of 512 pixels. The exact diagnostic used the same raster with no added nuisance. For the other regimes, each candidate received an independent deterministic draw from the same nuisance distribution, including JPEG encoding, rotation, scale and translation, perspective jitter, reflect-fill at the edge, blur, brightness, contrast, color gain, and noise. Moving the candidate within the output frame simulates some cropping. The ordinary regime used rotations up to 2.5°, scale 0.97–1.045, and JPEG quality 65–95. The difficult regime used rotations up to 6°, scale 0.92–1.13, and quality 42–95, along with stronger shifts and photometric changes.

The exact same edit geometry was available to the generator for both class draws, so the visibility filter had the same policy for authentic and edited candidates. It resampled geometry if retained raster mask area fell below 65% of the original raster mask, with a deterministic identity fallback after 20 attempts. The authentic label's final mask remained empty. Masks and physical support used nearest-neighbor geometric warps. Downsampling the source masks used area-to-binary projection to retain thin regions. The inference functions took images only. True transforms and warped masks were used after inference for diagnostics and localization. These are simulated nuisances. There were no independently acquired camera pairs or legitimate content revisions in Phase E.

## Classical and learned comparisons

The simple alignment baseline used the common raster with an identity transform. ORB matched grayscale descriptors and estimated a candidate-to-reference homography using RANSAC. Registration checks included inlier count and fraction, spatial spread, median reprojection residual, corner displacement, and estimated valid overlap. Accepted predictions exclude warped pixels outside the candidate support and an eroded rim. Rejected registrations use an identity diagnostic score in the all-pair metrics. The selective analysis withholds those predictions.

Classical maps use absolute RGB difference, robust photometrically normalized grayscale difference, or structural dissimilarity. The receipt score is the mean of the highest 0.5% of valid pixel differences. Local predictions use configured fixed heatmap thresholds, morphological opening, and connected components with at least five pixels. IoU and localization F1 average over edited candidates, using valid support in the applicable frame.

The 6,090-parameter Siamese CNN shares two convolution blocks across the images. It fuses the absolute feature difference and feature product, then classifies pooled features. The supervised localization variant adds a spatial BCE loss at weight 0.35 and foreground weight 8. The learned models use a common 160 × 224 input, which changes aspect ratio equally for both inputs. They trained for six epochs with AdamW at 0.001 on both ordinary and difficult transformed examples. The controlled variants changed registration and then mask supervision. Exact pairs were reserved for the classical diagnostic. A classification-only model has no supervised heatmap, so its localization scores are omitted.

### Registration quality

Grid error is the median of nine known-coordinate probes at evaluation time in pixels of the 512-side raster. Its use does not affect the estimated transform.

| Regime | Accepted | Genuine accepted | Edited accepted | Raw median error px | Accepted median error px | Accepted median reprojection px | Median ORB ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact | 100.0% | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 | 62.4 |
| transformed | 99.8% | 100.0% | 99.6% | 9.292 | 0.328 | 0.801 | 62.7 |
| difficult | 96.7% | 97.8% | 95.6% | 25.870 | 0.522 | 0.946 | 60.8 |

### Evaluation on 164 independent calibration-holdback pairs per regime

AUROC and average precision use raw scores. Precision, recall, FPR, and macro F1 in this table use calibrated probability 0.5. Localization thresholds are fixed across regimes. Only the registered methods can abstain on alignment.

| Regime | Model | AUROC | AP | Macro F1 | Precision | Recall | FPR | Pixel IoU | Pixel F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact | Raw RGB absolute difference | 1.000 | 1.000 | 0.824 | 100.0% | 65.9% | 0.0% | 0.237 | 0.362 |
| exact | Unregistered normalized grayscale difference | 0.989 | 0.975 | 0.749 | 97.8% | 53.7% | 1.2% | 0.128 | 0.211 |
| exact | Registered RGB difference | 1.000 | 1.000 | 0.831 | 100.0% | 67.1% | 0.0% | 0.234 | 0.358 |
| exact | Registered normalized grayscale difference | 0.989 | 0.976 | 0.764 | 97.9% | 56.1% | 1.2% | 0.126 | 0.208 |
| exact | Registered SSIM | 1.000 | 1.000 | 0.957 | 100.0% | 91.5% | 0.0% | 0.619 | 0.737 |
| transformed | Raw RGB absolute difference | 0.494 | 0.500 | 0.484 | 50.4% | 71.3% | 70.1% | 0.014 | 0.027 |
| transformed | Unregistered normalized grayscale difference | 0.490 | 0.496 | 0.503 | 51.9% | 75.0% | 69.5% | 0.017 | 0.033 |
| transformed | Registered RGB difference | 0.552 | 0.526 | 0.456 | 48.1% | 23.8% | 25.6% | 0.044 | 0.081 |
| transformed | Registered normalized grayscale difference | 0.553 | 0.534 | 0.451 | 48.6% | 21.3% | 22.6% | 0.063 | 0.112 |
| transformed | Registered SSIM | 0.647 | 0.618 | 0.591 | 63.1% | 47.0% | 27.4% | 0.104 | 0.172 |
| transformed | Siamese without registration | 0.512 | 0.523 | 0.501 | 50.3% | 43.9% | 43.3% | — | — |
| transformed | Siamese with registration | 0.507 | 0.496 | 0.496 | 49.7% | 45.7% | 46.3% | — | — |
| transformed | Siamese with registration and pixel loss | 0.478 | 0.490 | 0.461 | 45.9% | 34.1% | 40.2% | 0.000 | 0.000 |
| difficult | Raw RGB absolute difference | 0.502 | 0.505 | 0.478 | 48.3% | 35.4% | 37.8% | 0.010 | 0.020 |
| difficult | Unregistered normalized grayscale difference | 0.499 | 0.503 | 0.481 | 49.5% | 32.3% | 32.9% | 0.013 | 0.026 |
| difficult | Registered RGB difference | 0.518 | 0.521 | 0.485 | 52.9% | 27.4% | 24.4% | 0.018 | 0.035 |
| difficult | Registered normalized grayscale difference | 0.534 | 0.528 | 0.472 | 51.9% | 24.4% | 22.6% | 0.047 | 0.085 |
| difficult | Registered SSIM | 0.600 | 0.573 | 0.554 | 60.2% | 39.6% | 26.2% | 0.051 | 0.091 |
| difficult | Siamese without registration | 0.491 | 0.498 | 0.486 | 49.8% | 64.0% | 64.6% | — | — |
| difficult | Siamese with registration | 0.483 | 0.469 | 0.494 | 49.4% | 51.2% | 52.4% | — | — |
| difficult | Siamese with registration and pixel loss | 0.484 | 0.480 | 0.487 | 49.5% | 61.0% | 62.2% | 0.000 | 0.000 |

All-pair AUROC for raw RGB collapsed from 1.000 on exact pairs to 0.494 under ordinary nuisance. Registration and SSIM improved ordinary AUROC to 0.647, but localization IoU remained 0.104. Under difficult nuisance, the same method reached 0.600 AUROC and 0.051 IoU. None of the seed-42 Siamese comparisons was promising enough for additional seeds. Its supervised spatial head predicted no useful regions at the fixed 0.5 mask threshold. The different classical and learned raster resolutions limit claims about representation quality. The three learned controls use the same resolution and show no benefit from adding registration or localization supervision to that recipe.

### Operating points and latency

Each cutoff was based on the genuine calibration quantile. The table shows achieved evaluation recall and FPR for registered SSIM, including rejected cases using the diagnostic fallback.

| Regime | Nominal 5% FPR, recall / actual FPR | Nominal 10%, recall / actual FPR | Nominal 20%, recall / actual FPR |
| --- | ---: | ---: | ---: |
| exact | 100.0% / 1.2% | 100.0% / 11.0% | 100.0% / 20.7% |
| transformed | 6.7% / 4.9% | 26.8% / 9.8% | 35.4% / 15.2% |
| difficult | 11.6% / 9.8% | 16.5% / 11.6% | 29.9% / 18.3% |

| Registered SSIM | Brier | ECE, 10 bins | Alignment abstention | Coverage / error at 0.70 confidence | Coverage / error at 0.80 confidence | Median / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact | 0.034 | 0.113 | 0.0% | 91.8% / 1.0% | 87.5% / 0.0% | 107 / 124 |
| transformed | 0.238 | 0.085 | 0.0% | 23.2% / 32.9% | 9.1% / 16.7% | 103 / 124 |
| difficult | 0.251 | 0.074 | 3.4% | 14.0% / 45.7% | 5.2% / 41.2% | 102 / 122 |

These timings include registration, map scoring and morphological components on already decoded common rasters. A separate 48-comparison CPU benchmark, including PNG decode at the common raster, measured **123 ms median and 146 ms p95** end to end for registered SSIM. It excludes upload, native large-file decode, and response serialization. Ordinary transformed registered Siamese preprocessing and forward latency was about 83 ms median, versus 9 ms without registration, with decoded inputs and shared CPU load. JSON artifacts include all evaluated risk-coverage points and the calibration metrics for every method.

## Failure analysis and decision

The local alignment previews show that subpixel residuals and blur still create strong contours at unchanged lettering, artwork and dotted rules. One rejected difficult receipt had concentrated ORB matches in a small text block surrounded by blank page. The safety checks correctly withheld that alignment despite numerous local matches. Raw contrast and brightness changes dominated unregistered difference scores. In the main ordinary regime even the strongest method recalled only **6.7% at 4.9% false positives**. Raising confidence to 0.80 reduced ordinary coverage to 9.1% and still left 16.7% error; the difficult subset was worse. This is evidence of modest reference-assisted signal, with limited robustness and inadequate selective performance.

Preserve registered SSIM as the transparent benchmark. Stop extending the Siamese architecture on the current data. The next experiment is a paired acquisition study with legitimate recaptures from the same physical document and legitimate revised documents reviewed by an independent annotator. Freeze a conservative abstention and operating-point policy before scoring these data. Investigate edge-residual tolerance and local photometric correction using development pairs first. [Proposed comparison contract](REFERENCE_COMPARISON_CONTRACT.md)

## Fresh independent evaluation feasibility

- **AIForge-Doc v1** uses Gemini 2.5 Flash Image and Ideogram v2 Edit, with authentic test mirrors and gated research access. The v2 publisher reports 3,062 same-spec overlaps with v1. A version-only split therefore cannot establish source independence. With authorized access, join canonical upstream IDs, spec IDs, and decoded authentic-image hashes, and exclude any linked source from both development and calibration. Review visually near-duplicate authentic captures before freezing a generator holdout. [v1 dataset card](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v1) and [v2 pairing notes](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2)
- **DocTamper** covers a distinct non-AIGC mechanism. The official repository lists gated non-commercial access and points to authentic images from the authors' TFR dataset. A source-level join, mask correspondence, and permission for each source still need to be verified before including it in paired evaluation. [DocTamper release and access terms](https://github.com/qcf-568/DocTamper)
- No fresh source- and generator-disjoint paired benchmark was established in this phase. The external audit is based on release documentation. No external identities were declared disjoint without matched files. A frozen evaluation should reserve independent source roles and include actual recaptures, legitimate content changes, mismatched references, and different manipulation mechanisms.

## Reproduction and disclosure

Use `research/requirements-reference.txt` for the research-only OpenCV dependency. The raw, registered, and wire experiments use `python -m research.reference_classical` with the pinned manifest, asset root and a fresh output path. Use `--raw-only` for the raw run and `--pairs 24 --wire-bench` for decoding plus comparison timing. The learned comparison uses `python -m research.reference_siamese`, the same manifest and assets, `--cache artifacts/phase-e/cache`, and a fresh output path. The default configuration is 128 training pairs, 228 validation pairs, and six epochs at seed 42. Phase E galleries, scores, curves and weights are local under `artifacts/phase-e/`. The 38-test suite passed with the Windows socket-free ASGI runner.

AI assistance implemented and ran this research investigation and documented its measured results. No private datasets, derived images, checkpoints or API activation are included in the commit.
