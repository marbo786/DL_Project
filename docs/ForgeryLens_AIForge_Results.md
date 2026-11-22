# ForgeryLens AIForge-Doc benchmark

23 November 2026

The model selected on validation was the **local teacher**. Its AUROC was **0.500** on CORD and SROIE and **0.504** on the held-out WildReceipt corpus. Mean mask IoU was **0.022** and **0.026**, respectively.

Both results indicate chance-level discrimination and little overlap between the predicted and annotated edit regions.

## Dataset and method

AIForge-Doc v2.0.1 pairs authentic source receipts with GPT-Image-2 edits and binary masks. The full package provided 2,648 receipt pairs. The initial audit found byte-identical documents under different source IDs. Deduplication removed 31 entire source groups, prioritizing WildReceipt, the in-domain test set, validation, and training in that order. The resulting 2,617 pairs passed the full image, source, and mask audit.

| Partition | CORD pairs | SROIE pairs | WildReceipt pairs | Total pairs |
| --- | ---: | ---: | ---: | ---: |
| Train | 616 | 175 | 0 | 791 |
| Validation | 168 | 60 | 0 | 228 |
| In-domain test | 197 | 65 | 0 | 262 |
| Shifted test | 0 | 0 | 1,336 | 1,336 |

The experiment trained six configurations for eight epochs each using seeds 17, 42, and 71. The model reads a 112 × 160 grayscale image. For the local models, pixel mask IoU is measured on the 28 × 40 heatmap grid for edited examples. Validation selected the checkpoint, calibration, and confidence threshold before either holdout was scored.

## Held-out results

| Set | Images | Macro F1 | ROC-AUC | Average precision | Edited recall | Mean mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CORD + SROIE | 524 | 0.488 | 0.500 | 0.503 | 58.4% | 0.022 |
| WildReceipt | 2,672 | 0.498 | 0.504 | 0.505 | 63.3% | 0.026 |

Macro F1 and recall use a calibrated edit-probability threshold of 0.5. The confidence filter is evaluated separately below.

## Model comparisons

| Model | CORD + SROIE F1 | CORD + SROIE AUROC | WildReceipt F1 | WildReceipt AUROC | Parameters | Checkpoint bytes | Median CPU forward pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Whole-image baseline | 0.500 | 0.501 | 0.489 | 0.501 | 4,818 | 27,133 | 1.39 ms |
| Local teacher | 0.488 | 0.500 | 0.498 | 0.504 | 4,890 | 26,857 | 1.84 ms |
| FP16 storage | 0.488 | 0.500 | 0.498 | 0.504 | 4,890 | 17,069 | 1.79 ms |
| Distilled student | 0.501 | 0.502 | 0.500 | 0.502 | 1,294 | 12,921 | 1.49 ms |

The selected teacher was trained with seed 42. FP16 stores reduced-precision weights and loads them into a float32 CPU model. The student reduces the channel width from eight to four.

## Validation ablations

| Configuration | ROC-AUC, mean ± SD | Mask IoU, mean ± SD |
| --- | ---: | ---: |
| Whole-image baseline | 0.500 ± 0.000 | — |
| Local teacher | 0.503 ± 0.002 | 0.016 ± 0.004 |
| No high-pass residual | 0.502 ± 0.001 | 0.015 ± 0.004 |
| Average pooling | 0.500 ± 0.000 | 0.026 ± 0.007 |
| No global context | 0.503 ± 0.001 | 0.017 ± 0.003 |
| Classification loss only | 0.501 ± 0.000 | — |

## Calibration and image stress tests

The run fell back to a confidence cutoff of **0.990** because no tested cutoff achieved both nonzero validation coverage and at most 10% error on accepted predictions. Coverage is the share of predictions accepted. ECE is expected calibration error.

| Set | Coverage | Accepted error | ECE |
| --- | ---: | ---: | ---: |
| CORD + SROIE | 0.0% | — | 0.012 |
| WildReceipt | 0.0% | — | 0.003 |

| CORD + SROIE condition | Images | Macro F1 | ROC-AUC | Average precision | Edited recall | Mean mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unperturbed | 524 | 0.488 | 0.500 | 0.503 | 58.4% | 0.022 |
| Blur | 524 | 0.489 | 0.502 | 0.501 | 40.8% | 0.011 |
| Low light | 524 | 0.370 | 0.501 | 0.503 | 4.6% | 0.000 |
| JPEG compression | 524 | 0.493 | 0.501 | 0.505 | 59.5% | 0.022 |
| Occlusion | 524 | 0.491 | 0.501 | 0.505 | 61.8% | 0.020 |

## Shifted errors ranked by confidence

| WildReceipt evaluation index | True label | Predicted label | Edit probability | Mask IoU |
| ---: | --- | --- | ---: | ---: |
| 2397 | Edited | Authentic | 0.480 | 0.000 |
| 1851 | Edited | Authentic | 0.480 | 0.000 |
| 279 | Edited | Authentic | 0.483 | 0.000 |
| 503 | Edited | Authentic | 0.484 | 0.000 |
| 1083 | Edited | Authentic | 0.485 | 0.000 |

Indices are zero-based positions in the cleaned shifted evaluation split.

## Export and scope

The NumPy export was 20,477 bytes. Across 80 fictional parity inputs, maximum absolute logit error was 2.98e-07 for the score and 3.58e-06 for the heatmap. Its median warm CPU forward pass was 4.34 ms.

Both test sets use the same editing generator. WildReceipt measures source-corpus shift. The data cover paired benchmark receipts with one localized edit per authentic source. Exact-byte checks cannot rule out re-encoded duplicates or visually similar captures. The chance-level discrimination and low mask overlap limit this model to research use.

The measured JSON, checkpoints, and export are under `artifacts/aiforge`.

Run ID `manifest-cpu-8ep-791src-v2`

Manifest SHA-256 `3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80`

[AIForge-Doc v2 dataset card](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2)
