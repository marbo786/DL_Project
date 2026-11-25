# ForgeryLens single-image experiment ledger

The [independent verification audit](SINGLE_IMAGE_INDEPENDENT_AUDIT.md) records reconstructed metrics, training diagnostics, fresh TorchScript parity, and current external-baseline status.

The [controlled receipt ablations](RECEIPT_ABLATIONS.md) report matched three-seed changes in the selected recipe. They include an inference-only whole-page control and a fixed-epoch legacy mask-loss comparison.

The [frozen residual stress study](RECEIPT_STRESS.md) reports paired cutoff changes, class-specific routing coverage, and exploratory open-set acceptance for the selected checkpoint.

The [attribution and failure audit](RECEIPT_EXPLAINABILITY.md) tests the highlighted tile against matched deletion controls, compares edit-site and background score pooling, and documents five development failures with regression baselines.

This ledger records the receipt study originally developed in `research/source-disjoint-forensics`. [Frozen protocol](SINGLE_IMAGE_PROTOCOL.md) · [Reproduction and access](SINGLE_IMAGE_REPRODUCTION.md) · [Historical experiment index](EXPERIMENT_SUMMARY.md)

## Matched development follow-up

The [course paper](ForgeryLens_Technical_Report.pdf) includes a common development baseline and ranking-loss pilot. Both use the contained fitting groups and the original ablation calibration/development subsets. The from-scratch baseline uses a width-16 whole-page CNN, seed 42 and eight fixed epochs. Its paired AUROC difference against residual-42 was -0.0060 (95% source-group interval -0.0163 to +0.0025). The historical width-eight baseline has separate ID and shifted results.

| Seed-42 design on the 162-image matched subset | AUROC | Recall at nominal 5% FPR | Observed FPR |
| --- | ---: | ---: | ---: |
| Whole-page width-16 CNN | 0.5004 | 6.17% | 6.17% |
| Matched-triplet PatchNet control | 0.4952 | 8.64% | 8.64% |
| Same PatchNet with pairwise ranking loss | 0.4991 | 11.11% | 11.11% |

The new student comparison holds the 21-triplet minibatches, sample order, seed, optimizer, epochs and inference policy fixed. The auxiliary loss favors the annotated edit over the matched original and off-edit negative. The paired AUROC difference was +0.0039 (95% source-group interval -0.0023 to +0.0112). Top-tile mask hits fell from 6/81 to 2/81 edited receipts. The pilot supplies no evidence for changing the frozen demo. All runs used development data only.

## Dataset and leakage audit

| Check | Measured outcome |
| --- | --- |
| AIForge v2.0.1 inventory | 3,066 paired specs, of which 2,648 are receipt specs and 418 are form specs. One v2 generator, GPT-Image-2. |
| Historical byte-deduplicated receipts | 2,617 source pairs. All 2,617 had valid aligned nonempty masks, changed pixels within the annotation, and no differing RGB pixels outside the mask. |
| Decoded image equality | No additional merged source IDs in the historical byte-deduplicated manifest. |
| Perceptual candidate screen | 192 authentic-image candidates under dHash distance ≤3 and a ±5% aspect-ratio tolerance. All were treated as possible aliases. The table includes a second grayscale thumbnail-error measurement. Human identity verification remains open. |
| Contained partition | 51 components spanning historical splits were quarantined, representing 130 source IDs. Fitting uses 585 pairs in 556 groups, calibration 145 pairs in 141 groups, and development 200 pairs in 199 groups. The retained historical in-domain and shifted sets have 221 and 1,336 pairs. |
| Local v1 pairing leads | The supplied v2 package records 2,688 `gemini-nano` and 378 `qwen-inpaint` assignments across receipts and forms. Its metadata differs from the currently published generator description. Image versions and paired specs require verification before reporting generator results. |

Median mask area was 0.386% for CORD, 1.420% for SROIE, and 1.414% for WildReceipt. Genuine and edited files in each pair share dimensions and a PNG raster. These data provide a useful controlled localized-edit task with many identical pixels. Compression, scanned documents, and camera captures exhibit different nuisance conditions.

The historical manifest SHA-256 is `3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80`. The exact-decoded audit output is `cfe3dad1d28fefefe86f5777ce1f8fc73bfa140c811b95926f62d74eb9d463f3`, and the contained manifest is `1c395905027f9cd2e88489c26a5e741627697898dda10357490bcf8abb1f4f65`. See `artifacts/receipt-protocol-v1/audit.json` and `artifacts/receipt-protocol-v1-contained/`.

Historical artifacts retained in the workspace include the synthetic release under `release/`, the original AIForge checkpoints and calibration under `artifacts/aiforge/`, Phase C validation score arrays, Phase D source-error and paired-delta JSON, and Phase E comparator measurements under `artifacts/phase-e/raw/`. The release model SHA-256 is `632770846fc6a0258f680c4ac406940a0b1251f6cc14ca43f1fa647854d1ead4`. The full RGB seed-42 checkpoint SHA-256 is `6556696ddb2a2aa256912afa433a9fa71b0a52d32a1338a5d71329e9cef57bcc`.

## Executed evidence

| Experiment | Status | Finding |
| --- | --- | --- |
| Full MobileNetV3, 16-group fit pilot | Completed | Loss fell from 0.974 to 0.859 in two epochs. 140 trainable tensors updated. This only establishes that the training path executes. |
| Full RGB MobileNetV3, seed 42 | Completed | Development AUROC 0.5107 [0.5048, 0.5190], AP 0.5126, highest-tile edit IoU 0.0134, and edit hit rate 14%. |
| Frozen RGB MobileNetV3, seed 42 | Completed | Development AUROC 0.5054 and AP 0.5043. |
| Partial RGB MobileNetV3, seed 42 | Completed | Development AUROC 0.5134, AP 0.5158, highest-tile edit IoU 0.0228, and edit hit rate 18.5%. |
| Full residual-enhanced MobileNetV3, seed 42 | Completed | Development AUROC 0.5126 and AP 0.5112. |
| Whole-image MobileNetV3, seed 42 | Completed | Development AUROC 0.5002 [0.4969, 0.5037], AP 0.5056. |
| Nested learning curve, full RGB seed 42 | Completed | 139 / 278 / 556 fitting groups reached 0.5009 / 0.5052 / 0.5107 AUROC. |
| Symmetric perturbation screen, full RGB seed 42 | Completed | Same 24 development groups and frozen calibration across 12 conditions. Clean AUROC 0.5139 and combined AUROC 0.5165. |
| FP32 and oneDNN INT8 | Completed | Development AUROC 0.5107 and 0.5009. Quantized convolution and linear operators were present. |
| Matched PatchNet distillation control | Completed | Hard-label student 0.5062 AUROC. Distilled student 0.5017. Distillation delta −0.0046 [−0.0122, 0.0021]. |
| TruFor official inference | CPU panel verified | A fresh run reproduced all 80 official panel document scores within 0.00000036 and all highest-scoring tile boxes. See the independent audit for full-cohort status. |
| Matched AIForge cross-generator evaluation | Completed, retrospective | 179 matched edit specs in 178 source groups. AUROC was near 0.5 for both versions. Paired intervals included zero in both historical splits. |
| External single-image confirmation | Pending authorized data | The [fresh shift evidence status](FRESH_SHIFT_EVALUATION.md) records the intake requirements. The local XFUND form probe measures a same-release category and corpus shift under the frozen receipt policy. |
| Physical trusted-reference acquisition | Blocked | No rights-cleared set with recaptures, localized edits, legitimate revisions, wrong references, and device/session metadata was available. The existing Phase F evaluation contract and collection protocol are ready. |
| DCT, ELA and additional branches | Unexecuted | The matched controls showed little practical separation. Applying further complexity would precede comparison with an official external baseline. |

### Three-seed development comparison

| Recipe | AUROC, seeds 42 / 17 / 71 | Mean AUROC ± seed SD | Mean AP |
| --- | ---: | ---: | ---: |
| Whole-image RGB | 0.5002 / 0.5005 / 0.4990 | 0.4999 ± 0.0008 | 0.5058 |
| Native RGB, frozen encoder | 0.5054 / 0.5025 / 0.5060 | 0.5046 ± 0.0018 | 0.5030 |
| Native RGB, partial fine-tuning | 0.5134 / 0.5127 / 0.5106 | 0.5123 ± 0.0014 | 0.5140 |
| Native RGB, full fine-tuning | 0.5107 / 0.5265 / 0.5218 | 0.5197 ± 0.0081 | 0.5206 |
| Native RGB with residual enhancement, full fine-tuning | 0.5126 / 0.5318 / 0.5208 | 0.5218 ± 0.0096 | 0.5194 |

The predeclared AUROC ranking selected residual enhancement for the locked holdout probes at seed 42. Its paired AUROC differences against full RGB were +0.0019 [−0.0021, +0.0059], +0.0054 [−0.0004, +0.0128], and −0.0010 [−0.0069, +0.0050] across the three seeds. The comparison provides limited evidence of residual enhancement's incremental value. The complete matrix and per-seed source-group and paired intervals are in `artifacts/receipt-study-v1/`.

The selected seed had 10.5% FPR and 12.0% recall on development at the nominal 5% calibration cutoff. Its highest tile intersected the edit on 14.5% of edited documents, with mean IoU 0.0165. Its private case review includes five high-scoring genuine documents, five low-scoring edits, and three edits both localized and flagged. See `artifacts/receipt-figures-residual-v1/`.

### Frozen holdout probes

| Cohort | Source groups | AUROC [95% interval] | AP | Highest-tile edit IoU | Observed FPR / recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical CORD/SROIE test | 219 | 0.5134 [0.5072, 0.5217] | 0.5291 | 0.0195 | 9.5% / 13.1% |
| Historical WildReceipt corpus shift | 1,293 | 0.5015 [0.5003, 0.5029] | 0.5033 | 0.0094 | 2.4% / 2.9% |
| Held-back XFUND forms, corpus and category shift | 410 | 0.5007 [0.4998, 0.5019] | 0.5003 | 0.0011 | 87.6% / 87.8% |

The reported operating point targets 5% FPR on receipt calibration. The selected residual seed-42 checkpoint SHA-256 is `2076ae044aaa65e69df5ae941a71ecf64b397c241cd0065ff3fb3533291ff6c5`. The historical matching procedure and the complete frozen configuration are recorded in `artifacts/receipt-retrospective-v1/`.

The XFUND audit retained all 418 previously excluded form pairs in 410 groups, with nine perceptual alias edges within forms and no candidate links to the audited receipt cohort. All of the checked RGB differences were inside the provided masks. The median mask area was 0.042% of form pixels. This cohort shares the GPT-Image-2 generator and release with the receipt benchmarks while changing both corpus and document category. Fitted Brier score was 0.25085 versus 0.25000 for constant prevalence. The frozen receipt cutoff flagged 366 of 418 authentic forms and 367 of 418 edits. Form scoring retained 144 completed serial predictions and distributed the remaining 692 across three CPU workers with the same checkpoint and tile settings. The merge verified one prediction per audited image. The audit, pre-label lock, score table and evaluation are in `artifacts/receipt-xfund-v1/`.

For the full RGB run, a cutoff fitted to nominal 5% calibration FPR yielded 12.0% observed FPR and 12.5% recall in development. At calibrated probability 0.5, the confusion matrix was `[[96, 104], [94, 106]]` and macro F1 was 0.505. Fitted Brier score was 0.24994 and ECE was 0.00897, versus a constant predictor with Brier 0.25000 and ECE 0. At a 0.55 confidence cutoff, both classes had zero coverage. The observed gains over prevalence are too small to support a dependable document decision. [Single-image model card](MODEL_CARD.md)

The seed-42 case review examined the five highest-scoring authentic documents and five lowest-scoring edits. For example, authentic SROIE case `000003626` scored 1.847, above the calibrated operating threshold of 1.487. The highest tile missed both CORD edits `000000215` and `000000042`, whose masks covered 0.554% and 0.385% of the document. CORD case `000000383` illustrates a different failure. The highest tile did intersect the edit, while its document score 0.379 stayed far below the operating threshold. The private numerical case review and reliability/risk plots are in `artifacts/receipt-figures-v1/`.

[Development reliability diagram](figures/receipt-v1-reliability.png) · [Risk–coverage diagram](figures/receipt-v1-risk-coverage.png)

### Learning curve

| Fitting groups | Supervised examples | Development AUROC | 95% source bootstrap interval |
| ---: | ---: | ---: | ---: |
| 139 | 426 | 0.5009 | [0.4982, 0.5040] |
| 278 | 873 | 0.5052 | [0.5013, 0.5095] |
| 556 | 1,755 | 0.5107 | [0.5048, 0.5190] |

The subsets nest within one another. The fixed three-epoch schedule gives larger subsets more optimizer updates, so the curve estimates the effect of using more sources under that schedule. The whole-image control differs in scale and document supervision and receives no off-edit patch negatives. [Learning curve](figures/receipt-v1-learning-curve.png)

## Acquisition-independent perturbations

| Condition | AUROC, mild / stronger | Recall and FPR at the stronger level |
| --- | ---: | ---: |
| JPEG | 0.502 / 0.485 | 8.3% recall, 8.3% FPR |
| Blur | 0.509 / 0.506 | 16.7% recall, 16.7% FPR |
| Downscale and restore | 0.505 / 0.510 | 29.2% recall, 25.0% FPR |
| Brightness reduction | 0.508 / 0.498 | 0% recall, 0% FPR |
| Gaussian noise | 0.502 / 0.509 | 0% recall, 0% FPR |
| Clean control | 0.514 | 8.3% recall, 8.3% FPR |
| Combined disturbances | 0.516 | 4.2% recall, 4.2% FPR |

The score threshold was fixed from the clean calibration pool. Each group contributes a genuine image and an edit. Twenty-four images per class result in 4.17 percentage points per decision. Physical recapture and novel editing models require separate evaluations. Per-condition results and source-group intervals are in `artifacts/receipt-stress-v1/full-42/`.

## Efficiency on CPU

| Configuration | State dictionary | AUROC / AP | Full-document median / p95 |
| --- | ---: | ---: | ---: |
| MobileNetV3-Small full RGB, FP32 | 3,840,215 bytes | 0.5107 / 0.5126 | 153.0 / 1,209.5 ms |
| Same model, oneDNN INT8 | 1,210,551 bytes | 0.5009 / 0.5019 | 150.9 / 1,395.0 ms |
| PatchNet hard-label student | 56,981 bytes | 0.5062 / 0.5033 | See per-document scores |
| Identical PatchNet with distillation | 56,981 bytes | 0.5017 / 0.5025 | See per-document scores |

The MobileNet comparison covers 400 documents for accuracy and interleaved timing on 32 documents, repeated twice with eight warm-up documents per model. It uses a two-thread CPU setting and tile batch size 64. The timed path reads the file, decodes and orients it, constructs tiles, processes the network and serializes score and region. Both models use the same timing loop while other experiments were running. The small median difference and larger INT8 p95 provide no evidence of a meaningful speed gain. The historical FP16-labelled artifact compressed weights in storage. This INT8 graph executes quantized operators. See `artifacts/receipt-study-v1/efficiency-full-42.json` and `artifacts/receipt-distill-v1/`.

| Exported CPU runtime | TorchScript file | Peak process working set | Median / p95 per document |
| --- | ---: | ---: | ---: |
| FP32 | 4,095,430 bytes | 676,872,192 bytes | 367.1 / 3,041.3 ms |
| INT8 | 1,452,930 bytes | 681,046,016 bytes | 281.2 / 1,798.9 ms |

Both exports passed score parity checks on three calibration batches. Separate processes ran the same 32-document schedule with eight warm-up documents and two repetitions. Peak working set includes the interpreter, framework, image processing, and inference. Other research activity on the host and sequential runtime launches limit timing interpretation. The serialized model shrank while measured process peak stayed approximately 650 MiB. Export hashes and measurement settings are in `artifacts/receipt-runtime-v1/runtime.json`.

The independent 400-image replay reproduced the export AUROCs of 0.51070 and 0.50093, with a paired INT8 minus FP32 delta of −0.00978 [−0.01720, −0.00244]. At the same frozen FP32 cutoff, the exports flagged 49 and zero images respectively. Highest-scoring tile boxes agreed on 52 of 400 images. The eager/export parity within each precision does not extend to interchangeability between precisions.

## Application decision

The v2 receipt models are private research configurations. Current measured performance does not warrant application integration. Known-specimen digest gating in the historical demo recognizes designated fixture inputs. It provides a demonstration constraint, with no learned distribution-shift detection. In trusted-reference mode, an accepted alignment measures visible change within overlap and does not establish document identity or authorization.

Targeted validation passed 48 tests, including four tests for the new research code and the computational reference suite using the OpenCV package retained in the Phase E workspace. A direct smoke check processed the released fictional specimen through the app's analysis code, verified abstention on an unrelated image, and checked that comparison routes are unavailable in the public configuration. The HTTP test harness stalled in Windows `socketpair`. The Docker daemon was unavailable on this host.

Pseudonymized prediction exports contain 16,256 rows across the MobileNet, perturbation and holdout runs in `artifacts/receipt-share-v1/index.json`, plus 1,380 student rows in `artifacts/receipt-share-distill-v1/index.json`. The manifests and per-run results support independent recomputation. Reference verification still requires independently collected pairs.

## External comparison readiness

### Matched AIForge v1 and v2 receipts

Of 1,557 historical v2 candidates, 179 passed the local crosswalk. The audit excluded 314 without a matching v1 authentic filename, 1,058 with differing decoded originals, and six with release metadata disagreements. All accepted specs had identical decoded originals and masks across releases, identical field, text and bounding-box metadata, and changed pixels confined to the mask in both versions. The matched set contained 167 Gemini edits and 12 Ideogram edits in v1, each paired with a GPT Image 2 edit in v2.

| Historical cohort | Matched specs / source groups | v1 AUROC | v2 AUROC | Paired AUROC delta, v2 minus v1 [95% interval] | v1 / v2 recall | Shared FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CORD/SROIE | 136 / 135 | 0.5065 | 0.5133 | +0.0068 [−0.0019, +0.0167] | 11.8% / 14.7% | 11.0% |
| WildReceipt | 43 / 43 | 0.4941 | 0.4941 | 0.0000 [−0.0130, +0.0130] | 2.3% / 2.3% | 2.3% |

The frozen residual MobileNetV3 seed-42 checkpoint and the nominal 5% calibration cutoff were used for both arms. Genuine control logits agreed exactly between versions. The v2 logits also matched the historical run exactly. Intervals use 500 source-group bootstrap draws. The exact-match subset and small Ideogram sample limit generalization. The comparison measures the two released edit pipelines on this historical subset, with potential release-level differences in edit generation and processing.

The local audit, manifests, score tables, paired intervals and input hashes are in `artifacts/receipt-generator-evaluation-v1/`.

The official TruFor archive matched the published MD5, and the project checkpoint was byte-identical to its archived member. The pinned inference entry point produced both smoke outputs and 80 native-resolution CPU panel outputs in 9 minutes 38 seconds. The panel caps image area and contains 10 genuine/edit source groups in each of four roles. Calibration was fitted on its own panel role.

| Evaluation role | TruFor AUROC [95% interval] | Residual AUROC | Paired AUROC difference [95% interval] |
| --- | ---: | ---: | ---: |
| CORD/SROIE development | 0.480 [0.345, 0.590] | 0.525 | −0.045 [−0.208, +0.075] |
| CORD/SROIE historical test | 0.380 [0.240, 0.490] | 0.485 | −0.105 [−0.240, +0.040] |
| WildReceipt historical shift | 0.830 [0.710, 0.980] | 0.475 | +0.355 [+0.257, +0.540] |

The WildReceipt panel ranking performance should be read alongside operating behavior. At the receipt calibration cutoff nominally targeting 5% FPR, TruFor flagged zero images in all three evaluation roles. Both the small sample and native-size caps limit transfer of these results to the full cohorts. The WildReceipt panel shares the v2 generator with the other panels. The selection manifest, paired scores and bootstrap results are in `artifacts/trufor-native-panel-v1/`. Source and dependency digests are in `artifacts/trufor-upstream/provenance.json`.
