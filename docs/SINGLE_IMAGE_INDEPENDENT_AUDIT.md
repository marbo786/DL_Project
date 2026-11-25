# Independent ForgeryLens verification

The continuation starts from local research commit `613a4ac`, which descends from the requested `8351eef` and adds a pinned TruFor native CPU panel. The earlier v1/v2 runs and all other historical inputs were preserved. New private outputs are under `artifacts/receipt-verification-v1/`. No study data, private scores or source identifiers are included in this report.

## Audit ledger

| Check | Status | Evidence |
| --- | --- | --- |
| Checkpoint and training integrity | Verified with saved-state inspection and a fresh diagnostic | All 15 main fits and two learning-curve fits matched their recorded checkpoint and initial-weight digests. Full-population preprocessing was reconstructed for the main fits. Fresh gradient and overfit checks used eight training groups. |
| Model and operating policy selection | Reconstructed | The protocol specifies a fixed final epoch, mean development AUROC across three seeds, AP tie-break, then seed 42 for historical probes. The checkpoint, calibration, matrix and contained-manifest hashes matched the lock. |
| Receipt and XFUND point estimates | Independently recomputed | An independent tie-aware weighted-rank implementation matched every matrix AUROC and AP, checked prediction coverage and labels against the manifests, and reproduced the historical operating counts. |
| Source-group intervals | Independently recomputed | Weighted group draws retain repeated groups and use identical draws for paired differences. Calibration-resampled threshold intervals are also recorded. |
| XFUND partition | Independently rerun | The form audit rebuilt a byte-identical manifest with 418 pairs in 410 groups, nine within-form perceptual alias edges, and zero receipt-linked pairs under its exact and perceptual rules. The form lock uses the receipt fit and calibration with the same editing generator. The benchmark changed corpus and document category together. |
| FP32 and INT8 export parity | Independently verified and rescored | Each TorchScript export matched its reconstructed eager model on eight images. The full 400-image development replay showed a material quantization effect on score, decision and localization parity. |
| Distillation control | Independently recomputed | Same compact architecture and held-out development population. The paired AUROC effect was small and uncertain. |
| V1/v2 same-spec comparison | Retrospective, independently recomputed | 179 matched edit specs in 178 historical held-out source groups. The independently calculated paired intervals still include zero in both cohorts. |
| TruFor | Pinned checkpoint and historical panel metrics verified | The independent runner checks source blob and checkpoint hashes and uses the official detection head and native-resolution class-1 map. Full-cohort status is tracked below. |
| Fresh confirmation and physical acquisition | Open | A fresh rights-cleared cohort with verified source identity and the prescribed capture protocol is still needed. |

The attached patch reverse-applies to the local research index. The working-tree copy has later result and reproduction edits for the completed v1/v2 comparison. Those edits are included on this continuation. The prior run records provide evidence about experiment order through saved locks and configuration, while independent proof of all prior human label exposure is unavailable. Historical receipt holdouts and the v1/v2 subset remain retrospective. Development and its perturbation subsets retain their model-selection exposure. XFUND is a previously excluded cohort from the same release with inspected labels and scores now. A new partition of those receipts or forms would inherit this exposure.

The independent checks initially stopped on an incorrectly interpreted learning-curve run name, an overstrict assumption that AdamW tensor updates equal the nonzero-gradient set, and an XFUND audit invocation with a relative package root. The corrected invocations and checks completed. These issues were in the verification harness or its inputs. No evidence required changing the original fit or the frozen receipt policy.

## Independent results

| Frozen residual seed-42 cohort | AUROC [95% interval] | AP | FPR [95% interval] | Recall [95% interval] | Within-spec win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical CORD/SROIE | 0.5134 [0.5068, 0.5220] | 0.5291 | 9.5% [5.9%, 13.2%] | 13.1% [9.0%, 17.7%] | 57.9% |
| Historical WildReceipt | 0.5015 [0.5002, 0.5029] | 0.5033 | 2.4% [1.6%, 3.4%] | 2.9% [2.1%, 3.9%] | 51.7% |
| Previously excluded XFUND | 0.5007 [0.4998, 0.5019] | 0.5003 | 87.6% [84.3%, 90.7%] | 87.8% [84.7%, 91.0%] | 50.5% |

Within-spec win rate gives half credit to a score tie. Document AUROC compares every positive with every negative, while the within-spec metric compares only the two images in each edit spec. In both the synthetic regression case and real results the metrics differ. In the closely matched cohorts, unchanged content makes genuine and edited scores within a source highly dependent. Whole-group resampling retains that dependence and reproduced the narrow intervals. The intervals condition on the observed source-group definitions and fitted model. They do not capture uncertainty from training a new model or from sampling a different corpus.

The strict cutoff was `logit > 1.4312477111816406`. None of these cohorts had a tie exactly at the cutoff. The reconstructed XFUND confusion matrix is `[[52, 366], [51, 367]]`, with genuine and edited rows. Resampling calibration groups as well as evaluation groups widened the XFUND intervals to 80.5–95.3% FPR and 80.5–95.4% recall. This sensitivity analysis refits the cutoff from each resampled calibration pool and keeps weights fixed.

The best development mean AUROC was 0.521775 for the residual recipe, compared with 0.519658 for full RGB. The residual seed SD was 0.00964. The independently paired differences for seeds 42, 17 and 71 were +0.00194 [−0.00181, +0.00625], +0.00536 [−0.00095, +0.01234] and −0.00095 [−0.00724, +0.00494]. Seed variability and source-sampling intervals answer different questions. The learning curve recovered 0.50091, 0.50519 and 0.51070 AUROC at the three fit fractions. Distillation changed AUROC from 0.50624 to 0.50168, with paired delta −0.00456 [−0.01208, +0.00250].

| Cohort | Platt Brier | Platt ECE | Constant Brier | Constant ECE |
| --- | ---: | ---: | ---: | ---: |
| CORD/SROIE | 0.24985 | 0.00288 | 0.25000 | 0.00000 |
| WildReceipt | 0.25027 | 0.01057 | 0.25000 | 0.00000 |
| XFUND | 0.25085 | 0.02840 | 0.25000 | 0.00000 |

All three evaluation cohorts have balanced labels by design. The 0.5 constant therefore attains zero ECE, half error at full coverage, and no accepted cases above confidence 0.5. At a confidence cutoff of 0.55, the model accepted 0% of CORD/SROIE, 0.45% of WildReceipt, and 0.24% of XFUND. Half of the accepted cases in the latter two cohorts were errors. These statistics and AUROC give no useful operational evidence for the frozen detector.

## Training integrity

The training code initializes MobileNetV3-Small from the pinned ImageNet V1 state. The classifier head and trainable encoder tensors use AdamW at fixed learning rates 0.001 and 0.00015 respectively. The run uses positive-weighted BCE with logits, fixed mask-guided patch construction, no random image augmentations, and an image-only inference grid. Frozen blocks remain in evaluation mode while fitting. The inference path uses evaluation mode and the same RGB mapping and ImageNet normalization, plus the configured residual transform or whole-image resize. Saved encoder comparisons confirm that frozen blocks retained their initial tensors and fine-tuning updated the intended blocks. The selected residual run regenerated 1,755 tiles with 585 positives from 556 groups.

| Fresh backward pass | Optimizer tensors | Nonzero gradient tensors | Updated encoder tensors | Changed BN running buffers |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 2 | 2 | 0 | 0 |
| Partial | 31 | 31 | 29 | 14 |
| Full | 140 | 134 | 138 | 68 |

AdamW weight decay can update a tensor with zero gradient, so changed-parameter counts and nonzero-gradient counts need not coincide. In a 40-step, 24-tile diagnostic using eight training groups, the actual-mask loss fell from 1.0251 to 0.0068. In evaluation mode, balanced accuracy on those same tiles was 62.5%. The obvious-edit control fell from 1.0310 to 0.0053 and reached 100% balanced accuracy. The shuffled-label control fell from 0.9095 to 0.0043 and reached 78.1% balanced accuracy on its own shuffled examples. This verifies that gradients and fitting work and illustrates train-mode memorization and the effect of switching normalization to evaluation mode on a tiny batch. It provides no evidence of held-out generalization.

The historical efficiency report compares FP32 and INT8 on 400 development images. The fresh eager-versus-export audit found zero differences in logits and probabilities and eight of eight decision and localization matches within each precision arm. A second run rescored all 400 development images directly with the two exports. AUROC moved from 0.51070 to 0.50093, with paired INT8 minus FP32 delta −0.00978 [−0.01720, −0.00244]. The exported models agreed on 351 of 400 decisions and 52 of 400 highest-scoring tile boxes. At the frozen FP32 threshold, FP32 flagged 49 images and INT8 flagged zero. The maximum absolute document logit change was 1.977 and the maximum probability change under the same FP32 calibrator was 0.0870. This is evidence against swapping in the quantized export under the existing decision policy.

The public app instead resolves the released fictional-specimen artifact and uses a known-specimen digest gate. This is a demonstration constraint. The research model is not integrated into that path.

## External baseline and research decision

The pinned TruFor checkpoint comes from the official archive with listed MD5 `7bee48f3476c75616c3c5721ab256ff8`. The extracted checkpoint SHA-256 and pinned source revision are in the provenance record. Official input is RGB scaled by 1/256 and the model runs in evaluation mode. The document ranking uses the sigmoid detection head. Native map IoU at 0.5 and highest-mean-heatmap tile localization are separately identified aggregations. Possible pretraining provenance overlap beyond the upstream dataset descriptions remains unknown.

| Historical panel role | TruFor AUROC | Residual AUROC | Paired gain [95% interval] |
| --- | ---: | ---: | ---: |
| CORD/SROIE development | 0.480 | 0.525 | −0.045 [−0.195, +0.075] |
| CORD/SROIE historical | 0.380 | 0.485 | −0.105 [−0.235, +0.040] |
| WildReceipt historical | 0.830 | 0.475 | +0.355 [+0.255, +0.545] |

These are independent recomputations from the official checkpoint's existing predictions on the size-capped panel. The panel calibration threshold flagged zero documents across evaluation roles, and WildReceipt had ten groups. A fresh run reproduced all 80 document scores with maximum absolute difference 0.00000036 and all 80 highest-scoring tile boxes. An additional 569,502-pixel development-image replay matched the official dense map within the stated numeric tolerances. Median panel inference was 7.35 seconds and peak process working set was 2.44 GiB on this Windows CPU. Full native calibration has started. The full protocol has 3,804 receipt images and 836 XFUND images, with maxima of 12.2 and 8.9 megapixels respectively. The panel size caps were 0.25 and 0.60 megapixels. The runner retains per-image scores, localization summaries, time and peak working set. It validates completed roles against the source manifest and fits thresholds using only the complete calibration role.

The measured residual results do not justify deploying single-image detection. TruFor's small WildReceipt panel motivates testing whether its ranking advantage persists on the full corpus at a useful operating point. Further architecture work should depend on that result. A confirmed advantage would justify a controlled hypothesis about the benefit of confidence-weighted forensic noise features on WildReceipt. A weak full-cohort result would favor documenting the conditions in which the tested detectors struggle. The evidence here is limited to the evaluated models and datasets.
## Validation and remaining dependencies

The Windows suite passed 64 tests with the cached OpenCV and multipart dependencies on the Python path. An earlier default-environment run skipped three OpenCV-dependent modules. In-process API tests worked in this continuation. Docker client inspection found no Linux-engine named pipe, so this check did not build or run containers. The source-group statistics tests cover ties, repeated groups, paired sampling, and strict thresholds. The TruFor test compares the larger scaled dot-product attention path with explicit softmax attention.

Full TruFor scoring is resumable and was started on the Windows CPU. The run can be monitored through `artifacts/receipt-verification-v1/trufor-full-run.log` and its per-role CSVs. No full-cohort baseline estimate is available before the relevant inference and calibration tables complete. The fresh confirmatory and physical-acquisition studies still require newly collected, rights-cleared sources.
