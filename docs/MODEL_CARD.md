# ForgeryLens model card

## AIForge v2 single-image receipt detector

**Use.** Local course research on detecting localized visual edits from one receipt image. The demo illustrates routing for university reimbursement review. Actual claims require standard manual review. A high score is a cue to inspect the receipt and confirm it against transaction records before any escalation.

**Model and data.** Run `receipt-study-v1/residual-42` is the residual-enhanced MobileNetV3-Small, seed 42, with **927,585 parameters** and checkpoint SHA-256 `2076ae044aaa65e69df5ae941a71ecf64b397c241cd0065ff3fb3533291ff6c5`. It starts from TorchVision ImageNet V1 weights and scores native 112 px receipt tiles. The document logit averages the three highest tile logits. Fitting uses mask-supervised patch labels. Inference takes only the receipt image. The highest-scoring tile is a location cue with limited verified overlap. The contained AIForge v2.0.1 manifest fits on **585 receipt pairs in 556 source groups**, with **145 pairs in 141 groups** reserved for calibration. The dataset balances original and edited images. Dataset lineage, shift limitations, and weight terms are recorded in the [study protocol](SINGLE_IMAGE_PROTOCOL.md) and [reproduction guide](SINGLE_IMAGE_REPRODUCTION.md). Run evidence is in `artifacts/receipt-study-v1/residual-42/fit.json` and `artifacts/receipt-protocol-v1-contained/audit.json`.

**Policy.** The local demo uses the exact Platt slope and intercept in `artifacts/receipt-retrospective-v1/freeze.json`. The priority-review probe uses `logit > 1.4312477111806406`, the frozen cutoff calibrated to a nominal 5% false-positive rate. Demo policy `receipt-preview-v1` uses a 0.55 confidence gate, where confidence is `max(p, 1 − p)`. It previews **Prioritize review** above the frozen logit cutoff when confidence passes, **Normal queue** when confidence passes and `p < 0.5`, and **Inconclusive** otherwise. Malformed inputs and scoring errors are inconclusive. The gate is a fixed research demonstration. Broad open-set rejection and deployment readiness remain unvalidated.

### In-domain results

| Cohort and run | Images | AUROC | Macro F1 at p ≥ 0.5 | ECE | Recall / FPR at frozen logit cutoff | Coverage at 0.55 confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CORD/SROIE development, `receipt-study-v1/residual-42` | 400 | 0.5126 | 0.5048 | 0.0080 | 12.0% / 10.5% | 0% |
| CORD/SROIE retrospective test, `receipt-retrospective-v1` | 442 | 0.5134 | 0.5063 | 0.0029 | 13.1% / 9.5% | 0% |

Evidence is in `artifacts/receipt-study-v1/residual-42/metrics.json` and `artifacts/receipt-retrospective-v1/test.json`.

### Shift results

| Cohort and run | Images | AUROC | Macro F1 at p ≥ 0.5 | ECE | Recall / FPR at frozen logit cutoff | Coverage at 0.55 confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WildReceipt retrospective corpus shift, `receipt-retrospective-v1` | 2,672 | 0.5015 | 0.4863 | 0.0106 | 2.9% / 2.4% | 0.45% |
| XFUND held-back forms, `receipt-xfund-v1` | 836 | 0.5007 | 0.3333 | 0.0284 | 87.8% / 87.6% | 0.24% |

Evidence is in `artifacts/receipt-retrospective-v1/shifted_test.json` and `artifacts/receipt-xfund-v1/metrics.json`. Recall and FPR in both tables are measured before confidence gating. Coverage is the share passing the standalone confidence gate. Accepted error was **50%** on each shifted cohort and is undefined for in-domain cohorts at zero coverage. Both shift cohorts use the same AIForge v2 edit generator as the receipt study. The [fresh evaluation](FRESH_SHIFT_EVALUATION.md) awaits an authorized source-disjoint cohort.

**Stress and open set.** In run `receipt-residual-stress-v1`, the frozen cutoff flagged **25.00% of genuine and edited receipts under strong blur**, up from **8.33% of each class** clean, and flagged neither class under strong low light. The demo routed **0% of either class** across all 11 conditions on the 24-pair sample and inappropriately routed **0 of 13** exploratory open-set inputs. The clean deferral rate limits what the open-set result establishes. [Paired stress and open-set results](RECEIPT_STRESS.md)

**Attribution.** In run `receipt-explain-v1`, highlight deletion yielded a lower logit than matched off-mask deletion on **48 of 48** development documents. Compared with the clean score, highlight deletion lowered the logit on **46 of 48**. Ordinary full-receipt pooling scored the edited receipt higher in **7 of 24** matched pairs. The five case probes show how a score-influencing highlight can miss an edit or support a false flag. [Attribution and failure cases](RECEIPT_EXPLAINABILITY.md)

**Limitations and consequences.** The constant-prevalence predictor had a better Brier score on both shift cohorts. Low ECE alongside near-chance ranking and very low coverage gives weak evidence for practical routing. False suspicion can delay a legitimate reimbursement or trigger unnecessary scrutiny. Missed edits can postpone detection of an incorrect payment. Reviewers should corroborate visual cues with authorized records, document their rationale before escalation, and give claimants a way to correct or clarify the receipt.

**Privacy.** The launch command in the [demo guide](DEMO_GUIDE.md#frozen-receipt-detector) binds to loopback and disables usage statistics. The demo processes uploads and previews in session memory and contains no upload persistence or export path. Use only authorized images and close the local session after review. Checkpoint weights and private validation images are kept local.

### Frozen routing replay and follow-up

The full preview on historical WildReceipt prioritized zero, sent six genuine and six edited receipts to the normal queue, and deferred the other 2,660. Its priority recall was zero. At the binary raw cutoff alone the model found 39 edits and flagged 32 genuine receipts. In the complete development pooling audit, restricting inference to annotated edit sites ranked the edit higher in 144/200 pairs; ordinary pooling did so in 43/200. The mask-assisted result describes a diagnostic intervention. The matched development ranking-loss pilot supplied no basis to change the deployed checkpoint or gate. See the [receipt study](SINGLE_IMAGE_RESULTS.md) and [course paper](ForgeryLens_Technical_Report.pdf).

## Historical fictional-specimen release

**Release scope** Synthetic receipt research and software demonstration. The factory loads the validation-selected model from `release/model.npz`. Measured classification, calibration, localization, model size, cost and held-out shift results are stored under `release/`.

**Input and output** The model converts an EXIF-oriented document image to grayscale and resizes it to 112 × 160 pixels. It produces a document logit and coarse evidence map. Candidate regions are normalized to the original displayed image. Confidence uses validation-fitted affine logit calibration, and uncertainty is twice the smaller binary class probability.

**Training and evidence** Fictional source documents were paired with localized edit variants and masks. Source groups were assigned independently to training, validation, clean test and shifted test. The synthetic training edit changes the local tonal appearance and redraws a fictional amount. The shifted operator modifies quality around existing text. This operator shift is materially narrower than a real generator shift.

**Operating point** The temperature and acceptance threshold are fitted on validation only. The packaged synthetic model scores the normalized demo specimens. It abstains on arbitrary uploads and on the known low-quality specimen. A model trained with an independently authorized manifest can be exported through the same interface. Verify its scope and calibration before use.

**Failure modes and usage limits** Synthetic results do not establish skill on scans, camera captures, real receipts, or new editing systems. Resizing can suppress very small edits. Candidate regions reflect model activations and can miss or overestimate affected areas. This tool cannot establish authenticity, intent, identity, or transaction validity.

**Data handling** The shipped service processes encoded images in memory and retains no upload history. Hosting providers may collect routine request metadata. No external documents were bundled. The known demo specimens are fictional.

**Digest gating** The bundled model recognizes an allowlist of fictional specimens by digest. This is a fixture restriction. It provides no learned estimate of how unfamiliar an upload is.

**Attribution** Training and evaluation source and measurements are included. AI assistance generated the implementation and report from measured outputs.
