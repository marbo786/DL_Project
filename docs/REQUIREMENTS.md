# ForgeryLens requirements for university reimbursement review

## Decision and scope

This Track A study models a generic university finance office screening receipts attached to reimbursement claims. ForgeryLens takes one receipt image and estimates evidence of a localized visual edit. The decision is whether to move the claim earlier in the manual review queue. Finance staff still check the claim against receipts, approved revisions, and transaction records before deciding on payment. The scenario describes a proposed workflow. The university's actual edit prevalence, queue capacity, and error costs have not been measured.

| Outcome | Meaning | Reviewer action |
| --- | --- | --- |
| **Prioritize review** | A supported input passes the confidence gate and exceeds the review cutoff. | Move the claim earlier in the queue. Check the marked evidence against the receipt and transaction records before escalating. |
| **Normal queue** | A supported input passes the confidence gate and meets the policy's lower-risk criterion. | Follow the standard review checklist and timeline. |
| **Inconclusive** | Input quality, scope, or confidence prevents reliable routing. A deployment gate that has failed also produces this outcome. | Route through the standard manual process. Request a clearer receipt when needed and explain the reason for deferral. |

The current checkpoint is a research prototype. Its measured recall and confidence coverage do not support operational queue routing. In the research demo, the frozen policy and the reason for an inconclusive result must be visible. Score cutoffs and confidence gates require separate validation before use in the workflow.

## Stakeholders and ranked targets

| Stakeholder | Need |
| --- | --- |
| Finance reviewer | Find material edits early, see where to inspect, and keep priority review within available staff time. |
| Claimant | Receive timely, consistent review and a clear reason for a request to resubmit. Protect legitimate claims from unsupported escalation. |
| Audit and privacy owner | Trace routing to the checkpoint and policy, preserve human accountability, and minimize access to receipt images. |
| IT operator | Run reproducible batch-one inference on the target CPU, with a bounded model size and a working manual fallback. |

| Rank | Requirement | Measurement and stakeholder rationale |
| ---: | --- | --- |
| 1 | Edited-class recall **≥90%** | Measure prioritized edited receipts divided by all edited receipts, including any deferred cases, at the same policy used to measure false flags. Early review matters most to Finance because edits missed by triage can delay detection until manual reconciliation. |
| 2 | Observed FPR **≤5%** and **≤100 priority reviews per 1,000 receipts** | Track the fraction of genuine receipts prioritized and workload at a stated edit prevalence. These are proposed design budgets. Finance needs a manageable priority queue, and the claimant needs protection from unnecessary scrutiny and delay. |
| 3 | Shifted macro F1 **≥0.75** | Measure both classes at a frozen binary rule, and report the three-outcome routing coverage separately. The audit owner needs evidence that decisions transfer across receipt populations. |
| 4 | ECE **≤0.10** and explicit deferral | Report 10-bin ECE and Brier score alongside a calibration-prevalence baseline. For confidence gating, require a proposed minimum **50% coverage** and maximum **10% error among accepted** on the development cohort before considering use. Report class-specific coverage and errors. This gives the claimant a meaningful way to receive an inconclusive result and gives Audit a way to detect confidence collapse. |
| 5 | Batch-one CPU median **≤100 ms per receipt**, with p95 reported | Time receipt decode, preprocessing, tiling, inference, calibration, and decision output on the intended CPU. IT and Finance need to understand both typical turnaround and slow receipts. |
| 6 | Model size **≤25 million parameters** | Count all model parameters from the frozen checkpoint. IT needs a model that fits the intended resource budget. |

All claim dispositions require a human decision. A pilot must also demonstrate that every routing output records the checkpoint identifier, policy version, and reason, and that test uploads are absent from persistent storage and logs under the chosen demo configuration. These are proposed acceptance criteria for Audit and IT. End-to-end workflow compliance is pending.

Missed edits and false flags have different consequences. A missed edit can delay the detection of an incorrect payment. A false flag consumes reviewer time and can delay a genuine reimbursement. Deferral avoids an unsupported routing decision and transfers work to the manual queue. Lowering the review cutoff can catch more edits while increasing false flags. Tightening the confidence gate can increase deferral and reduce the number of edited receipts the system can prioritize. Native tiling preserves small fields while increasing latency, especially for large receipts. Actual financial costs require a pilot, so this spec uses observed misses, false flags, and workload as cost proxies.

## What calibration and development show

Run `receipt-study-v1/residual-42` uses residual MobileNetV3-Small, seed 42. The checkpoint SHA-256 is `2076ae044aaa65e69df5ae941a71ecf64b397c241cd0065ff3fb3533291ff6c5`. Platt calibration and all three raw-logit review cutoffs come from 290 calibration receipts in 141 groups. The table applies the saved strict `logit > cutoff` rule. The development cohort contains 400 receipts across 199 groups, with 200 of each class. Queue counts below describe a binary review-cutoff probe before any confidence gating.

| Nominal calibration FPR | Calibration flagged genuine / edited (145 each) | Development false flags / edits found (200 each) | Development FPR / recall | Development priority queue | Development edits missed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 1 / 1 | 3 / 5 | 1.5% / 2.5% | 8 / 400 (2.0%) | 195 / 200 |
| 5% | 7 / 7 | 21 / 24 | 10.5% / 12.0% | 45 / 400 (11.25%) | 176 / 200 |
| 10% | 14 / 15 | 35 / 37 | 17.5% / 18.5% | 72 / 400 (18.0%) | 163 / 200 |

At the nominal 5% cutoff, if **1%** of incoming receipts were edited and the development class-conditional rates transferred, a batch of 1,000 would yield **105.15 expected priority reviews**, including **103.95 false flags** and **1.20 edited receipts found**. **8.80** edited receipts would remain outside the priority queue. Expected priority reviews equal `1,000 × [(1 − prevalence) × FPR + prevalence × recall]`. The 1% prevalence is illustrative. The development evidence is too weak to support a deployed cutoff.

At the 0.55 confidence gate, development coverage is **0%** for both classes, so accepted error is undefined. At confidence 0.50, coverage is **100%** and accepted error is **49.5%**. These two diagnostic points fail the proposed coverage and error requirements. Development ECE is **0.0080**, while its Brier score **0.24990** is close to **0.25000** for the constant prevalence predictor. The queue table and confidence results evaluate separate rules. The [local research demo](DEMO_GUIDE.md#frozen-receipt-detector) combines the 0.55 confidence gate with the frozen priority-review cutoff for illustration. An operational policy remains pending.

The [fit metadata](../results/receipt-study-v1/residual-42/fit.json) and [metrics](../results/receipt-study-v1/residual-42/metrics.json) record the evidence. The private `calibration.csv` and `validation.csv` remain in the original run directory. [Metric implementation](../research/receipt/metrics.py)

## Target status and historical shift evidence

The following probes use previously saved outputs. Historical shift is descriptive evidence only. Thresholds, checkpoints, and policy choices use calibration and development data. For a fresh final shift evaluation, freeze the source manifest, checkpoint hashes, calibration, and policy in a lock file before a single scoring execution.

| Target | Status and evidence |
| --- | --- |
| Recall ≥90% and FPR ≤5% | **Missed jointly.** At the nominal 5% cutoff, historical WildReceipt produced **2.9% recall**, **2.4% FPR**, and **1,297 missed edits out of 1,336**. This binary probe categorized **100%** of receipts before abstention and prioritized **71 of 2,672 (2.66%)**. On the previously held-back XFUND forms, the same cutoff yielded **87.8% recall** with **87.6% FPR** and **100%** binary decision coverage. |
| ≤100 priority reviews per 1,000 | **Missed in the illustrative development scenario** at the nominal 5% cutoff, with **105.15** expected reviews per 1,000 at 1% assumed prevalence. Development FPR was **10.5%**, with **100%** coverage under the binary rule. Actual queue workload is pending. |
| Shifted macro F1 ≥0.75 | **Missed.** WildReceipt macro F1 was **0.4863** at probability 0.50, with **31.2% FPR**, **32.0% edited recall**, and **100%** binary decision coverage. At the frozen nominal 5% cutoff, macro F1 was **0.3589**. |
| ECE ≤0.10 and useful deferral | **ECE met, useful deferral unproven.** WildReceipt and XFUND ECE were **0.0106** and **0.0284**. On both cohorts, the constant 0.5 predictor had better Brier score. At confidence 0.55, WildReceipt covered **12 of 2,672 (0.45%)**, six genuine and six edited. All 12 were classified as genuine, yielding **50%** accepted error and **0%** FPR for this gate-plus-binary-classification probe. At confidence 0.50, coverage was **100%** and FPR was **31.2%**. |
| CPU median ≤100 ms | **Missed.** Run `receipt-pareto-v1` measured **156.46 ms median / 970.29 ms p95** for the selected residual checkpoint from file decode through JSON output on 400 development receipts. At the common 40-review budget, it achieved **10.5% recall** and **9.5% FPR**, with binary ranking applied to all inputs before abstention. [Benchmark protocol and Pareto figures](RECEIPT_PARETO.md) |
| ≤25 million parameters | **Met.** The residual checkpoint has **927,585** parameters. |

WildReceipt evidence is from run `receipt-retrospective-v1`, `results/receipt-retrospective-v1/freeze.json` and `shifted_test.json`. XFUND evidence is from run `receipt-xfund-v1`, `results/receipt-xfund-v1/freeze-before-labels.json` and `metrics.json`. Timing evidence is in `results/receipt-pareto-v1/summary.json`. The parameter count is from `results/receipt-study-v1/residual-42/fit.json`.

## Exact three-outcome preview replay

Counts below use the same frozen calibration and routing implementation as the Streamlit app. Each cell is **genuine / edited**. All cohorts have paired, balanced labels.

| Cohort | Priority review | Normal queue | Inconclusive |
| --- | ---: | ---: | ---: |
| Development | 0 / 0 | 0 / 0 | 200 / 200 |
| Historical ID | 0 / 0 | 0 / 0 | 221 / 221 |
| Historical WildReceipt | 0 / 0 | 6 / 6 | 1,330 / 1,330 |
| Held-back XFUND forms | 1 / 1 | 0 / 0 | 417 / 417 |

On WildReceipt, probability 0.50 flags 417 genuine and 427 edited receipts, the nominal 5% FPR cutoff flags 32 genuine and 39 edited receipts, and the full preview prioritizes zero. Recompute all three with the policy replay command in the reproduction guide. It verifies saved score digests and checks both binary confusion matrices against the recorded metrics.
