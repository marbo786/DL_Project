# Fresh shifted receipt evaluation

**Status as of 25 November 2026: pending.** The team confirmed that no new authorized original/edited receipt pairs are available. No new authorized, source-disjoint receipt cohort was identified for this workstream. The deliverable is an evidence status and an intake protocol. A fresh common-cohort evaluation of the whole-image CNN and residual detector is pending.

## Existing evidence

| Cohort | Evidence status | Scope |
| --- | --- | --- |
| CORD/SROIE, 442 images in 219 groups | Retrospective in-domain | Historical receipt test. Residual run `receipt-retrospective-v1`, `artifacts/receipt-retrospective-v1/test.json`. |
| WildReceipt, 2,672 images in 1,293 groups | Retrospective corpus shift | Historical receipt test from a different source corpus with the same AIForge v2 generator. Run `receipt-retrospective-v1`, `artifacts/receipt-retrospective-v1/shifted_test.json`. |
| XFUND, 836 images in 410 groups | Previously held-back form probe, now scored | Same-release category and corpus shift. The receipt recipe was locked before scoring these forms. Run `receipt-xfund-v1`, `artifacts/receipt-xfund-v1/freeze-before-labels.json` and `metrics.json`. |
| Matched AIForge v1/v2 edit specifications, 179 specs in 178 groups | Retrospective release and generator comparison | Historical same-source pairs selected by the existing crosswalk. Run `receipt-generator-evaluation-v1`, `artifacts/receipt-generator-evaluation-v1/matched-pairs.csv` and `summary.json`. |

Fresh evaluation requires new source lineages. Repartitioning these exposed documents would carry forward their prior use. Existing locks document the scope and order of the earlier probes. The historical whole-image CNN benchmark used its own cohort and manifest. Comparative claims about the two frozen detectors on fresh data require common eligible inputs and the same evaluation protocol.

## Intake and one-time scoring protocol

1. **Define the shift before collection.** For a capture shift, use authorized new receipt originals, document device and session conditions, and capture each original and its edited counterpart under matched conditions. For a generator shift, use new authorized receipt sources and record the edit mechanism, target field, and generation recipe. Build a separate in-domain cohort from new receipt sources collected under development-like conditions. Keep both evaluation cohorts balanced by matching each genuine document to its edited counterpart.
2. **Review authorization and lineage.** Record owner or dataset terms for the experiment, access controls, and permission for any disclosures. Assign stable source IDs and keep all versions, edits, re-encodes, crops, and recaptures of each original in one lineage group. Compare the new inventory against all earlier AIForge v1 and v2 sources, including quarantined groups and the previously held-back forms. Join confirmed aliases transitively across releases and cohorts. Compare file and decoded pixel digests, screen perceptual candidates, and manually review unresolved source links before including any group. Keep identity crosswalks, images, and masks private.
3. **Prepare a blind manifest.** Record the file hashes, source groups, evaluation role, shift metadata, paired identity, and aligned binary edit masks. Record exclusions before scoring. Keep evaluation labels and masks separate from the scoring inputs until predictions are saved. Verify that all fitting, calibration, and development groups are disjoint from both evaluation cohorts.
4. **Freeze both detectors.** Select the simple whole-image CNN and the residual MobileNetV3 on permitted development evidence. Establish the baseline's fitting lineage and calibrate it on separate authorized calibration groups before evaluation. Write a lock with the manifest, checkpoint and calibration hashes, preprocessing, decision thresholds, rejection rules, metric definitions, exclusions, and scoring code revision. Record the locked experiment ID before opening either cohort for scoring.
5. **Score once and report together.** Score both models on the same locked in-domain and shifted images in one execution. Save predictions before opening labels or masks. Report macro F1, AUROC, edit recall, FPR, precision, class-specific coverage, and paired uncertainty by source group. Record any failed predictions and the predeclared handling rule. Report the observed cohort effects and preserve the locked results.

## Illustrative prevalence scenarios

Keep the balanced evaluations as the primary evidence. To show workload sensitivity, the table projects the saved development rates at the residual model's frozen logit cutoff, **12.0% recall and 10.5% FPR**, onto 1,000 hypothetical receipts at each prevalence. These rates come from `artifacts/receipt-study-v1/residual-42/metrics.json`, `operating_points.0.05`. The projection assumes those class-specific rates transfer and applies the binary cutoff before the confidence gate. The scenarios are arithmetic examples and provide no evidence about campus edit prevalence or future capture performance.

| Assumed edit prevalence | Priority reviews per 1,000 | Edited receipts prioritized | Genuine receipts prioritized | Edited receipts outside priority queue | Precision among priority reviews |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5% | 105.075 | 0.600 | 104.475 | 4.400 | 0.57% |
| 1% | 105.150 | 1.200 | 103.950 | 8.800 | 1.14% |
| 5% | 105.750 | 6.000 | 99.750 | 44.000 | 5.67% |

Priority reviews equal `1,000 × (prevalence × recall + (1 − prevalence) × FPR)`. Precision is the expected number of edited receipts prioritized divided by expected priority reviews. The low projected precision helps explain why the current checkpoint is used for research preview and actual claims follow the standard manual process.
