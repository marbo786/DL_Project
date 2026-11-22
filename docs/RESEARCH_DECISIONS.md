# Team research decision log

Use one entry per substantive team decision. Record author, date, alternatives, rationale and evidence. Keep the final shifted set sealed during selection.

| Decision to record | Evidence to attach |
| --- | --- |
| Ranked stakeholder targets and cost of errors | Stakeholder rationale and operating constraints |
| Source-group identity and held-out generators | Dataset revisions, manifest, exclusions and leak audit |
| Baseline and proposed architecture | Team-authored model code, configurations and development runs |
| Four single-factor ablations | At least three seeds per configuration and seed-wise results |
| Two efficiency approaches | CPU timing protocol, model size and measured comparison |
| Calibration and rejection | Validation split, fitted parameters, reliability and risk-coverage plots |
| Robustness, OOD and explanations | Stress-test protocol, localized explanations and five failure cases |
| Final operating point | Frozen configuration, held-out results and model card |

AI assistance disclosure template. Identify the boilerplate modules used, describe your own contributions, and record how you verified and understood the generated infrastructure.

## Phase A diagnostic decision

The original tiny CNN failed to fit 48 training images under its existing combined objective and resize. Keep the existing checkpoint as the historical comparison. The next comparison isolates the classification and localization losses on the same source pairs, then changes one training factor at a time. Holdout evaluation and promotion to public inference remain gated on measured evidence of learning and useful validation performance. The baseline run was executed with AI assistance and is documented in [the failure analysis](MODEL_FAILURE_ANALYSIS.md).
