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

Controlled training on 24 AIForge receipt pairs identified width and update schedule as contributors to the original CNN's weak in-sample fit. The width-24, batch-8 recipe reached 0.82813 training AUROC, but validation AUROC stayed near 0.50 across width and batch controls. Retain the original benchmark as historical, and keep the wider configurations research-only. The next experiment targets local visual evidence with matched receipt patches and fixed validation tiling. The test partitions were excluded from model selection. AI assistance executed and analyzed the runs. [Measured controls and validation](MODEL_FAILURE_ANALYSIS.md)

## Phase B diagnostic decision

The crop study found modest held-out discrimination with known edit locations. Fixed tiling and trained pooling showed weak receipt-level transfer. Retain the patch experiments for research and prioritize same-source off-mask negatives before changing backbones. The final test partitions were excluded. [Measured results and AI-assistance attribution](PATCH_EVIDENCE_ANALYSIS.md)
