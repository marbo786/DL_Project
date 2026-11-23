# ForgeryLens project status

| Area | Status |
| --- | --- |
| Studio and research explorer | Implemented with editable document viewport, real predictions on curated fictional demo specimens, heatmaps, and measured comparison tables and charts. |
| Research release | Trained a baseline, local detector, four controlled variants, and a distilled student. Ran three seeds for each main variant, measured a reduced-precision storage comparison, calibrated on validation, and evaluated clean and shifted synthetic holdouts and image stress tests. |
| Observed performance | Selected model clean macro F1 0.733, recall 0.500 and ROC-AUC 0.794. Shifted macro F1 0.333, recall 0.000 and ROC-AUC 0.512. These are synthetic results. |
| Serving model | The pickle-free NumPy release matched PyTorch at 80 additional inputs. Median warm batch-one CPU forward pass was 2.81 ms. Demo specimens are selected independently of test sets and illustrate model behavior. Unknown user images return inconclusive. |
| Verification | Seventeen tests passed with an in-process ASGI test runner adapted to the workstation's socket restrictions. Coverage includes model package loading, image analysis, artifact availability, result schema, OOD abstention, and manifest deduplication. |
| Public deployment | [Live Forensic Studio](https://forgerylens.onrender.com/studio) serves the Docker deployment on Render's free instance from `main`. The deployed application commit is `b913eeca97a5ab2e78c6dc7d34a79647ae5295ef`. Browser verification covered the original, edited, and low-quality demo predictions, the heatmap control, and the measured results in the research explorer. |
| External benchmark | The AIForge-Doc v2.0.1 receipt experiment is complete. After removing 31 duplicate source groups, 2,617 pairs passed the manifest audit. The validation-selected model reached ROC-AUC 0.500 on CORD and SROIE and 0.504 on WildReceipt. Mean mask IoU was 0.022 and 0.026, respectively. See the [benchmark results](ForgeryLens_AIForge_Results.md). |
| Native patch evidence | Tight oracle crops averaged 0.574 AUROC across three seeds. Fixed image-only tiling reached 0.520 AUROC with max pooling in a 64-pair validation screen. See the [Phase B analysis](PATCH_EVIDENCE_ANALYSIS.md). |
| Off-mask negatives | Random negatives improved top-tile localization across all three seeds. Mined negatives reached 0.521 top-3-pooled receipt AUROC, close to the 0.520 historical max result. See the [Phase C analysis](HARD_NEGATIVE_ANALYSIS.md). |

See the technical report, model card, and release JSON for complete experiment details.
