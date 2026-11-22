# ForgeryLens project status

| Area | Status |
| --- | --- |
| Studio and research explorer | Implemented with editable document viewport, real predictions on curated fictional demo specimens, heatmaps, and measured comparison tables and charts. |
| Research release | Trained a baseline, local detector, four controlled variants, and a distilled student. Ran three seeds for each main variant, measured a reduced-precision storage comparison, calibrated on validation, and evaluated clean and shifted synthetic holdouts and image stress tests. |
| Observed performance | Selected model clean macro F1 0.733, recall 0.500 and ROC-AUC 0.794. Shifted macro F1 0.333, recall 0.000 and ROC-AUC 0.512. These are synthetic results. |
| Serving model | The pickle-free NumPy release matched PyTorch at 80 additional inputs. Median warm batch-one CPU forward pass was 2.81 ms. Demo specimens are selected independently of test sets and illustrate model behavior. Unknown user images return inconclusive. |
| Verification | Sixteen tests passed with an in-process ASGI test runner adapted to the workstation's socket restrictions. Coverage includes model package loading, the genuine image analysis route, artifact availability, result schema, OOD abstention, and manifest rules. |
| Public deployment | A Dockerfile and Render Blueprint are supplied. Localhost and the Docker engine were unavailable in this workspace, Next.js package installation could not reach npm, and GitHub access for creating the Render source repository was declined. The Render workspace was reached through its existing sign-in. No public URL has been verified. |
| External benchmarks | The AIForge-Doc v2 importer was verified on generated ZIP fixtures. The publisher archive has not reached the workspace, so real-document validation remains pending. The supplied model must not be used to make decisions about real documents. |

See the technical report, model card, and release JSON for complete experiment details.