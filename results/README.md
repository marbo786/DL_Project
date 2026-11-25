# Saved result evidence

These JSON files are byte-for-byte copies of the recorded research outputs. Subdirectory names identify the original runs under `artifacts/`. The copies exclude images, checkpoints, manifests, and per-document CSVs. Some frozen metrics JSON includes ranked error examples with dataset identifiers.

| Evidence | Files |
| --- | --- |
| Training and development | [Fit](receipt-study-v1/residual-42/fit.json), [metrics](receipt-study-v1/residual-42/metrics.json), [containment audit](receipt-protocol-v1-contained/audit.json) |
| Frozen policy and historical tests | [Freeze](receipt-retrospective-v1/freeze.json), [in-domain test](receipt-retrospective-v1/test.json), [WildReceipt](receipt-retrospective-v1/shifted_test.json), [XFUND](receipt-xfund-v1/metrics.json) |
| Follow-up experiments | [Ablations](receipt-ablations-v1/summary.json), [Pareto](receipt-pareto-v1/summary.json), [stress](receipt-residual-stress-v1/summary.json), [attribution](receipt-explain-v1/summary.json), [independent recomputation](receipt-verification-v1/recomputed.json) |

The following documented output paths were absent from the workspace:

- `artifacts/phase-f/v1/results.json` requires the independent physical-capture evaluation. No result was available to copy.
- `artifacts/receipt-verification-repeat/recomputed.json`, `artifacts/receipt-verification-repeat/training.json`, and `artifacts/receipt-verification-repeat/export.json` are destinations for optional reruns. The corresponding original records are in [receipt-verification-v1](receipt-verification-v1).

The [reproduction guide](../docs/SINGLE_IMAGE_REPRODUCTION.md) describes the private inputs and the commands for a new run. The checked-in frontend export retains its original source provenance.
