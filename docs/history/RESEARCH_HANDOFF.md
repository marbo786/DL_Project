# Historical synthetic-study handoff

## Reproduce the historical synthetic release

Run these commands from the repository root. They regenerate the synthetic release and specimen assets. Use a separate checkout to keep the published files intact.

Bash

```bash
python -m pip install -r research/requirements.txt
python -m research.train --epochs 12 --train-sources 144 --threads 2 --output release --web-assets web/public
export OPENBLAS_NUM_THREADS='1'
python -m research.export release/model.pt release/model.npz
python -m research.demo release web/public
python -m research.gallery release web/public
python -m research.report release docs/archive/ForgeryLens_Historical_Synthetic_Report.pdf
```

PowerShell

```powershell
python -m pip install -r research/requirements.txt
python -m research.train --epochs 12 --train-sources 144 --threads 2 --output release --web-assets web/public
$env:OPENBLAS_NUM_THREADS = '1'
python -m research.export release/model.pt release/model.npz
python -m research.demo release web/public
python -m research.gallery release web/public
python -m research.report release docs/archive/ForgeryLens_Historical_Synthetic_Report.pdf
```

For the historical whole-image AIForge experiment, follow the [benchmark write-up](../ForgeryLens_AIForge_Results.md). For the frozen single-image receipt study, follow the [reproduction guide](../SINGLE_IMAGE_REPRODUCTION.md).

The [receipt study](../SINGLE_IMAGE_RESULTS.md) subsequently ran on paired AIForge data. The [fresh receipt evaluation](../FRESH_SHIFT_EVALUATION.md) still requires new source lineages.

The completed synthetic run, model card and report describe the current demonstration. A real-document replication needs authorized data and an independent test protocol.

1. Record access terms and dataset versions. Resolve authentic and edited pairs to canonical source identity. Inspect uncertain and near-duplicate records across corpora before committing the manifest.
2. Freeze source-group splits and generators. Build and audit the manifest, inspect masks and format shortcuts, and flag missing authentic training counterparts.
3. Repeat training, controlled comparisons, optimization, calibration and robustness on the authorized manifest. Keep the shifted test outside model selection. Record scanner, compression and document-type limits.
4. Verify deployment terms before distributing any new weights or example documents. Measure the integrated public-host workflow using the released checkpoint and predeclared thresholds.
