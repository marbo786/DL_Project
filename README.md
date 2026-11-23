# ForgeryLens

ForgeryLens is an AI341 research prototype for inspecting visible document changes. The local Reference Studio compares two images using frozen registered SSIM. The historical single-image model scores its known fictional specimens and abstains on unfamiliar inputs. The independent real-world evaluation is still awaiting an authorized capture set.

[Final report](docs/FINAL_RESEARCH_REPORT.md) · [Recorded experiments](docs/EXPERIMENT_SUMMARY.md) · [Demo guide](docs/DEMO_GUIDE.md) · [Deployment guide](docs/DEPLOYMENT_GUIDE.md) · [Physical capture protocol](docs/REAL_CAPTURE_COLLECTION_GUIDE.md)

## Run on Windows

Open PowerShell at the project root. For the two-document research Studio, install the research requirements and opt in on loopback.

```powershell
py -m pip install -r research/requirements-reference.txt
$env:FORGERYLENS_REFERENCE_RESEARCH = '1'
py -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-proxy-headers --no-access-log
```

Open [Forensic Studio](http://127.0.0.1:8000/studio), choose a fictional example, and select **Compare documents**. Use **Single-image demo** for the historical workflow. The SSIM score measures differences in the registration overlap. Check reference identity independently.

For the historical release alone, leave research mode off and install only the API dependencies.

```powershell
py -m pip install -r api/requirements.txt
$env:FORGERYLENS_REFERENCE_RESEARCH = '0'
py -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Choose Original specimen or Edited specimen and click Analyze. The Low-quality specimen demonstrates abstention while still displaying the historical model heatmap. The separate Research page displays measured experiment artifacts. The curated demo pair was selected from a separate fictional population and is excluded from test metrics.

## Reproduce the released run

```powershell
py -m pip install -r research/requirements.txt
py -m research.train --epochs 12 --train-sources 144 --threads 2 --output release --web-assets web/public
$env:OPENBLAS_NUM_THREADS='1'
py -m research.export release/model.pt release/model.npz
py -m research.demo release web/public
py -m research.gallery release web/public
py -m research.report release docs/ForgeryLens_Technical_Report.pdf
```

To evaluate authorized data, prepare a normalized manifest with group-disjoint train, validation, test, and shifted_test partitions. Include the canonical source ID and pixel mask for each localized edit. Validate the corpus with `py -m research.manifest path/to/manifest.csv path/to/assets`. Add `--manifest path/to/manifest.csv --asset-root path/to/assets` to the training command, and use a separate output directory for the run. Review validation behavior and licensing before releasing any trained artifact.

## Import the published receipt benchmark

The publisher links the [full AIForge-Doc v2 ZIP](https://drive.google.com/file/d/14rUt2SIOHSSEW_gYeywIaXtRTaYcwr4T/view) from its [dataset card](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2#paired-authentic-images). Use the full package with authentic counterparts and masks.

```powershell
py -m research.aiforge AIForge-Doc-v2.0.zip data/aiforge
py -m research.manifest data/aiforge/manifest.csv data/aiforge/assets --deduplicated-output data/aiforge/manifest_deduplicated.csv
py -m research.train --manifest data/aiforge/manifest_deduplicated.csv --asset-root data/aiforge/assets --output artifacts/aiforge --epochs 8 --threads 6
```

The importer keeps originals and edits together by source image. The deduplication step retains one source per group of byte-identical receipts, prioritizing the held-out splits. Training audits the cleaned manifest before fitting. WildReceipt is held out as a source-corpus shift. The benchmark uses one editing generator.

The eight-epoch experiment and its measured outcomes are documented in the [AIForge-Doc benchmark results](docs/ForgeryLens_AIForge_Results.md).

## Checks and deployment

```powershell
py -m pip install -r api/requirements-dev.txt
py -m pytest -q
node --check web/public/studio.js
py tests/smoke.py http://127.0.0.1:8000 --image web/public/edited.png
```

The FastAPI container serves the Studio and loads the NumPy model without PyTorch. See [deployment](docs/DEPLOYMENT.md), [model card](docs/MODEL_CARD.md), and the [dataset audit](docs/DATASET_AUDIT.md).
