# ForgeryLens

ForgeryLens is an AI341 Track A experiment on single-image receipt review. A residual MobileNetV3-Small scores native-resolution patches and highlights the highest-scoring tile. In the historical WildReceipt evaluation the preview routed six genuine and six edited receipts to the normal queue and deferred the other 2,660. University reimbursement claims require manual review.

[Course report (PDF)](docs/ForgeryLens_Technical_Report.pdf) · [IEEEtran source](docs/ForgeryLens_Technical_Report.tex) · [Model card](docs/MODEL_CARD.md) · [Receipt experiments](docs/SINGLE_IMAGE_RESULTS.md) · [Requirements](docs/REQUIREMENTS.md) · [Team contributions](docs/AUTHORSHIP_AND_AI_DISCLOSURE.md) · [Archived synthetic report](docs/archive/ForgeryLens_Historical_Synthetic_Report.pdf)

## Receipt Studio frontend

From the repository root, run the Next.js studio with Node.js 20.9 or newer.

```powershell
npm --prefix web/frontend ci
$env:NEXT_TELEMETRY_DISABLED = '1'
npm --prefix web/frontend run dev
```

Open [localhost:3000](http://localhost:3000). Saved samples show recorded results from the frozen model. Local files can be previewed in the browser.

The frontend includes an interactive receipt specimen, an original/edited comparison, a scrollable pipeline walkthrough, and a threshold explorer backed by published aggregates. Results, method, and limits are readable with JavaScript disabled. Charts and the receipt studio load on demand.

### Build and redeploy the static frontend

```powershell
Remove-Item Env:NEXT_PUBLIC_FORGERYLENS_API_URL -ErrorAction SilentlyContinue
npm --prefix web/frontend run build
```

The default build writes the saved-sample site to `web/frontend/out`. It works while the API is unavailable and keeps local uploads in preview mode. For a local check of the export, run `py -m http.server 8001 --bind 127.0.0.1 --directory web/frontend/out` and open [localhost:8001](http://localhost:8001).

Deploy to the existing `patient-cake-9531` Worker from a session authenticated to its Cloudflare account. Wrangler runs as a deployment CLI, outside the frontend's package dependencies. Repeat the build and deploy commands for subsequent releases.

```powershell
npx wrangler login
npx wrangler deploy --config web/frontend/wrangler.jsonc
```

After deploy, check the [existing Worker URL](https://patient-cake-9531.marboo786.workers.dev/). Verify both themes, open the published charts, replay a saved specimen, and try a PDF to confirm the unsupported-input message. Open the page with JavaScript disabled to check that results, method, and limits remain readable. [Cloudflare Workers static assets](https://developers.cloudflare.com/workers/static-assets/)

For a dashboard static-file upload, package only the contents of `web/frontend/out`, with `index.html` at the archive root. See the [deployment guide](docs/DEPLOYMENT.md#cloudflare-dashboard-upload) for the ZIP command and how to keep the existing Worker URL.

### Connect an authorized API

For local live analysis, use the frozen receipt environment and artifacts described in the [demo guide](docs/DEMO_GUIDE.md#frozen-receipt-detector). Add FastAPI and Uvicorn to that environment, then start the adapter from the repository root in a separate PowerShell session.

```powershell
python -m pip install fastapi==0.135.1 uvicorn==0.41.0
$env:FORGERYLENS_FROZEN_RECEIPT = '1'
$env:FORGERYLENS_ORIGINS = 'http://localhost:3000'
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Set `NEXT_PUBLIC_FORGERYLENS_API_URL` to `http://127.0.0.1:8000` before starting the frontend. For a hosted live build, set it to the authorized HTTPS API origin before building. The API must already allow `https://patient-cake-9531.marboo786.workers.dev` in its comma-separated `FORGERYLENS_ORIGINS`. The frontend URL variable is public build-time configuration. Keep credentials in the backend environment. The saved-specimen controls stay available in live builds, including when the API fails. [Frontend API contract](docs/API_CONTRACT.md)

## Run the course demo

From the project root with the local frozen artifacts, run:

```powershell
py -m pip install -r research/requirements-receipt-demo.txt
py -m streamlit run research/receipt_demo.py --server.address 127.0.0.1 --server.port 8501 --server.maxUploadSize 8 --browser.gatherUsageStats false
```

[Required checkpoint and calibration files](docs/DEMO_GUIDE.md#frozen-receipt-detector) · [Reproduction commands](docs/SINGLE_IMAGE_REPRODUCTION.md) · [Shift and exposure history](docs/FRESH_SHIFT_EVALUATION.md)

## Earlier two-document and fictional demonstrations

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

## Reproduce the historical synthetic run

```powershell
py -m pip install -r research/requirements.txt
py -m research.train --epochs 12 --train-sources 144 --threads 2 --output release --web-assets web/public
$env:OPENBLAS_NUM_THREADS='1'
py -m research.export release/model.pt release/model.npz
py -m research.demo release web/public
py -m research.gallery release web/public
py -m research.report release docs/archive/ForgeryLens_Historical_Synthetic_Report.pdf
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
