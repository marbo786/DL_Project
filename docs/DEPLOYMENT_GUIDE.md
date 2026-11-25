# ForgeryLens deployment guide

## Release profiles

| Profile | Entry point | Evidence and deployment status |
| --- | --- | --- |
| Static Receipt Studio | [Cloudflare Worker](https://patient-cake-9531.marboo786.workers.dev/) or `web/frontend/out` | Recorded fictional samples and published aggregates. The redesigned export passed local checks. Production verification is pending. [Build and deployment](DEPLOYMENT.md) |
| Historical synthetic release | Default `/studio` or `/studio/single`, with `/api/v1/analyze` | Known fictional specimens. Existing Render profile is defined in [render.yaml](../render.yaml). |
| Local reference research | Opt-in `/studio`, also `/studio/compare`, with `/api/v1/compare` | Visual comparison with registered SSIM and a trusted reference. Local loopback only. |
| Public reference upload | No active deployment profile | Requires separate privacy, security, capacity, and cost review plus explicit release authorization. |

The operator flag defaults off in Docker and Render. The research comparison checks both the client peer and HTTP host against loopback names and rejects a supplied cross-origin POST. Run it directly on loopback with proxy headers disabled. Do not route traffic from a public reverse proxy, tunnel, or hosted frontend to the local research listener. The guard assumes direct local access. The two-image workbench depends on the research source, policy, pinned OpenCV and scientific stack, and multipart support. The production image deliberately omits these dependencies, the research page and scripts, datasets, private artifacts, and unneeded checkpoints.

## Local research demonstration

```powershell
py -m pip install -r research/requirements-reference.txt
$env:FORGERYLENS_REFERENCE_RESEARCH = '1'
$env:OPENBLAS_NUM_THREADS = '1'
py -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-proxy-headers --no-access-log
```

Open [Reference Studio](http://127.0.0.1:8000/studio). Use a single worker for the demonstration. The research adapter verifies the frozen comparator source hashes at startup. The full requirements include PyTorch, which the unchanged research import chain uses even though the comparator is classical. Use the FastAPI-served interface for reference mode.

## Transport and privacy

The pair API streams multipart data into memory. It allows two named files, 8 MiB each, plus 8 KiB of envelope overhead, with a 20-second upload deadline and one in-flight comparison per process. It rejects duplicate fields, incomplete messages, wrong types, animated files, and images over 16 megapixels. Multipart buffers close on success or error and no temporary image files are written. Successful responses and upload errors have no-store headers. The browser holds selection previews and comparison output in memory. The process may still be subject to OS paging, crash reports, or system logging. Keep uploads fictional or explicitly authorized and avoid public tunnels. The unchanged registration gate abstains without evidence maps when alignment is unreliable. The research requirements pin a parser version for each supported Python version, above the releases affected by the published preamble and header resource-exhaustion advisories. [Parser advisories](https://github.com/Kludex/python-multipart/security/advisories) · [Parser release history](https://pypi.org/project/python-multipart/#history)

## Historical Docker deployment

```powershell
docker build -t forgerylens .
docker run --rm -p 127.0.0.1:8000:8000 forgerylens
py tests/smoke.py http://127.0.0.1:8000 --image web/public/edited.png
```

The production build context is allowlisted. Render remains configured for the `main` branch. The branch under review uses `/api/v1/ready` for readiness. It returns HTTP 503 if the historical model failed to load. The liveness and model-contract routes keep their existing semantics. Render's free tier has constrained memory, idle spin-downs, and usage limits. Treat it as a historical demo host. No Phase G deployment was performed. [Render free services](https://render.com/docs/free) · [Render pricing](https://render.com/pricing) · [Render health checks](https://render.com/docs/health-checks)

## Release gates

For a limited public preview, prefer packaging only the four fictional examples and their precomputed responses from a documented policy run. This would keep user files in the browser and remove server-side image parsing from the public surface. A separate upload service requires sandboxed decoding, reverse-proxy body and deadline limits, admission control across workers, privacy and retention review, monitoring, load and memory testing on the intended instance, and cost approval. Establish the conditions for which the independent physical-capture evidence supports any operational performance claim. Keep all claims scoped to comparing visible structure with a separately trusted reference.

## Historical Phase G verification

Verified the default and opt-in FastAPI modes in loopback API and Chromium workflows. The optional Next.js production build passed in an isolated copy. Its dependency ranges allow resolver drift, so the historical Docker/FastAPI image remains the deployment target. The Docker daemon was unavailable on this workstation, so container build and startup are unverified. No production rollout, externally hosted load test, or independent accuracy evaluation was performed. [Experiment summary](EXPERIMENT_SUMMARY.md)

## Phase G repository audit

| Priority | Finding | Disposition |
| --- | --- | --- |
| High | The earlier upload handler buffered the full body and used a multipart parser that could spool user images to disk. The local-mode flag did not itself require a local peer. | Bounded memory-only parsing, upload deadline, direct loopback and origin checks, failure cleanup, and parser pins. |
| High | The historical readiness route reported healthy even when the model was missing. | Render template checks the new model-aware readiness route. |
| High | Setting the comparison flag on the historical image would fail because it lacks the research source, HTML and runtime. | Documented separate local profile and set the Render and Docker defaults to disabled. |
| Medium | The earlier Studio offered mostly static preview output with weak coordinate inspection. | Added synchronized controls, fixed coordinate-frame metadata, aligned overlays, and keyboard and responsive checks. |
| Medium | Historical website copy described a held-out editing generator. The external receipt holdout used a new source corpus with the same generator. | Corrected the methodology and linked the separate historical and reference results. |
| Medium | The previous Docker context excluded named folders but could still include other local files and extra checkpoints. | Allowlisted the build context and kept the historical release assets. |
| Research boundary | Training, manifests, checkpoints, calibration, and prior aggregate results use different partitions and task scopes. | Reviewed and cross-referenced Phases A–F with their partition labels. The sealed datasets, frozen algorithm, and previous measurements remain intact. |

