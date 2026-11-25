# Deploy ForgeryLens

The Receipt Studio is a static Next.js export for Cloudflare Workers. The earlier FastAPI demos have separate [release profiles](DEPLOYMENT_GUIDE.md).

## Update the existing Cloudflare Worker

Build the saved-sample release from the repository root. The build copies the checked-in result JSON and requires neither the private research artifacts nor a live inference service.

```powershell
npm --prefix web/frontend ci
$env:NEXT_TELEMETRY_DISABLED = '1'
Remove-Item Env:NEXT_PUBLIC_FORGERYLENS_API_URL -ErrorAction SilentlyContinue
npm --prefix web/frontend run build
npx wrangler login
npx wrangler deploy --config web/frontend/wrangler.jsonc
```

Use the Cloudflare account that owns `patient-cake-9531`. After deploying, run the [Receipt Studio smoke test](RELEASE_READINESS_CHECKLIST.md#receipt-studio-release) on the [existing Worker URL](https://patient-cake-9531.marboo786.workers.dev/). [Cloudflare static asset deployment](https://developers.cloudflare.com/workers/static-assets/get-started/)

## Cloudflare dashboard upload

The static uploader expects `index.html`, `_next/`, and `data/` at the archive root. Package the build output without the Wrangler configuration, source code, audit reports, or enclosing `out/` directory.

```powershell
py -c "import shutil; shutil.make_archive('ForgeryLens-Cloudflare-Static-Upload', 'zip', 'web/frontend/out')"
```

The dashboard's **Create an app** flow creates a new Worker. For an update that retains the existing URL, use the Wrangler workflow above.

## Optional live API

Set `NEXT_PUBLIC_FORGERYLENS_API_URL` to a separately authorized HTTPS API origin before building. The API must allow the deployed frontend origin in `FORGERYLENS_ORIGINS`. The saved specimens remain available during service outages. The default static build previews custom uploads locally. [Frontend API contract](API_CONTRACT.md)

## Historical single service

Build from the project root with `docker build -t forgerylens .` and run with `docker run --rm -p 8000:8000 forgerylens`. The container bundles the historical frontend, aggregate experiment artifacts, and NumPy model. The Render Blueprint targets a Docker web service and checks `/api/v1/ready`. Use an authorized Render account, import the repository, and verify that the image starts and loads a model.

Run `python tests/smoke.py https://your-host --image web/public/edited.png`. This requires a real model response, plus the frontend routes, asset and readiness checks. The bundled model produces research scores for the demo specimens and returns inconclusive for unfamiliar documents.

A free Render service can sleep during inactivity. Check resume time and steady-state behavior before a live demo. Test provider request limits with representative images.

## Custom model

To replace the built-in model, set `FORGERYLENS_ADAPTER=package.module:create_predictor` with an installed, trusted package. The factory returns a stable `version` string and synchronous `predict(rgb_image)` method. The input is an EXIF-oriented RGB Pillow image. Responses conform to `api/contracts.py`. Regions use normalized image coordinates and heatmaps use row-major order.

Use `FORGERYLENS_EXPERIMENTS_DIR` to point to approved, measured JSON artifacts. Summary, robustness, ablations and calibration files require a nonempty `run_id`. The bundled API validates file signatures, supports single-frame JPEG, PNG and WebP, enforces 8 MiB and 16 MP limits, and rate-limits expensive requests per process. A public host should apply its own request timeouts and abuse controls.

- [Render free service limits](https://render.com/docs/free)
