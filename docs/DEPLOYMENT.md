# Deploy ForgeryLens

The Receipt Studio is a static Next.js export for Cloudflare Workers. The earlier FastAPI demos have separate [release profiles](history/DEPLOYMENT_GUIDE.md).

## Update the existing Cloudflare Worker

Build the saved-sample release from the repository root. The build copies the checked-in result JSON and requires neither the private research artifacts nor a live inference service.

Bash

```bash
npm --prefix web/frontend ci
export NEXT_TELEMETRY_DISABLED='1'
unset NEXT_PUBLIC_FORGERYLENS_API_URL
npm --prefix web/frontend run build
npx wrangler login
npx wrangler deploy --config web/frontend/wrangler.jsonc
```

PowerShell

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

```sh
python -c "import shutil; shutil.make_archive('ForgeryLens-Cloudflare-Static-Upload', 'zip', 'web/frontend/out')"
```

The dashboard's **Create an app** flow creates a new Worker. For an update that retains the existing URL, use the Wrangler workflow above.

## Optional live API

Set `NEXT_PUBLIC_FORGERYLENS_API_URL` to a separately authorized HTTPS API origin before building. The API must allow the deployed frontend origin in `FORGERYLENS_ORIGINS`. The saved specimens remain available during service outages. The default static build previews custom uploads locally. [Frontend API contract](API_CONTRACT.md)

For local scoring, activate the [frozen receipt environment](DEMO_GUIDE.md#frozen-receipt-detector) and start the opt-in adapter. Run the studio in another shell with `NEXT_PUBLIC_FORGERYLENS_API_URL` set to `http://127.0.0.1:8000`.

Bash

```bash
python -m pip install fastapi==0.135.1 uvicorn==0.41.0
export FORGERYLENS_FROZEN_RECEIPT='1'
export FORGERYLENS_ORIGINS='http://localhost:3000'
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

PowerShell

```powershell
python -m pip install fastapi==0.135.1 uvicorn==0.41.0
$env:FORGERYLENS_FROZEN_RECEIPT = '1'
$env:FORGERYLENS_ORIGINS = 'http://localhost:3000'
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

## Historical single service

Build from the project root with `docker build -t forgerylens .` and run with `docker run --rm -p 8000:8000 forgerylens`. The container bundles the historical frontend, aggregate experiment artifacts, and NumPy model. The Render Blueprint targets a Docker web service and checks `/api/v1/ready`. Use an authorized Render account, import the repository, and verify that the image starts and loads a model.

Run `python tests/smoke.py https://your-host --image web/public/edited.png`. This requires a real model response, plus the frontend routes, asset and readiness checks. The bundled model produces research scores for the demo specimens and returns inconclusive for unfamiliar documents.

A free Render service can sleep during inactivity. Check resume time and steady-state behavior before a live demo. Test provider request limits with representative images.

## Adapter selection and request limits

Leave `FORGERYLENS_ADAPTER` unset to discover the bundled model in `release/`. The allowlist accepts `api.signal:create_predictor`. An explicitly empty value disables the analyzer. A factory outside the allowlist leaves the model unavailable and readiness returns HTTP 503.

The historical analyzer tracks requests by the ASGI peer IP and serializes scoring through a bounded, timed queue. The limiter and queue live in each worker. Behind a reverse proxy, configure Uvicorn's `--forwarded-allow-ips` for your trusted proxy addresses so the ASGI peer can reflect the client. With an untrusted proxy, traffic can share the proxy's bucket. [Uvicorn settings](https://uvicorn.dev/settings/#http)

Use `FORGERYLENS_EXPERIMENTS_DIR` to point to approved, measured JSON artifacts. Summary, robustness, ablations and calibration files require a nonempty `run_id`. The [API contract](API_CONTRACT.md) describes accepted image formats and retry responses.

- [Render free service limits](https://render.com/docs/free)
