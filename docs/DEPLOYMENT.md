# Deploy ForgeryLens

## Single service

Build from the project root with `docker build -t forgerylens .` and run with `docker run --rm -p 8000:8000 forgerylens`. The container bundles the frontend, comparison artifacts, and NumPy model. The Render Blueprint targets a Docker web service and checks `/api/v1/health`. Use an authorized Render account, import the repository, and verify that the image starts and loads a model.

Run `python tests/smoke.py https://your-host --image web/public/edited.png`. This requires a real model response, plus the frontend routes, asset and readiness checks. The bundled model produces research scores for the demo specimens and returns inconclusive for unfamiliar documents.

A free Render service can sleep during inactivity. Check resume time and steady-state behavior before a live demo. For a separate Next.js frontend, select the `web` root in Vercel and set `FORGERYLENS_API_ORIGIN` to the HTTPS backend. Test provider request limits with representative images.

## Custom model

To replace the built-in model, set `FORGERYLENS_ADAPTER=package.module:create_predictor` with an installed, trusted package. The factory returns a stable `version` string and synchronous `predict(rgb_image)` method. The input is an EXIF-oriented RGB Pillow image. Responses conform to `api/contracts.py`. Regions use normalized image coordinates and heatmaps use row-major order.

Use `FORGERYLENS_EXPERIMENTS_DIR` to point to approved, measured JSON artifacts. Summary, robustness, ablations and calibration files require a nonempty `run_id`. The bundled API validates file signatures, supports single-frame JPEG, PNG and WebP, enforces 8 MiB and 16 MP limits, and rate-limits expensive requests per process. A public host should apply its own request timeouts and abuse controls.

- [Render free service limits](https://render.com/docs/free)
- [Next.js on Vercel](https://vercel.com/docs/frameworks/full-stack/nextjs)