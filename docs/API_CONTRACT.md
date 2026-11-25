# ForgeryLens frontend API contract

The approved adapter is available when `FORGERYLENS_FROZEN_RECEIPT=1` is set on the local API. The frontend defaults to saved-sample mode when its API URL is unset.

## Current interfaces

`POST /api/v1/analyze` accepts raw JPEG, PNG, or WebP bytes. The bundled model scores its fictional-specimen allowlist. Its `Analysis` response includes `analysis_id`, `model_version`, `decision` (`no_strong_evidence`, `suspicious`, or `inconclusive`), nullable `predicted_class`, `calibrated_probability` and `uncertainty`, normalized `regions`, nullable bounded row-major `heatmap`, `warnings`, `inference_ms`, `image_width`, `image_height`, and `rejected`. Its timer starts after decoding. Its heatmap and regions come from the historical model.

`GET /api/v1/health` reports model readiness. `GET /api/v1/ready` returns 503 when the model is unavailable. `GET /api/v1/model-info` reports availability, version, scope, and input limits. `GET /api/v1/experiments/{section}` returns `{status, artifact}` for `summary`, `calibration`, `robustness`, or `ablations`; the default source is the historical `release/` directory.

`POST /api/v1/compare` takes multipart `reference` and `candidate` images in opt-in loopback research mode. It returns an SSIM decision, reason, score, threshold, policy version, alignment, and timing, with previews and raster metadata when alignment is supported.

`research/receipt_demo.py` runs the frozen `receipt-preview-v1` policy in Streamlit. The HTTP adapter uses the same frozen checkpoint, tile scorer, pooling, calibration, and routing policy. It also exposes the native tile bounds and raw logits for the overlay.

## Frozen receipt endpoint

`POST /api/v1/receipt/analyze` accepts raw bytes with `image/jpeg`, `image/png`, or `image/webp`. Use the existing receipt limits of 8 MiB and 16 megapixels, with EXIF orientation applied and one frame per file. Return JSON with `Cache-Control: no-store`. The UI can show upload progress from bytes sent when available, then switch to indeterminate processing while waiting for a response.

```ts
type Bounds = { x: number; y: number; width: number; height: number };
type Tile = Bounds & { logit: number };

type ReceiptResult =
  | {
      outcome: 'analyzed';
      analysis_id: string;
      model_version: string;
      policy_version: string;
      decision: 'prioritize_review' | 'normal_queue' | 'inconclusive';
      calibrated_probability: number;
      confidence: number;
      advice: string;
      inference_ms: number;
      image_width: number;
      image_height: number;
      top_tile: Bounds;
      tiles: Tile[] | null;
    }
  | {
      outcome: 'unsupported_input';
      reason: 'unreadable' | 'unsupported_format' | 'animated' | 'too_large' | 'empty';
      advice: string;
    };
```

- Use the saved checkpoint and frozen policy outputs as the source of the calibrated probability, confidence (`max(p, 1 - p)`), decision, and advice. An analyzed receipt can be inconclusive and still contain evidence. Unsupported input describes file validation. Open-set document rejection remains unvalidated.
- Bounds use fractions of the EXIF-oriented image, with the origin at top left. Use the actual inference tile geometry, including edge tiles and overlaps. A tile logit is an uncalibrated model score. The client can shade real tile scores for the overlay and show exact tile bounds for the grid. With `tiles: null`, show only the available top-tile highlight.
- Measure `inference_ms` from decode start through completed scoring and policy evaluation, excluding upload, queueing, serialization, and cold startup. Label measured browser wait separately if displayed. Return typed unsupported-input responses with HTTP 400, 413, or 415 as appropriate. Treat 429 and 503 as retryable service states and 5xx as service failures.
- A static host calls `NEXT_PUBLIC_FORGERYLENS_API_URL` directly and needs its own origin in `FORGERYLENS_ORIGINS`. The local adapter requires the frozen checkpoint, calibration table and freeze file described in [the demo guide](DEMO_GUIDE.md#frozen-receipt-detector). Start the API with `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log`. A public live endpoint would need separate authorization for the frozen weights and hosting.
- Saved-sample mode uses `web/frontend/data/sample-results.json`, generated from the bundled fictional receipts by `web/frontend/scripts/export-specimens.py`. The included times are recorded local inference times. Arbitrary uploads in this mode stay in the browser and show a preview only.
