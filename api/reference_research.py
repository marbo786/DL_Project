"""Opt-in, local research comparison endpoint."""

import asyncio
from base64 import b64encode
from io import BytesIO
from math import isfinite
from time import perf_counter

import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from research.reference_verification import MAX_BYTES, POLICY, PRIMARY, compare, decode, verify_freeze

from .comparison_upload import comparison_files

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
NO_CACHE = {"Cache-Control": "no-store"}


def local_research(request):
    return (request.client is not None and request.client.host in LOCAL_HOSTS
            and request.url.hostname in LOCAL_HOSTS)


def studio_response(page):
    return FileResponse(page, headers={**NO_CACHE, "Referrer-Policy": "no-referrer",
                                      "X-Content-Type-Options": "nosniff",
                                      "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; "
                                                                 "style-src 'self' 'unsafe-inline'; "
                                                                 "frame-ancestors 'none'; base-uri 'none'"})


def png(array):
    encoded = BytesIO()
    Image.fromarray(array).save(encoded, format="PNG", compress_level=3)
    return "data:image/png;base64," + b64encode(encoded.getvalue()).decode("ascii")


def comparison_payload(reference, candidate):
    started = perf_counter()
    try:
        result = compare(decode(reference), decode(candidate))
    except ValueError as error:
        raise HTTPException(415, str(error)) from error
    alignment = result.alignment
    response = {
        "decision": result.decision,
        "reason": result.reason,
        "score": result.score,
        "threshold": PRIMARY,
        "policy_version": POLICY["version"],
        "alignment": {"supported": result.reason is None},
    }
    if alignment is not None:
        error = alignment.reprojection_px
        response["alignment"].update(inliers=alignment.inliers, inlier_ratio=alignment.inlier_ratio,
                                     reprojection_px=error if error is not None and isfinite(error) else None)
    if result.reason is None:
        opacity = (np.sqrt(result.heatmap) * alignment.valid * 225).astype(np.uint8)
        heat = np.zeros((*opacity.shape, 4), np.uint8)
        heat[..., :3] = [255, 115, 49]
        heat[..., 3] = opacity
        outline = np.zeros_like(heat)
        outline[..., :3] = [255, 218, 95]
        outline[..., 3] = result.mask * np.uint8(185)
        overlay = Image.alpha_composite(Image.fromarray(alignment.image).convert("RGBA"), Image.fromarray(outline))
        response["alignment"]["valid_overlap"] = float(alignment.valid.mean())
        response["region_pixels"] = int(np.count_nonzero(result.mask))
        response["previews"] = {
            "reference": png(result.reference), "aligned_candidate": png(alignment.image),
            "heatmap": png(heat), "region_overlay": png(np.asarray(overlay)),
            "regions": png(outline),
            "valid_overlap": png(alignment.valid.astype(np.uint8) * 255),
        }
        response["raster"] = {"width": result.reference.shape[1], "height": result.reference.shape[0],
                              "frame": "normalized_reference"}
    response["inference_ms"] = round((perf_counter() - started) * 1000, 2)
    return response


def install(app, studio_page):
    verify_freeze()
    busy = asyncio.Lock()

    @app.get("/studio/compare", include_in_schema=False)
    def comparison_studio(request: Request):
        if not local_research(request):
            raise HTTPException(404)
        return studio_response(studio_page)

    @app.post("/api/v1/compare", include_in_schema=False)
    async def comparison(request: Request):
        if not local_research(request):
            raise HTTPException(404)
        origin = request.headers.get("origin")
        if origin is not None and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Open the comparison Studio on this local server.")
        if busy.locked():
            raise HTTPException(429, "Comparison is busy. Try again shortly.")
        async with busy:
            try:
                reference, candidate = await comparison_files(request, MAX_BYTES)
                payload = await run_in_threadpool(comparison_payload, reference, candidate)
            except HTTPException as error:
                error.headers = {**(error.headers or {}), **NO_CACHE}
                raise
            return JSONResponse(payload, headers=NO_CACHE)
