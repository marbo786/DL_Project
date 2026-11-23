"""Opt-in, local research comparison endpoint."""

import asyncio
from base64 import b64encode
from io import BytesIO
from time import perf_counter

import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from research.phase_f_compare import MAX_BYTES, POLICY, PRIMARY, compare, decode, verify_freeze


def png(array):
    encoded = BytesIO()
    Image.fromarray(array).save(encoded, format="PNG")
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
        response["alignment"].update(inliers=alignment.inliers, inlier_ratio=alignment.inlier_ratio,
                                     reprojection_px=alignment.reprojection_px)
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
        response["previews"] = {
            "reference": png(result.reference), "aligned_candidate": png(alignment.image),
            "heatmap": png(heat), "region_overlay": png(np.asarray(overlay)),
            "valid_overlap": png(alignment.valid.astype(np.uint8) * 255),
        }
    response["inference_ms"] = round((perf_counter() - started) * 1000, 2)
    return response


def install(app, studio_page):
    verify_freeze()
    busy = asyncio.Lock()

    @app.get("/studio/compare", include_in_schema=False)
    def comparison_studio():
        return FileResponse(studio_page, headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/compare", include_in_schema=False)
    async def comparison(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "multipart/form-data":
            raise HTTPException(415, "Upload a multipart reference and candidate.")
        if busy.locked():
            raise HTTPException(429, "Comparison is busy. Try again shortly.")
        async with busy:
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > 2 * MAX_BYTES + 8192:
                    raise HTTPException(413, "Use two images no larger than 8 MiB each.")
                body.extend(chunk)

            async def receive():
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            buffered = Request(request.scope, receive)
            async with buffered.form(max_files=2, max_fields=0) as form:
                if sorted(key for key, _ in form.multi_items()) != ["candidate", "reference"]:
                    raise HTTPException(400, "Provide exactly one reference and one candidate.")
                uploads = [form[key] for key in ("reference", "candidate")]
                if not all(isinstance(item, UploadFile) for item in uploads):
                    raise HTTPException(400, "Both inputs must be files.")
                reference, candidate = [await item.read(MAX_BYTES + 1) for item in uploads]
            if max(len(reference), len(candidate)) > MAX_BYTES:
                raise HTTPException(413, "Use images no larger than 8 MiB each.")
            payload = await run_in_threadpool(comparison_payload, reference, candidate)
            return JSONResponse(payload, headers={"Cache-Control": "no-store"})
