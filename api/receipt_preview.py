import asyncio
from functools import lru_cache
from io import BytesIO
from math import exp, isfinite
from time import perf_counter
from uuid import uuid4

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from research.receipt_demo import FORMATS, MAX_BYTES, MAX_PIXELS, POLICY_VERSION, FrozenReceiptDemo, route_receipt
from research.receipt_model import inference_tiles, pooled_logit, tile_logits

router = APIRouter()
busy = asyncio.Lock()
media_types = {"image/jpeg", "image/png", "image/webp"}
no_store = {"Cache-Control": "no-store"}
decisions = {"Prioritize review": "prioritize_review", "Normal queue": "normal_queue", "Inconclusive": "inconclusive"}


class UnsupportedImage(ValueError):
    def __init__(self, reason, advice, status):
        super().__init__(advice)
        self.reason = reason
        self.status = status

    def response(self):
        return JSONResponse({"outcome": "unsupported_input", "reason": self.reason, "advice": str(self)},
                            status_code=self.status, headers=no_store)


@lru_cache(maxsize=1)
def detector():
    return FrozenReceiptDemo()


def decode(content):
    if not content:
        raise UnsupportedImage("empty", "Choose a nonempty image.", 400)
    if len(content) > MAX_BYTES:
        raise UnsupportedImage("too_large", "Choose an image no larger than 8 MiB.", 413)
    try:
        with Image.open(BytesIO(content), formats=FORMATS) as source:
            if source.width * source.height > MAX_PIXELS:
                raise UnsupportedImage("too_large", "Choose an image of at most 16 megapixels.", 413)
            if getattr(source, "is_animated", False):
                raise UnsupportedImage("animated", "Choose a single-frame image.", 415)
            return ImageOps.exif_transpose(source).convert("RGB")
    except Image.DecompressionBombError as error:
        raise UnsupportedImage("too_large", "Choose an image of at most 16 megapixels.", 413) from error
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise UnsupportedImage("unreadable", "Choose a readable JPEG, PNG, or WebP image.", 415) from error


def score_receipt(content, frozen):
    started = perf_counter()
    image = decode(content)
    width, height = image.size
    boxes, tensors = inference_tiles(image, frozen.model.mode)
    scores = tile_logits(frozen.model, tensors, frozen.batch_size)
    values = scores.tolist()
    logit = pooled_logit(scores)
    if not isfinite(logit) or not all(map(isfinite, values)):
        raise ValueError("Nonfinite receipt score")
    scaled = max(-60., min(60., frozen.slope * logit + frozen.intercept))
    probability = 1 / (1 + exp(-scaled))
    decision, advice = route_receipt(logit, probability, frozen.review_cutoff)

    def bounds(box):
        x0, y0, x1, y1 = box
        return {"x": x0 / width, "y": y0 / height,
                "width": (min(x1, width) - x0) / width, "height": (min(y1, height) - y0) / height}

    return {"outcome": "analyzed", "analysis_id": str(uuid4()), "model_version": frozen.checkpoint_sha256,
            "policy_version": POLICY_VERSION, "decision": decisions[decision], "calibrated_probability": probability,
            "confidence": max(probability, 1 - probability), "advice": advice,
            "image_width": width, "image_height": height, "top_tile": bounds(boxes[int(scores.argmax())]),
            "tiles": [{**bounds(box), "logit": value} for box, value in zip(boxes, values)],
            "inference_ms": round((perf_counter() - started) * 1000, 2)}


@router.post("/api/v1/receipt/analyze")
async def analyze_receipt(request: Request):
    if request.headers.get("content-type", "").split(";", 1)[0].lower() not in media_types:
        return UnsupportedImage("unsupported_format", "Choose a JPEG, PNG, or WebP image.", 415).response()
    if busy.locked():
        raise HTTPException(429, "Receipt analysis is busy. Try again shortly.", headers=no_store)
    async with busy:
        content = bytearray()
        try:
            with anyio.fail_after(30):
                async for chunk in request.stream():
                    if len(content) + len(chunk) > MAX_BYTES:
                        return UnsupportedImage("too_large", "Choose an image no larger than 8 MiB.", 413).response()
                    content.extend(chunk)
        except (ClientDisconnect, TimeoutError):
            raise HTTPException(408, "Upload timed out or was interrupted.", headers=no_store) from None
        if not content:
            return UnsupportedImage("empty", "Choose a nonempty image.", 400).response()
        try:
            frozen = await run_in_threadpool(detector)
        except (OSError, RuntimeError, ValueError):
            raise HTTPException(503, "The frozen receipt detector is unavailable.", headers=no_store) from None
        try:
            result = await run_in_threadpool(score_receipt, content, frozen)
        except UnsupportedImage as error:
            return error.response()
        except (RuntimeError, ValueError):
            raise HTTPException(500, "Receipt scoring could not complete.", headers=no_store) from None
        return JSONResponse(result, headers=no_store)
