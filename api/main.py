import importlib
import json
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from time import monotonic, perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .contracts import Analysis, Prediction

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web" / "public"
RELEASE = ROOT / "release"
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
FORMATS = ("PNG", "JPEG", "WEBP")
EXPERIMENTS = ("summary", "robustness", "ablations", "calibration")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    adapter = os.getenv("FORGERYLENS_ADAPTER")
    if adapter is None and (RELEASE / "model.npz").is_file():
        adapter = "api.signal:create_predictor"
    if adapter:
        try:
            module, factory = adapter.rsplit(":", 1)
            predictor = getattr(importlib.import_module(module), factory)()
            if not isinstance(predictor.version, str) or not predictor.version.strip():
                raise ValueError("missing model version")
            if not callable(predictor.predict):
                raise ValueError("missing predictor method")
            app.state.predictor = predictor
        except Exception:
            log.exception("Model adapter failed to load")
    yield


app = FastAPI(title="ForgeryLens", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in os.getenv("FORGERYLENS_ORIGINS", "").split(",") if item.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.mount("/assets", StaticFiles(directory=PUBLIC), name="assets")
recent_requests: deque[float] = deque()
active_requests = 0


def decode_document(content: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(content), formats=FORMATS) as image:
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Use an image with at most 16 megapixels.")
            if getattr(image, "is_animated", False):
                raise HTTPException(415, "Use a single-frame JPEG, PNG or WebP image.")
            return ImageOps.exif_transpose(image).convert("RGB")
    except Image.DecompressionBombError as error:
        raise HTTPException(413, "Use an image with at most 16 megapixels.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise HTTPException(415, "Upload a readable JPEG, PNG or WebP image.") from error


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "model_ready": app.state.predictor is not None}


@app.get("/api/v1/model-info")
async def model_info():
    predictor = app.state.predictor
    return {
        "available": predictor is not None,
        "version": predictor.version if predictor else None,
        "scope": getattr(predictor, "meta", {}).get("scope") if predictor else None,
        "max_bytes": MAX_BYTES,
        "max_pixels": MAX_PIXELS,
        "formats": list(FORMATS),
    }


def analyze_document(predictor, content: bytes) -> Analysis:
    image = decode_document(content)
    started = perf_counter()
    try:
        prediction = Prediction.model_validate(predictor.predict(image))
    except Exception as error:
        log.error("Model prediction failed (%s)", type(error).__name__)
        raise HTTPException(500, "Model analysis failed.") from error

    return Analysis(
        **prediction.model_dump(),
        analysis_id=str(uuid4()),
        model_version=predictor.version,
        inference_ms=round((perf_counter() - started) * 1000, 2),
        image_width=image.width,
        image_height=image.height,
        rejected=prediction.decision == "inconclusive",
    )


@app.post("/api/v1/analyze", response_model=Analysis)
async def analyze(request: Request):
    global active_requests
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(503, "Analysis is unavailable until a trained model is installed.")
    now = monotonic()
    while recent_requests and now - recent_requests[0] >= 60:
        recent_requests.popleft()
    if len(recent_requests) >= 30 or active_requests >= 1:
        raise HTTPException(429, "Analysis is busy. Try again shortly.", headers={"Retry-After": "5"})
    if request.headers.get("content-type", "").split(";", 1)[0] not in (
        "image/jpeg", "image/png", "image/webp", "application/octet-stream"
    ):
        raise HTTPException(415, "Send the encoded image as the request body.")
    recent_requests.append(now)
    active_requests += 1
    try:
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > MAX_BYTES:
                raise HTTPException(413, "Use an image no larger than 8 MiB.")
            content.extend(chunk)
        if not content:
            raise HTTPException(400, "The image is empty.")
        return await run_in_threadpool(analyze_document, predictor, bytes(content))
    finally:
        active_requests -= 1


@app.get("/api/v1/experiments/{section}")
def experiments(section: str):
    if section not in EXPERIMENTS:
        raise HTTPException(404, "Unknown experiment section.")
    folder = os.getenv("FORGERYLENS_EXPERIMENTS_DIR", str(RELEASE))
    if folder:
        path = Path(folder) / (section + ".json")
        try:
            with path.open("rb") as artifact:
                envelope = json.loads(artifact.read(512 * 1024 + 1))
            if isinstance(envelope, dict) and isinstance(envelope.get("run_id"), str) and envelope["run_id"]:
                return {"status": "available", "artifact": envelope}
        except (OSError, ValueError):
            pass
    return {"status": "pending", "artifact": None}


@app.get("/", include_in_schema=False)
@app.get("/studio", include_in_schema=False)
@app.get("/research", include_in_schema=False)
@app.get("/methodology", include_in_schema=False)
@app.get("/about", include_in_schema=False)
@app.get("/privacy", include_in_schema=False)
def frontend():
    return FileResponse(PUBLIC / "index.html")


if os.getenv("FORGERYLENS_REFERENCE_RESEARCH") == "1":
    from .reference_research import install

    install(app, ROOT / "web" / "reference_studio.html")

    @app.middleware("http")
    async def reference_studio_link(request: Request, call_next):
        if request.method == "GET" and request.url.path == "/studio":
            html = (PUBLIC / "index.html").read_text(encoding="utf-8")
            html = html.replace("</header>", '<p class="wrap"><a href="/studio/compare">'
                                'Compare references</a></p>\n</header>', 1)
            return HTMLResponse(html, headers={"Cache-Control": "no-store"})
        return await call_next(request)
