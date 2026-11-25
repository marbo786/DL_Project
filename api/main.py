import asyncio
import json
import logging
import os
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from io import BytesIO
from math import ceil
from pathlib import Path
from time import monotonic, perf_counter
from uuid import uuid4

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from .contracts import Analysis, Prediction
from .signal import create_predictor

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web" / "public"
RELEASE = ROOT / "release"
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
FORMATS = ("PNG", "JPEG", "WEBP")
EXPERIMENTS = ("summary", "robustness", "ablations", "calibration")
RATE_LIMIT = 30
RATE_WINDOW_SECONDS = 60
MAX_RATE_CLIENTS = 2048
MAX_QUEUED_REQUESTS = 2
QUEUE_WAIT_SECONDS = 5
UPLOAD_TIMEOUT_SECONDS = 20
DEFAULT_ADAPTER = "api.signal:create_predictor"
ADAPTER_FACTORIES = {DEFAULT_ADAPTER: create_predictor}
log = logging.getLogger(__name__)


class AnalysisAdmission:
    def __init__(self):
        self.clients: OrderedDict[str, deque[float]] = OrderedDict()
        self.worker = asyncio.Lock()
        self.pending = 0

    def check_rate(self, client: str):
        now = monotonic()
        cutoff = now - RATE_WINDOW_SECONDS
        while self.clients and next(iter(self.clients.values()))[-1] <= cutoff:
            self.clients.popitem(last=False)
        recent = self.clients.get(client)
        if recent is None:
            if len(self.clients) >= MAX_RATE_CLIENTS:
                oldest = next(iter(self.clients.values()))[-1]
                raise HTTPException(503, "Analysis is temporarily at capacity.",
                                    headers={"Retry-After": str(max(1, ceil(oldest + RATE_WINDOW_SECONDS - now)))})
            recent = self.clients[client] = deque()
        while recent and recent[0] <= cutoff:
            recent.popleft()
        if len(recent) >= RATE_LIMIT:
            raise HTTPException(429, "This client has reached its analysis limit.",
                                headers={"Retry-After": str(max(1, ceil(recent[0] + RATE_WINDOW_SECONDS - now))),
                                         "Cache-Control": "no-store"})
        recent.append(now)
        self.clients.move_to_end(client)

    @asynccontextmanager
    async def slot(self):
        retry = {"Retry-After": str(max(1, ceil(QUEUE_WAIT_SECONDS)))}
        if self.pending >= 1 + MAX_QUEUED_REQUESTS:
            raise HTTPException(503, "The analysis queue is full. Try again shortly.", headers=retry)
        self.pending += 1
        try:
            try:
                with anyio.fail_after(QUEUE_WAIT_SECONDS):
                    await self.worker.acquire()
            except TimeoutError:
                raise HTTPException(503, "The analysis queue wait expired. Try again shortly.", headers=retry) from None
            try:
                yield
            finally:
                self.worker.release()
        finally:
            self.pending -= 1


def load_predictor():
    adapter = os.getenv("FORGERYLENS_ADAPTER")
    if adapter is None:
        adapter = DEFAULT_ADAPTER if (RELEASE / "model.npz").is_file() else ""
    if not adapter:
        return None
    factory = ADAPTER_FACTORIES.get(adapter)
    if factory is None:
        log.error("Unsupported FORGERYLENS_ADAPTER setting")
        return None
    try:
        predictor = factory()
        if not isinstance(predictor.version, str) or not predictor.version.strip():
            raise ValueError("missing model version")
        if not callable(predictor.predict):
            raise ValueError("missing predictor method")
    except Exception:
        log.exception("Model adapter failed to load")
        return None
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = load_predictor()
    app.state.analysis_admission = AnalysisAdmission()
    yield


app = FastAPI(title="ForgeryLens", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in os.getenv("FORGERYLENS_ORIGINS", "").split(",") if item.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.mount("/assets", StaticFiles(directory=PUBLIC), name="assets")

if os.getenv("FORGERYLENS_FROZEN_RECEIPT") == "1":
    from .receipt_preview import router as receipt_router

    app.include_router(receipt_router)

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


@app.get("/api/v1/ready", include_in_schema=False)
async def readiness():
    if app.state.predictor is None:
        raise HTTPException(503, "The configured release model is unavailable.")
    return {"status": "ready"}


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
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(503, "Analysis is unavailable until a trained model is installed.")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() not in (
        "image/jpeg", "image/png", "image/webp", "application/octet-stream"
    ):
        raise HTTPException(415, "Send the encoded image as the request body.")
    admission = request.app.state.analysis_admission
    admission.check_rate(request.client.host if request.client else "")
    async with admission.slot():
        content = bytearray()
        try:
            with anyio.fail_after(UPLOAD_TIMEOUT_SECONDS):
                async for chunk in request.stream():
                    if len(content) + len(chunk) > MAX_BYTES:
                        raise HTTPException(413, "Use an image no larger than 8 MiB.")
                    content.extend(chunk)
        except (TimeoutError, ClientDisconnect):
            raise HTTPException(408, "The image upload timed out or was interrupted.") from None
        if not content:
            raise HTTPException(400, "The image is empty.")
        return await run_in_threadpool(analyze_document, predictor, bytes(content))


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
@app.get("/studio/single", include_in_schema=False)
@app.get("/research", include_in_schema=False)
@app.get("/methodology", include_in_schema=False)
@app.get("/about", include_in_schema=False)
@app.get("/privacy", include_in_schema=False)
def frontend():
    return FileResponse(PUBLIC / "index.html")


REFERENCE_RESEARCH = os.getenv("FORGERYLENS_REFERENCE_RESEARCH") == "1"
REFERENCE_STUDIO = ROOT / "web" / "reference_studio.html"

if REFERENCE_RESEARCH:
    from .reference_research import install, local_research, studio_response

    install(app, REFERENCE_STUDIO)


@app.get("/studio", include_in_schema=False)
def studio(request: Request):
    if REFERENCE_RESEARCH and local_research(request):
        return studio_response(REFERENCE_STUDIO)
    return frontend()
