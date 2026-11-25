from base64 import b64decode
from io import BytesIO
from pathlib import Path
from threading import Event

import anyio
import numpy as np
import pytest
from fastapi import FastAPI
from PIL import Image

pytest.importorskip("cv2")
pytest.importorskip("python_multipart")

from api import reference_research


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "web" / "public" / "comparison-examples"
pytestmark = pytest.mark.anyio


@pytest.fixture
def comparison_app():
    app = FastAPI()
    reference_research.install(app, ROOT / "web" / "reference_studio.html")
    return app


@pytest.fixture
async def client(asgi_client, comparison_app):
    async with asgi_client(comparison_app) as connection:
        yield connection


def upload(reference=b"broken", candidate=b"broken"):
    return [("reference", ("ref.png", reference)), ("candidate", ("candidate.png", candidate))]


def compare(client, **kwargs):
    return client.post("/api/v1/compare", **kwargs)


async def test_research_endpoint_scope_and_invalid_media(client, asgi_client, comparison_app):
    assert (await compare(client, files=upload(), headers={"Origin": "https://example.org"})).status_code == 403
    assert (await compare(client, files=upload(), headers={"Host": "example.org"})).status_code == 404
    async with asgi_client(comparison_app, peer=("192.0.2.1", 40001)) as remote:
        assert (await compare(remote, files=upload())).status_code == 404
    assert (await compare(client, content=b"bad")).status_code == 415
    response = await compare(client, files=upload())
    assert response.status_code == 415
    assert response.headers["cache-control"] == "no-store"
    assert (await compare(client, files=upload(candidate=b""))).status_code == 400


async def test_multipart_duplicate_truncated_oversized_and_recovery(client, monkeypatch):
    assert (await compare(client, files=[upload()[0]] * 2)).status_code == 400
    headers = {"Content-Type": "multipart/form-data; boundary=123"}
    truncated = b'--123\r\nContent-Disposition: form-data; name="reference"; filename="a"\r\n\r\na'
    assert (await compare(client, content=truncated, headers=headers)).status_code == 400
    monkeypatch.setattr(reference_research, "MAX_BYTES", 64)
    assert (await compare(client, files=upload(reference=b"x" * 65))).status_code == 413
    assert (await compare(client, content=b"x" * 9000, headers=headers)).status_code == 413
    assert (await compare(client, files=upload())).status_code == 415


async def test_pixel_limit_before_conversion(client, monkeypatch):
    from research import reference_verification

    monkeypatch.setattr(reference_verification, "MAX_PIXELS", 100)
    raw = (EXAMPLES / "north-quay.png").read_bytes()
    response = await compare(client, files=upload(reference=raw, candidate=raw))
    assert response.status_code == 415
    assert "16 megapixels" in response.json()["detail"]


async def test_concurrent_comparison_is_rejected_and_slot_recovers(client, monkeypatch):
    started, release = anyio.Event(), Event()

    def held_comparison(*_):
        anyio.from_thread.run_sync(started.set)
        assert release.wait(timeout=10)
        return {"fixture": True}

    async def first_request():
        assert (await compare(client, files=upload())).status_code == 200

    monkeypatch.setattr(reference_research, "comparison_payload", held_comparison)
    async with anyio.create_task_group() as group:
        group.start_soon(first_request)
        try:
            with anyio.fail_after(5):
                await started.wait()
                assert (await compare(client, files=upload())).status_code == 429
        finally:
            release.set()
    assert (await compare(client, files=upload())).status_code == 200


async def test_raster_and_overlays_share_reference_coordinates(client):
    files = upload(reference=(EXAMPLES / "north-quay.png").read_bytes(),
                   candidate=(EXAMPLES / "changed-total.png").read_bytes())
    response = await compare(client, files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["alignment"]["supported"]
    images = {key: np.asarray(Image.open(BytesIO(b64decode(value.split(",", 1)[1]))))
              for key, value in data["previews"].items()}
    raster = data["raster"]
    assert raster["frame"] == "normalized_reference"
    assert all(image.shape[:2] == (raster["height"], raster["width"]) for image in images.values())
    invalid = images["valid_overlap"] == 0
    assert not images["heatmap"][invalid, 3].any()
    assert not images["regions"][invalid, 3].any()
    ys, xs = np.nonzero(images["regions"][:, :, 3])
    assert data["region_pixels"] == len(xs) > 0
    assert data["decision"] == "no_difference_flagged"
    assert np.median(xs) / raster["width"] > .65
    assert .60 < np.median(ys) / raster["height"] < .70
