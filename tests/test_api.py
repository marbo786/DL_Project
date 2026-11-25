from io import BytesIO

import anyio
import httpx
from PIL import Image
import pytest

from api import main


pytestmark = pytest.mark.anyio


class ContractFixture:
    """Test-only predictor used to exercise transport and rendering contracts."""

    version = "test-fixture"

    def predict(self, image):
        assert image.mode == "RGB"
        return {
            "decision": "inconclusive",
            "regions": [{"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}],
            "heatmap": {"width": 2, "height": 1, "values": [0.0, 1.0]},
        }


@pytest.fixture
async def client(monkeypatch, asgi_client):
    monkeypatch.setenv("FORGERYLENS_ADAPTER", "")
    monkeypatch.setenv("FORGERYLENS_EXPERIMENTS_DIR", "")
    async with asgi_client(main.app) as connection:
        yield connection


def sample(client, content, **headers):
    return client.post("/api/v1/analyze", content=content, headers={"Content-Type": "image/png", **headers})


def png(sample_color="white", second_color=None):
    buffer = BytesIO()
    image = Image.new("RGB", (32, 16), sample_color)
    if second_color is None:
        image.save(buffer, format="PNG")
    else:
        image.save(buffer, format="PNG", save_all=True,
                   append_images=[Image.new("RGB", image.size, second_color)])
    return buffer.getvalue()


async def test_missing_model_is_explicit(client):
    assert (await client.get("/api/v1/health")).json() == {"status": "ok", "model_ready": False}
    assert (await client.get("/api/v1/ready")).status_code == 503
    assert (await client.get("/api/v1/model-info")).json()["available"] is False
    assert (await sample(client, b"fixture")).status_code == 503
    assert (await client.get("/api/v1/experiments/summary")).json()["status"] == "pending"


async def test_routes_and_assets(client):
    for route in ("/", "/studio", "/research", "/methodology", "/about", "/privacy"):
        assert "ForgeryLens" in (await client.get(route)).text
    assert (await client.get("/assets/original.png")).content.startswith(b"\x89PNG")
    assert (await client.get("/studio/single")).status_code == 200
    assert (await client.get("/missing")).status_code == 404


async def test_transport_accepts_validated_prediction(client):
    main.app.state.predictor = ContractFixture()
    response = await sample(client, png())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["rejected"] is True
    assert result["image_width"] == 32
    assert result["regions"][0]["width"] == 0.5
    assert result["heatmap"]["values"] == [0.0, 1.0]


async def test_reject_invalid_input_and_size(client, monkeypatch):
    main.app.state.predictor = ContractFixture()
    assert (await client.post("/api/v1/analyze", content=png(),
                              headers={"Content-Type": "application/pdf"})).status_code == 415
    assert (await sample(client, b"not a png")).status_code == 415
    assert (await sample(client, png("white", "black"))).status_code == 415
    assert (await sample(client, b"")).status_code == 400

    async def chunks():
        yield b"0" * 6
        yield b"0" * 5

    with monkeypatch.context() as config:
        config.setattr(main, "MAX_BYTES", 10)
        assert (await sample(client, chunks())).status_code == 413
    assert (await sample(client, png())).status_code == 200


async def test_rate_limit_is_per_peer_and_ignores_forwarding_header(client, monkeypatch):
    main.app.state.predictor = ContractFixture()
    monkeypatch.setattr(main, "RATE_LIMIT", 1)
    monkeypatch.setattr(main, "monotonic", lambda: 0)
    assert (await sample(client, png())).status_code == 200
    limited = await sample(client, png(), **{"X-Forwarded-For": "192.0.2.1"})
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == str(main.RATE_WINDOW_SECONDS)
    assert limited.headers["Cache-Control"] == "no-store"
    transport = httpx.ASGITransport(app=main.app, client=("192.0.2.2", 40000))
    async with httpx.AsyncClient(transport=transport, base_url=client.base_url) as other:
        assert (await sample(other, png())).status_code == 200


async def test_queue_accepts_a_waiter_and_rejects_overflow(client, monkeypatch):
    main.app.state.predictor = ContractFixture()
    monkeypatch.setattr(main, "MAX_QUEUED_REQUESTS", 1)
    started, finish = anyio.Event(), anyio.Event()
    order = []

    async def analyze(function, predictor, content):
        order.append(content)
        started.set()
        await finish.wait()
        return function(predictor, content)

    async def submit(content):
        assert (await sample(client, content)).status_code == 200

    monkeypatch.setattr(main, "run_in_threadpool", analyze)
    first, next_image = png("white"), png("black")
    async with anyio.create_task_group() as group:
        with anyio.fail_after(5):
            group.start_soon(submit, first)
            await started.wait()
            group.start_soon(submit, next_image)
            await anyio.wait_all_tasks_blocked()
            try:
                overflow = await sample(client, png("red"))
                assert overflow.status_code == 503
                assert int(overflow.headers["Retry-After"]) > 0
                assert order == [first]
            finally:
                finish.set()
    assert order == [first, next_image]
    assert (await sample(client, png("red"))).status_code == 200


async def test_queue_and_upload_timeouts_release_capacity(client, monkeypatch):
    main.app.state.predictor = ContractFixture()
    admission = main.app.state.analysis_admission
    monkeypatch.setattr(main, "QUEUE_WAIT_SECONDS", .01)
    async with admission.slot():
        timed_out = await sample(client, png())
        assert timed_out.status_code == 503
        assert "queue wait expired" in timed_out.json()["detail"]
        assert admission.pending == 1

    async def stalled_upload():
        yield b"partial"
        await anyio.sleep_forever()

    monkeypatch.setattr(main, "UPLOAD_TIMEOUT_SECONDS", .01)
    assert (await sample(client, stalled_upload())).status_code == 408
    assert (await sample(client, png())).status_code == 200


async def test_report_reading(client, monkeypatch, tmp_path):
    monkeypatch.setenv("FORGERYLENS_EXPERIMENTS_DIR", str(tmp_path))
    (tmp_path / "summary.json").write_text('{"run_id": "test-run"}', encoding="utf-8")
    assert (await client.get("/api/v1/experiments/summary")).json()["artifact"]["run_id"] == "test-run"
    assert (await client.get("/api/v1/experiments/other")).status_code == 404


@pytest.mark.artifacts("release/model.npz", "release/summary.json")
async def test_bundled_inference_through_api(monkeypatch, asgi_client):
    monkeypatch.delenv("FORGERYLENS_ADAPTER", raising=False)
    monkeypatch.delenv("FORGERYLENS_EXPERIMENTS_DIR", raising=False)
    async with asgi_client(main.app) as client:
        assert (await client.get("/api/v1/health")).json()["model_ready"]
        assert (await client.get("/api/v1/experiments/summary")).json()["status"] == "available"
        result = await sample(client, (main.PUBLIC / "edited.png").read_bytes())
        assert result.status_code == 200, result.text
        assert result.json()["heatmap"] is not None
        assert result.json()["calibrated_probability"] is not None
