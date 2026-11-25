from io import BytesIO

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
    main.recent_requests.clear()
    async with asgi_client(main.app) as connection:
        yield connection


def sample(client, content):
    return client.post("/api/v1/analyze", content=content, headers={"Content-Type": "image/png"})


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
    buffer = BytesIO()
    Image.new("RGB", (32, 16)).save(buffer, format="PNG")
    response = await sample(client, buffer.getvalue())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["rejected"] is True
    assert result["image_width"] == 32
    assert result["regions"][0]["width"] == 0.5
    assert result["heatmap"]["values"] == [0.0, 1.0]


async def test_reject_invalid_input_and_size(client, monkeypatch):
    main.app.state.predictor = ContractFixture()
    assert (await sample(client, b"not a png")).status_code == 415
    assert (await sample(client, b"")).status_code == 400
    monkeypatch.setattr(main, "MAX_BYTES", 10)
    assert (await sample(client, b"0" * 11)).status_code == 413
    assert main.active_requests == 0


async def test_report_reading(client, monkeypatch, tmp_path):
    monkeypatch.setenv("FORGERYLENS_EXPERIMENTS_DIR", str(tmp_path))
    (tmp_path / "summary.json").write_text('{"run_id": "test-run"}', encoding="utf-8")
    assert (await client.get("/api/v1/experiments/summary")).json()["artifact"]["run_id"] == "test-run"
    assert (await client.get("/api/v1/experiments/other")).status_code == 404


@pytest.mark.artifacts("release/model.npz", "release/summary.json")
async def test_bundled_inference_through_api(monkeypatch, asgi_client):
    monkeypatch.delenv("FORGERYLENS_ADAPTER", raising=False)
    monkeypatch.delenv("FORGERYLENS_EXPERIMENTS_DIR", raising=False)
    main.recent_requests.clear()
    async with asgi_client(main.app) as client:
        assert (await client.get("/api/v1/health")).json()["model_ready"]
        assert (await client.get("/api/v1/experiments/summary")).json()["status"] == "available"
        result = await sample(client, (main.PUBLIC / "edited.png").read_bytes())
        assert result.status_code == 200, result.text
        assert result.json()["heatmap"] is not None
        assert result.json()["calibrated_probability"] is not None
