import pytest
from fastapi import HTTPException

from api import main


def test_rate_window_expires_and_client_table_is_bounded(monkeypatch):
    now = 100.0
    monkeypatch.setattr(main, "monotonic", lambda: now)
    monkeypatch.setattr(main, "MAX_RATE_CLIENTS", 2)
    monkeypatch.setattr(main, "RATE_LIMIT", 1)
    admission = main.AnalysisAdmission()
    admission.check_rate("192.0.2.1")
    admission.check_rate("192.0.2.2")
    with pytest.raises(HTTPException) as capacity:
        admission.check_rate("192.0.2.3")
    assert capacity.value.status_code == 503
    assert len(admission.clients) == 2

    now += main.RATE_WINDOW_SECONDS - .25
    with pytest.raises(HTTPException) as limited:
        admission.check_rate("192.0.2.1")
    assert limited.value.status_code == 429
    assert limited.value.headers["Retry-After"] == "1"

    now += .25
    admission.check_rate("192.0.2.3")
    assert list(admission.clients) == ["192.0.2.3"]


def test_adapter_accepts_only_the_bundled_factory(monkeypatch, tmp_path, caplog):
    class Predictor:
        version = "fixture"

        def predict(self, image):
            return image

    predictor = Predictor()
    calls = []

    def factory():
        calls.append(True)
        return predictor

    monkeypatch.setitem(main.ADAPTER_FACTORIES, main.DEFAULT_ADAPTER, factory)
    monkeypatch.setattr(main, "RELEASE", tmp_path)
    monkeypatch.setenv("FORGERYLENS_ADAPTER", "api.signal:Predictor")
    assert main.load_predictor() is None
    assert not calls
    assert "Unsupported FORGERYLENS_ADAPTER" in caplog.text

    monkeypatch.setenv("FORGERYLENS_ADAPTER", main.DEFAULT_ADAPTER)
    assert main.load_predictor() is predictor
    monkeypatch.setenv("FORGERYLENS_ADAPTER", "")
    assert main.load_predictor() is None
    monkeypatch.delenv("FORGERYLENS_ADAPTER")
    assert main.load_predictor() is None
    (tmp_path / "model.npz").touch()
    assert main.load_predictor() is predictor
    assert len(calls) == 2
