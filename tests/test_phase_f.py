from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from fastapi import FastAPI

cv2 = pytest.importorskip("cv2")

from research.reference_verification import POLICY, PRIMARY, Result, compare, verify_freeze
from research.reference_verification_dataset import audit
from research.reference_verification_evaluate import evaluate, summarize


def encoded(image):
    stream = BytesIO()
    Image.fromarray(image).save(stream, "PNG")
    return stream.getvalue()


def texture():
    rng = np.random.default_rng(23)
    image = np.full((320, 256, 3), 239, dtype=np.uint8)
    for x, y, radius in rng.integers([14, 14, 2], [242, 306, 8], (130, 3)):
        cv2.circle(image, (int(x), int(y)), int(radius), (23, 23, 23), -1)
    return image


@pytest.fixture
def inventory(tmp_path):
    image = texture()
    target = np.zeros(image.shape[:2], np.uint8)
    target[120:150, 112:145] = 255
    captures = {"reference": image, "new_scan": np.roll(image, 2, axis=1),
                "edit": np.where(target[..., None], 85, image).astype(np.uint8),
                "revision": np.where(target[..., None], 155, image).astype(np.uint8),
                "unrelated": np.full_like(image, 200)}

    def write(name, array):
        content = encoded(array)
        filename = f"{name}.png"
        (tmp_path / filename).write_bytes(content)
        return {"path": filename, "sha256": sha256(content).hexdigest()}

    mask = write("mask", target)
    images = []
    for name, array in captures.items():
        lineage = "two" if name == "unrelated" else "one"
        images.append(dict(write(name, array), id=name, capture_id=name, capture_session=name,
                           document_id=lineage, source_group=lineage, acquisition="scanner",
                           license="test fixture", provenance="fabricated bytes for unit testing"))
    pairs = []
    for scenario, candidate in [("recapture", "new_scan"), ("localized_edit", "edit"),
                                ("legitimate_revision", "revision"), ("wrong_reference", "unrelated")]:
        pair = dict(id=scenario, reference="reference", candidate=candidate, scenario=scenario,
                    annotation="fabricated fixture label")
        if scenario in {"localized_edit", "legitimate_revision"}:
            pair["mask"] = mask.copy()
        pairs.append(pair)
    return dict(schema_version=1, dataset_version="fixture",
                review=dict(reviewer="fixture", review_date="fixture", rights_record="fixture",
                            source_crosswalk_record="fixture", rights_verified=True,
                            source_independence_verified=True, pairing_verified=True),
                images=images, pairs=pairs)


def audit_fixture(tmp_path, data, excluded=frozenset()):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return audit(path, tmp_path, excluded)


def test_frozen_threshold_and_abstention():
    verify_freeze()
    assert POLICY["primary_nominal_fpr"] == "5%"
    assert Result(None, score=PRIMARY).decision == "no_difference_flagged"
    assert Result(None, score=np.nextafter(PRIMARY, 1)).decision == "difference_detected"
    blank = Image.new("RGB", (256, 320), "white")
    result = compare(blank, blank)
    assert result.decision == "inconclusive"
    assert result.score is None and result.mask is None and result.heatmap is None


def test_audit_and_evaluate_fixture_in_candidate_coordinates(tmp_path, inventory):
    data, images, counts = audit_fixture(tmp_path, inventory)
    assert sum(counts.values()) == 4
    rows = evaluate(data, images, tmp_path)
    assert rows[0]["decision"] == "no_difference_flagged"
    assert rows[1]["iou"] > 0.4
    assert rows[3]["score"] is None
    metrics = summarize(rows)
    assert metrics["explicit_mismatch_detection"] == "unsupported"
    assert metrics["strata"]["physical"]["classification"]["accepted_only_auroc"] == 1
    assert metrics["strata"]["physical"]["scenarios"]["wrong_reference"]["abstention"] == 1


@pytest.mark.parametrize(("mutation", "error"), [
    (lambda d: d["images"][1].update(acquisition="simulated"), "physical capture sessions"),
    (lambda d: d["images"][1].update(capture_session="reference"), "physical capture sessions"),
    (lambda d: d["images"][1].update(source_group="two"), "spans source groups"),
    (lambda d: d["pairs"][-1].update(candidate="revision"), "Invalid or duplicated pair"),
    (lambda d: d["images"][0].update(path="../outside.png"), "escapes the data root"),
    (lambda d: d["images"][0].update(sha256="bad"), "SHA-256 mismatch"),
    (lambda d: d["review"].update(rights_verified=False), "Complete rights"),
    (lambda d: d["pairs"][1]["mask"].update(d["images"][0] | {
        "path": d["images"][0]["path"], "sha256": d["images"][0]["sha256"]}), "grayscale PNG"),
])
def test_audit_rejects_unverifiable_inputs(tmp_path, inventory, mutation, error):
    mutation(inventory)
    with pytest.raises(ValueError, match=error):
        audit_fixture(tmp_path, inventory)


def test_audit_rejects_prior_source(tmp_path, inventory):
    with pytest.raises(ValueError, match="overlapping prior source"):
        audit_fixture(tmp_path, inventory, {"one"})


def test_mismatch_score_is_not_an_identity_decision():
    metrics = summarize([
        dict(scenario="wrong_reference", mode="physical", candidate_group="b", score=.99, ms=1),
        dict(scenario="localized_edit", mode="physical", candidate_group="a", score=None, ms=2, iou=None),
        dict(scenario="recapture", mode="physical", candidate_group="a", score=.2, ms=3),
    ])
    physical = metrics["strata"]["physical"]
    assert physical["scenarios"]["wrong_reference"]["coverage"] == 1
    assert metrics["explicit_mismatch_detection"] == "unsupported"
    assert physical["classification"]["frozen_operating_points"]["5%"]["recall_all_edits"] == 0


@pytest.mark.anyio
async def test_research_routes_opt_in_and_suppress_unreliable_previews(asgi_client):
    pytest.importorskip("python_multipart")
    from api import main
    from api.reference_research import install

    assert "/api/v1/compare" not in {route.path for route in main.app.routes}
    app = FastAPI()
    install(app, Path(__file__).resolve().parents[1] / "web" / "reference_studio.html")
    blank = encoded(np.full((320, 256, 3), 210, dtype=np.uint8))
    original = encoded(texture())
    async with asgi_client(app) as client:
        assert "Reference comparison" in (await client.get("/studio/compare")).text
        partial = await client.post("/api/v1/compare", files={"reference": ("a.png", original)})
        assert partial.status_code == 400
        for candidate, expected in [(blank, "inconclusive"), (original, "no_difference_flagged")]:
            response = await client.post(
                "/api/v1/compare", files={"reference": ("a.png", original), "candidate": ("b.png", candidate)}
            )
            assert response.status_code == 200, response.text
            assert response.headers["cache-control"] == "no-store"
            assert response.json()["decision"] == expected
            assert ("previews" in response.json()) == (expected != "inconclusive")
