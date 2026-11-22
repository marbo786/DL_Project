from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from api.contracts import Prediction
from api.signal import Predictor, regions

ROOT = Path(__file__).resolve().parents[1]


def test_normalized_regions():
    grid = np.zeros((8, 4))
    grid[2:4, 1:3] = 0.8
    grid[7, 3] = 0.9
    assert regions(grid) == [{"x": .25, "y": .25, "width": .5, "height": .25}]


@pytest.fixture
def trained():
    model_path = ROOT / "release" / "model.npz"
    if not model_path.exists():
        pytest.skip("Build the measured release artifact before inference tests.")
    return Predictor(model_path)


def test_unknown_documents_abstain(trained):
    result = Prediction.model_validate(trained.predict(Image.new("RGB", (120, 120), "#aaa")))
    assert result.decision == "inconclusive"
    assert result.heatmap is None
    assert result.calibrated_probability is None


def test_real_trained_predictor_on_demo_assets(trained):
    for filename in ("original.png", "edited.png", "blurred.png"):
        with Image.open(ROOT / "web" / "public" / filename) as image:
            result = Prediction.model_validate(trained.predict(image))
        assert result.heatmap is not None
        if filename == "blurred.png":
            assert result.decision == "inconclusive"
            assert result.calibrated_probability is None
        else:
            assert 0 <= result.calibrated_probability <= 1