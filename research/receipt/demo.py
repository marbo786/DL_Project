"""Local demonstration of the frozen single-image receipt detector."""

from dataclasses import dataclass
from io import BytesIO
import json
from math import exp, isfinite
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import torch

from research.manifest import file_sha256
from research.receipt.model import load_detector, predict


ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = Path("artifacts/receipt-study-v1/residual-42")
LOCK_PATH = Path("artifacts/receipt-retrospective-v1/freeze.json")
LOCK_SHA256 = "d65421f5457ad6ecb91747b1ed6cd2fb4a86ea0c40a023124290a8239d6688e0"
POLICY_VERSION = "receipt-preview-v1"
CONFIDENCE_CUTOFF = .55
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
FORMATS = ("PNG", "JPEG", "WEBP")


@dataclass(frozen=True)
class ReceiptResult:
    logit: float
    probability: float
    confidence: float
    decision: str
    advice: str
    tile: tuple[int, int, int, int]


def route_receipt(logit, probability, review_cutoff):
    confidence = max(probability, 1 - probability)
    if confidence < CONFIDENCE_CUTOFF:
        return "Inconclusive", "Confidence is below the demo gate. Use standard manual review."
    if logit > review_cutoff:
        return "Prioritize review", "Inspect the marked tile and verify the receipt against transaction records before escalating."
    if probability < .5:
        return "Normal queue", "Follow the standard reimbursement checklist and confirm the transaction as usual."
    return "Inconclusive", "The score does not meet either routing criterion. Use standard manual review."


def decode_upload(content):
    if not 0 < len(content) <= MAX_BYTES:
        raise ValueError("Choose a nonempty image no larger than 8 MiB.")
    try:
        with Image.open(BytesIO(content), formats=FORMATS) as source:
            if source.width * source.height > MAX_PIXELS or getattr(source, "is_animated", False):
                raise ValueError("Choose a single-frame image of at most 16 megapixels.")
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as error:
        raise ValueError("Choose a readable single-frame JPEG, PNG, or WebP image.") from error


class FrozenReceiptDemo:
    def __init__(self, root=ROOT):
        folder = root / RUN_PATH
        lock_path = root / LOCK_PATH
        if file_sha256(lock_path) != LOCK_SHA256:
            raise ValueError("The receipt freeze file does not match the demo lock digest.")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        fit = json.loads((folder / "fit.json").read_text(encoding="utf-8"))
        if (lock["configuration"], lock["seed"], lock["checkpoint_sha256"]) != (
            fit.get("mode"), fit.get("seed"), fit.get("checkpoint_sha256")
        ):
            raise ValueError("The receipt checkpoint metadata does not match the frozen recipe.")
        if fit.get("batch_size") != 64:
            raise ValueError("The receipt tile batch size does not match the frozen recipe.")
        if file_sha256(folder / "calibration.csv") != lock["calibration_sha256"]:
            raise ValueError("The receipt calibration table does not match the demo lock.")
        self.checkpoint_sha256 = lock["checkpoint_sha256"]
        self.review_cutoff = lock["operating_points"]["0.05"]
        self.slope = lock["calibration"]["slope"]
        self.intercept = lock["calibration"]["intercept"]
        self.batch_size = fit["batch_size"]
        self.model = load_detector(folder)

    def score(self, image):
        logit, tile = predict(self.model, image, self.batch_size)
        if not isfinite(logit):
            raise ValueError("The model could not produce a finite score. Use standard manual review.")
        scaled = max(-60., min(60., self.slope * logit + self.intercept))
        probability = 1 / (1 + exp(-scaled))
        confidence = max(probability, 1 - probability)
        decision, advice = route_receipt(logit, probability, self.review_cutoff)
        return ReceiptResult(logit, probability, confidence, decision, advice, tile)


def main():
    import streamlit as st

    st.set_page_config(page_title="ForgeryLens Receipt Review", layout="wide")
    st.title("ForgeryLens Receipt Review")
    st.write("Research preview for university reimbursement review. Route actual claims through standard manual review.")

    @st.cache_resource
    def load_demo():
        torch.set_num_threads(2)
        return FrozenReceiptDemo()

    try:
        demo = load_demo()
    except (OSError, RuntimeError, ValueError):
        st.error("Frozen receipt model unavailable. Check the local checkpoint, calibration table, and freeze file.")
        st.stop()

    with st.expander("Model and routing policy"):
        st.write(f"Residual MobileNetV3-Small, seed 42. Policy `{POLICY_VERSION}`.")
        st.code(demo.checkpoint_sha256, language=None)
        st.write(f"Confidence gate {CONFIDENCE_CUTOFF:.0%}. Priority review requires a logit above {demo.review_cutoff:.6f}.")
        st.write("This research gate deferred every receipt in the saved development set. Open-set rejection is unvalidated.")

    with st.form("inspect_receipt"):
        upload = st.file_uploader("Authorized receipt image", type=("png", "jpg", "jpeg", "webp"))
        submitted = st.form_submit_button("Analyze receipt")
    if not submitted:
        return
    if upload is None:
        st.warning("Choose a receipt image first.")
        return

    try:
        image = decode_upload(upload.getvalue())
        with st.spinner("Scoring receipt tiles"):
            result = demo.score(image)
    except (RuntimeError, ValueError) as error:
        st.subheader("Inconclusive")
        st.warning(str(error) if isinstance(error, ValueError) else "Scoring failed. Use standard manual review.")
        return

    preview, findings = st.columns((3, 2))
    with preview:
        x0, y0, x1, y1 = result.tile
        ImageDraw.Draw(image).rectangle((x0, y0, x1 - 1, y1 - 1), outline="#ca6800", width=3)
        st.image(image, caption="Highest-scoring tile. Use it as an inspection cue.", width="stretch")
    with findings:
        st.subheader(result.decision)
        st.metric("Estimated edit probability", f"{result.probability:.2%}")
        st.metric("Confidence in the binary prediction", f"{result.confidence:.2%}")
        st.write(result.advice)


if __name__ == "__main__":
    main()
