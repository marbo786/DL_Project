import json
from hashlib import sha256
from pathlib import Path

import torch

from api.receipt_preview import detector, score_receipt


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    frozen = detector()
    samples = {}
    root = Path(__file__).resolve().parents[3]
    for name in ("original", "edited", "blurred"):
        content = (root / "web/public" / f"{name}.png").read_bytes()
        score_receipt(content, frozen)
        result = score_receipt(content, frozen)
        result["analysis_id"] = "fixture-" + name
        samples[name] = {"sha256": sha256(content).hexdigest(), "result": result}
    output = root / "web/frontend/data/sample-results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(samples, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
