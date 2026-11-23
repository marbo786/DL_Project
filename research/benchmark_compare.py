"""Measure the complete loopback HTTP comparison using the fictional Studio pairs."""

import argparse
import json
import platform
from io import BytesIO
from pathlib import Path
from time import perf_counter
from urllib.request import Request, ProxyHandler, build_opener
from uuid import uuid4

import numpy as np
from PIL import Image


def multipart(reference, candidate):
    boundary = uuid4().hex
    parts = []
    for name, image in (("reference", reference), ("candidate", candidate)):
        prefix = (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                  'filename="fictional.png"\r\nContent-Type: image/png\r\n\r\n')
        parts.extend([prefix.encode(), image, b"\r\n"])
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def summarize(values):
    return {"median": round(float(np.median(values)), 2), "p95": round(float(np.percentile(values, 95)), 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two repetitions.")
    folder = Path(__file__).resolve().parents[1] / "web" / "public" / "comparison-examples"
    fixtures = {p.stem: p.read_bytes() for p in folder.glob("*.png")}
    reference = fixtures["north-quay"]
    def larger(encoded):
        with Image.open(BytesIO(encoded)) as image:
            scaled = image.resize((image.width * 2, image.height * 2), Image.Resampling.BICUBIC)
            output = BytesIO()
            scaled.save(output, "PNG")
            return output.getvalue()

    cases = {
        "unchanged": (reference, reference),
        "localized_edit": (reference, fixtures["changed-total"]),
        "severe_misalignment": (reference, fixtures["misaligned"]),
        "different_reference": (fixtures["different-reference"], reference),
        "localized_edit_2x": (larger(reference), larger(fixtures["changed-total"])),
    }
    opener = build_opener(ProxyHandler({}))
    address = f"http://127.0.0.1:{args.port}/api/v1/compare"
    report = {"platform": platform.platform(), "repeats_per_case": args.repeats, "fixtures": {}}
    all_times = []
    for name, pair in cases.items():
        body, media = multipart(*pair)
        request = Request(address, data=body, headers={"Content-Type": media})
        elapsed, internal, outcomes = [], [], set()
        for index in range(args.repeats + 1):
            started = perf_counter()
            with opener.open(request, timeout=40) as response:
                raw = response.read()
            latency = (perf_counter() - started) * 1000
            payload = json.loads(raw)
            if index:
                elapsed.append(latency)
                internal.append(payload["inference_ms"])
                outcomes.add((payload["decision"], payload["score"], payload["reason"]))
        all_times.extend(elapsed)
        report["fixtures"][name] = {
            "http_ms": summarize(elapsed), "decode_through_previews_ms": summarize(internal),
            "outcomes": sorted(outcomes, key=str), "body_bytes": len(body), "response_bytes": len(raw),
        }
    report["all_http_ms"] = summarize(all_times)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
