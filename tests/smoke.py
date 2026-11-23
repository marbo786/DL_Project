"""Check local or hosted routes. An optional image requires live inference."""

import argparse
import json
import mimetypes
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def fetch(base, path, image=None):
    request = Request(base.rstrip("/") + path)
    if image:
        mime = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        request = Request(request.full_url, data=image.read_bytes(), headers={"Content-Type": mime})
    with urlopen(request, timeout=75) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--public-profile", action="store_true")
    args = parser.parse_args()
    for path in ("/", "/studio", "/research", "/methodology", "/about", "/privacy"):
        assert b"ForgeryLens" in fetch(args.url, path), path
    assert fetch(args.url, "/assets/original.png").startswith(b"\x89PNG")
    health = json.loads(fetch(args.url, "/api/v1/health"))
    assert health["status"] == "ok"
    if args.image:
        assert health["model_ready"], "A live model is required for image analysis."
        result = json.loads(fetch(args.url, "/api/v1/analyze", args.image))
        assert result["analysis_id"] and result["model_version"]
        assert json.loads(fetch(args.url, "/api/v1/ready")) == {"status": "ready"}
    if args.public_profile:
        for path, method in (("/studio/compare", "GET"), ("/api/v1/compare", "POST"),
                             ("/assets/reference_studio.js", "GET"),
                             ("/assets/comparison-examples/north-quay.png", "GET")):
            request = Request(args.url.rstrip("/") + path, method=method)
            try:
                with urlopen(request, timeout=15):
                    raise AssertionError(f"Public profile unexpectedly served {path}")
            except HTTPError as error:
                assert error.code == 404, (path, error.code)
    print("Smoke checks passed. Model ready:", health["model_ready"])


if __name__ == "__main__":
    main()
