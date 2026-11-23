"""Check the effective Docker context exported by BuildKit."""

import re
import sys
from pathlib import Path


REQUIRED = {
    "api/__init__.py", "api/contracts.py", "api/main.py", "api/requirements.txt", "api/signal.py",
    "release/model.npz", "release/summary.json", "release/ablations.json",
    "release/calibration.json", "release/robustness.json",
    "web/public/index.html", "web/public/studio.js", "web/public/studio.css", "web/public/specimen.json",
    "web/public/blurred.png", "web/public/edited.png", "web/public/original.png",
    "web/public/failure-115.png", "web/public/failure-17.png", "web/public/failure-27.png",
    "web/public/failure-69.png", "web/public/failure-73.png",
}
OPTIONAL = {"Dockerfile", ".dockerignore"}
SIGNATURE = re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
                       r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")


def audit(root):
    paths = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    assert REQUIRED <= paths, sorted(REQUIRED - paths)
    assert paths <= REQUIRED | OPTIONAL, sorted(paths - REQUIRED - OPTIONAL)
    for name in paths:
        file = root / name
        assert not file.is_symlink(), name
        if file.suffix not in {".png", ".npz"}:
            assert not SIGNATURE.search(file.read_text(encoding="utf-8")), f"Credential pattern in {name}"
    print(f"Public context allowlist passed ({len(paths)} files).")


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
