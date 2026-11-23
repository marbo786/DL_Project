#!/usr/bin/env bash
set -euo pipefail

temp="$(mktemp -d)"
public_name="forgerylens-public-qa-$$"
research_name="forgerylens-research-qa-$$"
cleanup() {
  for name in "$public_name" "$research_name"; do
    if docker container inspect "$name" >/dev/null 2>&1; then
      docker logs "$name" || true
      docker rm -f "$name" >/dev/null || true
    fi
  done
  rm -rf -- "$temp"
}
trap cleanup EXIT

# Export what Docker actually admits after applying the public .dockerignore.
printf 'FROM scratch\nCOPY . /\n' |
  docker build --file - --output "type=local,dest=$temp/public" .
python tests/audit_public_context.py "$temp/public"
docker build --tag forgerylens:release-qa .
docker run --rm --entrypoint python forgerylens:release-qa -m pip check
docker run --rm --entrypoint python forgerylens:release-qa -c \
  "from importlib.util import find_spec; assert not any(find_spec(name) for name in ('torch', 'cv2', 'python_multipart', 'research'))"
docker run -d --name "$public_name" -p 127.0.0.1:8780:8000 forgerylens:release-qa
curl --fail --silent --show-error --retry 20 --retry-connrefused --retry-delay 1 http://127.0.0.1:8780/api/v1/ready
python tests/smoke.py http://127.0.0.1:8780 --image web/public/edited.png --public-profile
python tests/browser_smoke.py http://127.0.0.1:8780

# The QA-only overlay uses tracked research/test files and is never published.
mkdir "$temp/research"
git archive HEAD api research tests pytest.ini web/reference_studio.html web/public/reference_studio.css \
  web/public/reference_studio.js web/public/comparison-examples | tar -xf - -C "$temp/research"
docker build --file tests/Dockerfile.qa --tag forgerylens:research-qa "$temp/research"
docker run --rm --entrypoint python forgerylens:research-qa -m pytest -q -p no:cacheprovider
docker run -d --name "$research_name" --network host -e FORGERYLENS_REFERENCE_RESEARCH=1 \
  --entrypoint python forgerylens:research-qa -m uvicorn api.main:app \
  --host 127.0.0.1 --port 8781 --no-proxy-headers --no-access-log
curl --fail --silent --show-error --retry 20 --retry-connrefused --retry-delay 1 http://127.0.0.1:8781/api/v1/ready
python tests/smoke.py http://127.0.0.1:8781 --image web/public/edited.png
python tests/browser_compare.py http://127.0.0.1:8781
