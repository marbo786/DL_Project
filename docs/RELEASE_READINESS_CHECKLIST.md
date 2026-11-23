# Phase H release-readiness checklist

## Candidate scope

The comparison base was `main` at `1fcf29cddf227e2042314d9eef4234f80067f3db`. The verified Phase G starting commit was `02d461b98e8f681df8547820d32e59ebe19e02ab`, eight commits ahead with no commits behind. The incoming work comprises research phases, the local workbench, tests, documentation, and deployment configuration. Review Render implications before merging. CI validates the candidate only and publishes no image.

| Check | Evidence / release gate |
| --- | --- |
| Branch diff and packaged files | Review the compare and PR files for data, checkpoints, credential paths and unrelated changes. CI exports the effective Docker context, checks an exact file allowlist and scans included text for common credential signatures. |
| Public image | Require a green Linux job that builds and starts the public Dockerfile, checks installed dependencies and readiness, and verifies historical inference, Chromium interaction and 404 responses for compare routes and assets. |
| Opt-in research image | Require the same job to build the QA-only overlay, execute the full Python suite in it, start the loopback research service, and exercise the paired Studio in Chromium. The production build excludes this overlay. |
| Frozen comparison policy | Source hash verification plus the altered-total regression. Expected document score 0.1062 against threshold 0.5850, with a visible region and a no-flag decision. |
| Presentation behavior | Wrong-reference identity warning, absten­tion without evidence, and localized evidence displayed independently from the document decision. Desktop, tablet and mobile are included in browser regression checks. |
| Public exposure | Public uploads remain out of scope. Keep research mode off in Render and leave the service deployment unchanged while the PR is under review. |
| Independent capture coverage | External research dependency. Rights-cleared, source-disjoint captures are still needed for the frozen evaluation. [Capture guide](REAL_CAPTURE_COLLECTION_GUIDE.md) |
| Academic handoff | Course-owned submission format and human authorship sign-off remain with the submission team. Preserve the phase reports’ recorded AI assistance and use the [presentation outline](PRESENTATION_OUTLINE.md) as a rehearsal scaffold. |

## Reproduce the container gate

On a Linux checkout with a working Docker engine, Python 3.12, and Chromium prerequisites, run:

```bash
python -m pip install playwright==1.58.0
python -m playwright install --with-deps chromium
bash tests/run_container_qa.sh
```

The script uses only the public image context and tracked QA source and fixtures. It removes its containers on exit. Check the [candidate Actions runs](https://github.com/marbo786/DL_Project/actions/workflows/release-qa.yml?query=branch%3Aphase-a-controls) for the exact commit. Local Windows CLI attempts encountered missing Docker engine pipes and a Buildx access error. A successful native or CI test run must be recorded separately from these attempts.
