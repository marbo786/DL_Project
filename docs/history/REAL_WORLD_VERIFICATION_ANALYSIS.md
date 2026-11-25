# Phase F independent evaluation of reference-assisted verification

## Status

The registered SSIM protocol is frozen. Independent evaluation is **blocked on data**. The repository and linked file search surfaced the AIForge-Doc pairs and the Phase E measurements, with no rights-cleared independent set covering the required comparison scenarios. The deduplicated AIForge manifest matched SHA-256 `3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80`. Only source identifiers from the manifest are used for the prior-source exclusion check. No sealed images, masks, or scores were opened for Phase F.

The [SmartDoc 2015 capture benchmark](https://sites.google.com/site/icdar15smartdoc/challenge-1/dataset) documents a CC BY 4.0 release with 30 printed document models recaptured in four settings and frame-level document quadrilaterals. This could support a separate registration or unchanged-recapture study after review of source identities and original asset terms. Its published annotations do not establish paired localized edits or documented revisions. Neighboring frames from a clip cannot serve as independent capture sessions. No SmartDoc samples were scored.

Phase E remains the relevant measured evidence. Registered SSIM reached 0.647 AUROC on ordinary simulated acquisition differences and 0.600 on the difficult simulated regime. Ordinary-transform recall was 6.7% at 4.9% FPR, with 0.104 pixel IoU. Those results are from the [Phase E analysis](REFERENCE_VERIFICATION_ANALYSIS.md). Phase F reports no measured independent AUROC, operating-point performance, localization, coverage, mismatch rejection, or timing.

## Frozen comparison specification

[Policy v1](../../research/phase_f_policy.json) fixes the Phase E comparator at commit `ce65ce68c34fb127ac59bc2a2cabd36cd001d21e`, captures its source digests, and records the saved calibration score digest. The only inputs are a trusted reference and a candidate. The caller establishes that they represent the same document. A score describes visual dissimilarity within the registered overlap. Document authorization, authenticity, identity, and fraudulent intent require separate evidence.

| Item | Frozen rule |
| --- | --- |
| Input | Single-frame JPEG, PNG, or WebP, up to 8 MiB and 16 megapixels per image. Orient by EXIF and convert to RGB. |
| Raster | Set the reference long side to 512 pixels. Resize both images to the reference raster with area interpolation. The unequal-size candidate adapter was defined before independent scoring. A native or resulting dimension below 24 pixels causes abstention. Invalid uploads return a validation error. |
| Registration | Phase E ORB matches and RANSAC homography. Retain the existing minimum correspondence/inlier, spread, reprojection, warp, and valid-overlap gates. Use only eroded valid overlap. Registration sees no edit masks or identity annotations. |
| Comparison | Phase E luminance normalization and SSIM difference. Score the mean of the highest 0.5% of valid pixel differences. Apply the fixed 0.13 pixel cutoff and Phase E morphology for regions. |
| Abstention | Reject unsupported dimensions or failed registration. Suppress the difference score, heatmap, and regions. |
| Primary rule | Flag a difference only when score is strictly greater than **0.5850238270653461**. |
| Secondary rules | Report **0.5132816869413678** and **0.4876910396336857** with the same strict comparison and abstention policy. |

The three document thresholds reproduce the higher 95th, 90th, and 80th percentiles of genuine scores from the first 64 Phase E ordinary-transformation validation-calibration sources. Their nominal 5%, 10%, and 20% FPR labels describe development targets. The new set reports actual FPR achieved at each frozen threshold and cannot be used to set them. Record the dataset digest, policy digest, code revision, software versions, and machine before publishing results. Version any later preprocessing, registration, or operating-point changes as a new policy.

## Private dataset and review

Version the inventory before scoring and keep images, edit masks, consent records, crosswalks, and per-pair outputs local. Use new receipt layouts, issuers, or source documents where feasible. A stable source group includes the physical or digital origin and all derivative captures and revisions. Group by this lineage across development and independent collection. Use separate acquisition sessions with varied devices, lighting, viewpoints, and operators when possible. Preassign conditions and retain hard failures. Where possible, capture every version on the same acquisition schedule to avoid capture-process shortcuts.

| Scenario | Ground truth and evidence |
| --- | --- |
| `recapture` | Same physical/digital document version, photographed or scanned in distinct sessions. Record the session, acquisition device or scanner, document key, and conditions. |
| `localized_edit` | Same lineage, with a controlled, documented edit and a candidate-space localization mask. To qualify for the physical stratum, both assets must come from physical capture, for example a captured printed edit. An image edited after acquisition must be recorded as simulated or born digital as appropriate. |
| `legitimate_revision` | Same lineage, with a documented authorized content revision and a candidate-space change mask. Report its visual-change rate separately. |
| `wrong_reference` | Two independently verified document identities in different groups. Include same-template hard mismatches and visually distinct pairs under a fixed sampling plan. |
| `simulated_unchanged` | Optional unchanged controls with simulated acquisition. Report these separately. |

Use a private JSON inventory with `schema_version: 1`, `dataset_version`, `review`, `images`, and `pairs`. The review object requires `reviewer`, `review_date`, `rights_record`, `source_crosswalk_record`, and the true booleans `rights_verified`, `source_independence_verified`, and `pairing_verified`. Record who reviewed rights to use and disclose images, where permissions are stored, how the developer and independent sources were cross-checked, and how pairing and genuine recapture claims were established. A manifest declaration or checksum cannot authenticate captures, licenses, authorization, or near-duplicate provenance. The reviewer must check collection records and aliases independently.

Each image has `id`, `path` relative to the authorized root, `sha256`, `document_id`, `source_group`, `capture_id`, `acquisition`, `license`, and `provenance`. Acquisition is one of `camera`, `scanner`, `born_digital`, or `simulated`. Physical assets also require `capture_session`. Capture IDs are unique asset identifiers. The provenance entry points to collection records, device/session detail, the origin of any digital edit, and, where applicable, the generator or edit mechanism. A pair has `id`, `reference` and `candidate` image IDs, `scenario`, and `annotation` pointing to the supporting version or identity evidence. Known changes require a `mask` object with `path` and `sha256`; supply a nonempty binary grayscale PNG on the EXIF-oriented candidate raster. The audit checks asset digests, scope, supported encoding, relationships, duplicate bytes, source aliases visible in the pinned manifest, and mask geometry.

## Evaluation procedure

Install `research/requirements-reference.txt` into the research environment. Run the audit first and record its output. In a fresh output directory, run the frozen comparison once the rights, provenance and source review is complete.

```sh
python -m research.phase_f_evaluate audit --manifest data/phase-f/inventory.json --data-root data/phase-f --aiforge-manifest data/aiforge/manifest_deduplicated.csv
python -m research.phase_f_evaluate evaluate --manifest data/phase-f/inventory.json --data-root data/phase-f --aiforge-manifest data/aiforge/manifest_deduplicated.csv --output artifacts/phase-f/v1/results.json
```

The primary AUROC compares localized edits with unchanged recaptures among aligned physical pairs. Report class coverage alongside AUROC. At each threshold report recall and FPR across all requested comparisons, where abstentions count as no flags, and among scored comparisons. A missing class yields no AUROC, and sample and source group counts must accompany estimates. Results with simulated or born-digital inputs remain in their own strata. Localization applies the estimated warp to the candidate mask only after inference. Report IoU within valid overlap on evaluable edits and an all-edits IoU with failed or unobservable comparisons assigned zero. Report revision flags, mismatch alignment rejection and wrong-reference acceptance separately. No explicit mismatch detection is available, including when an accepted wrong pair yields a high score.

Time both image decodes through normalization, registration, scoring, and localization on CPU and report median and p95. The offline latency excludes file reads, checksum verification, ground-truth loading, network transport, and UI rendering. The Studio endpoint reports its own decode-through-preview latency. For a publishable evaluation, preplan source-group uncertainty intervals, device/session and document-type slices, sample size adequate to resolve low FPR, and privacy-reviewed visual error examples.

## Development follow-up

Use a separate rights-cleared development collection with capture landmarks and spatial edit annotations. For registration rejection, measure landmark reprojection and overlap and categorize failures by glare, crop, blur, low texture, and repeated templates. For unchanged-text edge residuals, compare residual distributions near repeatable reference edges against text interiors and annotated edits. For photometry, compare repeat captures under controlled exposure, shadow, and white-balance changes and test whether residual reduction preserves localized edits. Keep registration and comparison changes as separate ablations. Freeze any alternative before comparing it with v1 on a newly untouched partition if the first independent results were inspected. None of these follow-up experiments has been run on independent captures.

## Research Studio and decision

For authorized local images only, set `FORGERYLENS_REFERENCE_RESEARCH=1` and run `python -m uvicorn api.main:app --host 127.0.0.1`. Open `/studio` and choose **Compare references**. The research interface presents aligned previews, valid overlap, difference heatmap, highlighted regions, and registration diagnostics. It is disabled by default. Use the existing studio for the historical production demo. See the [comparison contract](REFERENCE_COMPARISON_CONTRACT.md).

| Decision | Evidence |
| --- | --- |
| Research demonstration | Suitable for illustrating classical registration, visible differences, and abstention with a trusted reference. |
| Reliable real-world change detection | Undemonstrated. Phase E used simulated acquisition variations and recall was weak at low FPR. |
| Broader deployment | Requires independent physical captures and localized edits, documented revisions and mismatch controls, source-level uncertainty at practical operating points, and an external way to establish reference identity and trust. |

AI-assistance disclosure: Codex assisted with the protocol, implementation, tests, and documentation. The implementation tests use artificial fixtures and supply no real-world performance evidence. Dataset rights, capture provenance, and submissions require human verification.
