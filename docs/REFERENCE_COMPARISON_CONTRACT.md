# Reference comparison contract

`POST /api/v1/compare` is available for direct loopback clients when `FORGERYLENS_REFERENCE_RESEARCH=1`. The existing single-image route serves the synthetic demonstration. The [Phase F specification](REAL_WORLD_VERIFICATION_ANALYSIS.md) records the frozen operating point and data requirements.

## Request

A multipart request supplies two independent files in `reference` and `candidate`. The caller is responsible for the trusted original's provenance and for supplying the same document in both fields. The comparison measures visual change relative to that reference. Legitimate content changes can also trigger evidence.

Decode both images, apply EXIF orientation, and enforce explicit byte and pixel limits before registration. Return a validation error for malformed, unsupported, or missing inputs. Treat a low-confidence registration or inadequate overlap as an abstention.

## Local research response

| Field | Meaning |
| --- | --- |
| `decision` | `difference_detected`, `no_difference_flagged`, or `inconclusive` |
| `reason` | Nullable abstention code, currently `unsupported_dimensions` or `registration_rejected` |
| `score` | Registered SSIM difference score, nullable on abstention |
| `threshold`, `policy_version` | Frozen primary score cutoff and comparison policy identifier |
| `alignment` | Support status and, when available, inlier count, inlier ratio, reprojection residual, and valid overlap fraction |
| `raster` | Width, height, and coordinate frame identifier `normalized_reference`, present on accepted alignment. |
| `previews` | Reference, aligned candidate, heatmap, valid support, isolated region layer and composed region overlay as PNG data URLs in the normalized reference raster. Present only when registration is accepted. |
| `inference_ms` | Decode through scoring and preview encoding latency, excluding upload and network transport |

The comparison endpoint returns an uncalibrated visual-difference score. Reference identity is a caller responsibility; a high score cannot identify a wrong reference. Responses have a no-store cache directive.

## Proposed future operational response

| Field | Meaning |
| --- | --- |
| `decision` | One of `change_detected`, `no_change_detected`, or `abstain` |
| `comparison_score` | Method-specific score, nullable on abstention |
| `probability` | Calibrated probability within the documented benchmark scope, nullable when unavailable |
| `calibration` | Version, calibration split, operating threshold and coverage policy |
| `alignment` | Status, overlap fraction, inlier count and ratio, reprojection error and the candidate-to-reference transform |
| `regions` | Suspected polygons or components in the oriented reference frame, with the corresponding valid support |
| `heatmap` | Difference map, width and height, reference-frame mapping, and a validity mask |
| `model_version` | Version of preprocessing, alignment, scoring and thresholds |
| `warnings` | Limitations and reason codes, including low overlap, uncertain alignment, or reference mismatch |
| `latency_ms` | Decode-to-response comparison latency |

An abstention returns the alignment diagnostics and reason. It does not return a change decision or overlays interpreted as suspected edits. The Studio can use the reference, candidate and aligned previews for a before/after view. Draw evidence only over valid overlapping pixels. Experiment galleries stay local and illustrate behavior rather than implying authenticity.

Any operational adapter requires independent evaluation of real recaptures, mismatched references and legitimate document revisions, plus a frozen versioned operating point.
