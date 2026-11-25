# Collect an independent physical-capture evaluation set

Use the frozen [Phase F protocol](history/REAL_WORLD_VERIFICATION_ANALYSIS.md) with a versioned private inventory. Start with original fictional layouts or documents whose owners explicitly permit the intended experiments. Prefer generated-for-study names and amounts, and exclude names, addresses, payment numbers, health records, or real transactions. Before capture, agree who may access the files, where to store them, when to delete them, and whether aggregate results or images may be disclosed. Keep the consent, terms, signed permissions, raw images, masks, and crosswalk in authorized local storage.

## Capture plan

1. Assign each document a stable `document_id` and source lineage `source_group`. Compare issuer, layout, provenance and hash against development sources. Have a reviewer resolve aliases and near-duplicates. Keep every capture, edit, and revision in its source group. Keep all independent groups outside subsequent development.
2. Select the trusted physical or printed version for each reference. Capture it and its unchanged recapture in independent camera or scanner sessions. Reposition the page or device between sessions. Adjacent video frames or software perturbations are unsuitable as independent sessions.
3. Create a controlled localized edit with the document owner's permission or use a fictional version designed for study. Photograph or scan the physical edited version with the same acquisition schedule. Track edit mechanism and fields. Record a digital-only edge case under `simulated` or `born_digital` as appropriate.
4. Make a legitimate revision with a written change record. Retain the updated document’s original source group and capture it as a separate acquisition.
5. Select wrong-reference controls from distinct document identities and source groups. Include visually different and same-template controls. Record the identity crosswalk without exposing personal information in model inputs.
6. Annotate each edited and revised candidate with a nonempty grayscale 0/255 PNG on the EXIF-oriented candidate raster. Review the coverage at original resolution. Record missing or ambiguous ground truth explicitly before freezes. Do not crop or discard comparisons based on model outputs.
7. For each asset log SHA-256, rights, provenance, device class, session ID, capture conditions, orientation, and any processing between capture and evaluation. Predetermine the scenarios and storage access. Approve and sign off on pairing, rights, and prior-source independence before scoring.

Begin with a small feasibility set and report source counts and uncertainty. Plan a separate, larger sample before treating a low measured false-positive rate as stable.

## Audit and run the frozen protocol

Place the private inventory at `data/phase-f/inventory.json` under the authorized root. The [Phase F inventory specification](history/REAL_WORLD_VERIFICATION_ANALYSIS.md#private-dataset-and-review) gives the field names and admissible acquisition labels.

```sh
python -m research.reference_verification_evaluate audit --manifest data/phase-f/inventory.json --data-root data/phase-f --aiforge-manifest data/aiforge/manifest_deduplicated.csv
python -m research.reference_verification_evaluate evaluate --manifest data/phase-f/inventory.json --data-root data/phase-f --aiforge-manifest data/aiforge/manifest_deduplicated.csv --output artifacts/phase-f/v1/results.json
```

The software checks consistency and file digests. The human review establishes physical acquisition, consent, pairing, near-duplicate provenance, and permitted disclosure.

