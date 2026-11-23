"""Validate private inventories for the independent comparison benchmark."""

import csv
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from research.phase_f_compare import MAX_BYTES, decode
from research.reference_data import PINNED_MANIFEST

SCENARIOS = {"recapture", "localized_edit", "legitimate_revision", "wrong_reference"}
PHYSICAL = {"camera", "scanner"}


def required_text(record, key):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"Missing or invalid {key}")
    return value


def read_hashed(folder, record):
    location = folder / required_text(record, "path")
    path = location.resolve()
    if not path.is_relative_to(folder) or not path.is_file():
        raise ValueError("Missing input or path escapes the data root")
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Input size outside allowed bounds")
    content = path.read_bytes()
    if sha256(content).hexdigest() != required_text(record, "sha256"):
        raise ValueError("Input SHA-256 mismatch")
    return content


def load_mask(folder, record, size):
    from io import BytesIO

    with Image.open(BytesIO(read_hashed(folder, record)), formats=("PNG",)) as image:
        if (image.size != size or image.mode not in ("1", "L")
                or image.getexif().get(274, 1) != 1 or getattr(image, "is_animated", False)):
            raise ValueError("Mask must be a grayscale PNG on the oriented candidate raster")
        mask = np.asarray(image.convert("L"))
        if not set(np.unique(mask)) <= {0, 255} or not mask.any():
            raise ValueError("Mask must be nonempty and binary (0/255)")
        return mask > 0


def excluded_sources(aiforge_manifest):
    if sha256(aiforge_manifest.read_bytes()).hexdigest() != PINNED_MANIFEST:
        raise ValueError("AIForge manifest does not match the pinned checksum")
    # Read source identifiers only. No sealed images, masks or model outputs.
    with aiforge_manifest.open(newline="", encoding="utf-8-sig") as stream:
        return {row["source_document_id"].casefold() for row in csv.DictReader(stream)}


def audit(manifest, folder, excluded):
    folder = folder.resolve()
    data = json.loads(manifest.read_bytes())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema")
    required_text(data, "dataset_version")
    review = data["review"]
    for field in ("reviewer", "review_date", "rights_record", "source_crosswalk_record"):
        required_text(review, field)
    if any(review.get(field) is not True for field in
           ("rights_verified", "source_independence_verified", "pairing_verified")):
        raise ValueError("Complete rights, source-independence and pairing review before ingestion")
    images, groups, sizes = {}, {}, {}
    hashes, capture_ids = set(), set()
    for image in data["images"]:
        for field in ("id", "document_id", "source_group", "capture_id", "acquisition",
                      "license", "provenance"):
            required_text(image, field)
        identity, group = image["document_id"].casefold(), image["source_group"].casefold()
        capture_id = image["capture_id"].casefold()
        if image["id"] in images or capture_id in capture_ids or group in excluded or identity in excluded:
            raise ValueError("Duplicate image/capture ID or overlapping prior source")
        if groups.setdefault(identity, group) != group:
            raise ValueError("A document identity spans source groups")
        if image["acquisition"] not in PHYSICAL | {"simulated", "born_digital"}:
            raise ValueError("Unknown acquisition type")
        if image["acquisition"] in PHYSICAL:
            required_text(image, "capture_session")
        content = read_hashed(folder, image)
        digest = image["sha256"]
        if digest in hashes:
            raise ValueError("Duplicate image bytes across inventory entries")
        hashes.add(digest)
        sizes[image["id"]] = decode(content).size
        images[image["id"]] = image
        capture_ids.add(capture_id)
    counts, seen, pair_ids, used = Counter(), set(), set(), set()
    for pair in data["pairs"]:
        for field in ("id", "reference", "candidate", "scenario", "annotation"):
            required_text(pair, field)
        scenario = pair["scenario"]
        key = (pair["reference"], pair["candidate"])
        if scenario not in SCENARIOS | {"simulated_unchanged"} or key[0] == key[1] or key in seen or pair["id"] in pair_ids:
            raise ValueError("Invalid or duplicated pair")
        if not set(key) <= images.keys():
            raise ValueError("Pair refers to an unknown image")
        ref, can = (images[k] for k in key)
        same = ref["document_id"].casefold() == can["document_id"].casefold()
        same_group = ref["source_group"].casefold() == can["source_group"].casefold()
        if scenario == "wrong_reference":
            if same or same_group:
                raise ValueError("Wrong-reference controls must come from different lineages")
        elif not same or not same_group:
            raise ValueError("Same-source controls must share document identity and lineage")
        acquisitions = {ref["acquisition"], can["acquisition"]}
        if scenario == "recapture" and (acquisitions - PHYSICAL
                or ref["capture_session"].casefold() == can["capture_session"].casefold()):
            raise ValueError("A genuine recapture requires separate physical capture sessions")
        if scenario == "simulated_unchanged" and "simulated" not in acquisitions:
            raise ValueError("A simulated unchanged control needs a simulated acquisition")
        edited = scenario in {"localized_edit", "legitimate_revision"}
        if edited:
            load_mask(folder, pair["mask"], sizes[key[1]])
        elif "mask" in pair:
            raise ValueError("Only known changes may carry an edit mask")
        seen.add(key)
        pair_ids.add(pair["id"])
        used.update(key)
        counts[scenario] += 1
    if not SCENARIOS <= counts.keys() or used != images.keys():
        raise ValueError("Include all four scenarios and only images referenced by the inventory")
    return data, images, dict(counts)
