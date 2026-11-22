import csv

from PIL import Image

from research.manifest import COLUMNS, audit


def build_fixture(folder, second_source="doc-a", second_split="train", same_pixels=False):
    for name, color in (("a", 0), ("b", 0 if same_pixels else 55), ("mask", 255)):
        Image.new("L", (20, 10), color).save(folder / f"{name}.png")
    rows = [
        dict(sample_id="a", source_document_id="doc-a", variant_id="original", label="genuine",
             split="train", image_path="a.png"),
        dict(sample_id="b", source_document_id=second_source, variant_id="edit", label="manipulated",
             split=second_split, image_path="b.png", generator="fixture", mask_path="mask.png"),
    ]
    path = folder / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, source_dataset="fixture", document_category="receipt",
                                 image_width=20, image_height=10))
    return path


def test_valid_manifest(tmp_path):
    assert audit(build_fixture(tmp_path), tmp_path) == []


def test_source_group_leakage(tmp_path):
    assert "source group" in audit(build_fixture(tmp_path, second_split="shifted_test"), tmp_path)[0]


def test_duplicate_content_under_other_id(tmp_path):
    manifest = build_fixture(tmp_path, second_source="doc-b", second_split="test", same_pixels=True)
    assert "identical image bytes" in audit(manifest, tmp_path)[0]


def test_mask_must_match_document(tmp_path):
    manifest = build_fixture(tmp_path)
    Image.new("L", (2, 2), 255).save(tmp_path / "mask.png")
    assert "aligned" in audit(manifest, tmp_path)[0]