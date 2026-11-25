import numpy as np
from PIL import Image

from research.receipt_stress import occlusion_box, policy_metrics, transform


def test_occlusion_is_paired_and_targeted_rectangle_covers_border_edit():
    mask = Image.new("L", (120, 160))
    mask.paste(255, (115, 155, 120, 160))
    random_box = occlusion_box(mask, "random-occlusion-2", "receipt-1")
    assert random_box == occlusion_box(Image.new("L", mask.size), "random-occlusion-2", "receipt-1")
    targeted = occlusion_box(mask, "edit-occlusion-2", "receipt-1")
    assert mask.crop(targeted).histogram()[255] == 25
    assert (targeted[2] - targeted[0], targeted[3] - targeted[1]) == (36, 48)
    for color in ("white", "black"):
        image = Image.new("RGB", mask.size, color)
        changed = transform(image, "edit-occlusion-2", "source", targeted)
        assert changed.size == mask.size
        assert changed.getpixel((119, 159)) == (127, 127, 127)
        assert changed.getpixel((0, 0)) == image.getpixel((0, 0))


def test_coverage_counts_actual_routes_separately_from_cutoff_flags():
    result = policy_metrics(np.array([0, 0, 1, 1]), np.array([0., 2., 3., 1.]),
                            np.array([False, False, True, False]), 1.)
    assert result == {"auroc": .75, "recall": .5, "fpr": .5, "coverage_genuine": 0., "coverage_edited": .5}
