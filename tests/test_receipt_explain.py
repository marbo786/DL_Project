import torch
from PIL import Image

from research.receipt.explain import matched_control, neutralize, offset_boxes, restricted


def test_control_excludes_overlay_and_mask_and_matches_luminance():
    image = Image.new("RGB", (40, 10), (230, 230, 230))
    image.paste((220, 220, 220), (20, 0, 30, 10))
    image.paste((10, 10, 10), (30, 0, 40, 10))
    boxes = [(0, 0, 10, 10), (8, 0, 18, 10), (10, 0, 20, 10), (20, 0, 30, 10), (30, 0, 40, 10)]
    control = matched_control(image, boxes, 0, [False, False, True, False, False])
    assert control == boxes[3]
    assert neutralize(image, control).getpixel((20, 0)) == (127, 127, 127)
    assert image.getpixel((20, 0)) == (220, 220, 220)


def test_restricted_pooling_keeps_background_outside_edit_and_offset_keeps_edges():
    assert restricted(torch.tensor([1., 2., 3., 4., 5.]), [True, False, False, False, False]) == {
        "edit_tiles": 1, "background_tiles": 4, "edit_logit": 1., "background_logit": 4.,
    }
    assert restricted(torch.tensor([1.]), [True])["background_logit"] is None
    boxes = offset_boxes((224, 240))
    assert (56, 56, 168, 168) in boxes
    assert (0, 0, 112, 112) == boxes[0]
    assert (112, 128, 224, 240) == boxes[-1]
