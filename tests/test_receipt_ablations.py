import torch
from torch.utils.data import TensorDataset

from research import train as legacy
from research.receipt_model import Detector, predict


def test_legacy_final_epoch_rule_ignores_earlier_validation_rank(monkeypatch):
    images = torch.rand(4, 1, 32, 32, generator=torch.Generator().manual_seed(4))
    masks = torch.zeros_like(images)
    masks[1::2, :, 10:14, 12:16] = 1
    dataset = TensorDataset(images, masks, torch.tensor([0., 1., 0., 1.]))
    torch.set_num_threads(2)

    def fit(rule, epochs):
        ranks = iter([.9, .1])
        monkeypatch.setattr(legacy, "summary_metrics", lambda *_args, **_kwargs: {"roc_auc": next(ranks)})
        return legacy.train("local", {"width": 2}, dataset, dataset, epochs, 42, batch_size=4,
                            checkpoint_rule=rule)[0]

    picked = fit("historical", 2)
    final = fit("final epoch", 2)
    assert not torch.equal(picked.first[0].weight, final.first[0].weight)
    assert int(picked.first[1].num_batches_tracked) == 1
    assert int(final.first[1].num_batches_tracked) == 2


def test_residual_freeze_control_and_whole_view_keep_model_mode():
    from PIL import Image

    detector = Detector("residual", trainable_tail=3).train()
    assert all(not block.training for block in detector.features[:-3])
    assert all(block.training for block in detector.features[-3:])
    assert not any(parameter.requires_grad for block in detector.features[:-3] for parameter in block.parameters())
    assert all(parameter.requires_grad for block in detector.features[-3:] for parameter in block.parameters())
    _, bbox = predict(detector.eval(), Image.new("RGB", (160, 220)), 8, geometry="whole")
    assert bbox == (0, 0, 160, 220)
    assert detector.mode == "residual"
