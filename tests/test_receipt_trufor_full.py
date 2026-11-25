import torch
from torch import nn

from research.receipt_trufor_full import bounded_attention


def test_large_attention_matches_direct_softmax():
    generator = torch.Generator().manual_seed(19)
    layer = nn.Module()
    layer.num_heads = 1
    layer.sr_ratio = 1
    layer.scale = .5
    layer.q = nn.Linear(4, 4)
    layer.kv = nn.Linear(4, 8)
    layer.proj = nn.Linear(4, 4)
    layer.proj_drop = nn.Identity()
    layer.eval()
    tokens = torch.randn(1, 3500, 4, generator=generator)

    def direct(module, values, height, width):
        query = module.q(values)
        key, value = module.kv(values).chunk(2, dim=-1)
        matrix = (query @ key.transpose(-1, -2)) * module.scale
        return module.proj(matrix.softmax(dim=-1) @ value)

    with torch.inference_mode():
        expected = direct(layer, tokens, 50, 70)
        actual = bounded_attention(layer, tokens, 50, 70, direct)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
