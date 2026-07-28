"""Tests for my_kws.model — shape contracts, size budget, basic sanity."""

import torch

from my_kws.model import DSCNN, count_parameters


def test_forward_output_shape():
    """(batch, 1, 40, 98) in -> (batch, 1) logits out."""
    model = DSCNN()
    x = torch.randn(4, 1, 40, 98)
    out = model(x)
    assert out.shape == (4, 1)


def test_forward_batch_size_one():
    """BatchNorm layers must still work in eval mode with batch=1."""
    model = DSCNN().eval()
    x = torch.randn(1, 1, 40, 98)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1)


def test_parameter_budget():
    """Stay small enough for edge deployment (quantization comes later)."""
    model = DSCNN()
    assert count_parameters(model) < 150_000


def test_output_is_finite():
    model = DSCNN().eval()
    x = torch.randn(8, 1, 40, 98)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


def test_gradients_flow():
    """One backward pass should give every trainable parameter a gradient."""
    model = DSCNN()
    x = torch.randn(2, 1, 40, 98)
    y = torch.tensor([[1.0], [0.0]])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"


def test_eval_mode_is_deterministic():
    """Dropout/BN in eval mode: same input -> same output."""
    model = DSCNN().eval()
    x = torch.randn(2, 1, 40, 98)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2)
