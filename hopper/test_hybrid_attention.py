"""CPU semantics tests; CUDA comparisons are in test_hybrid_attention_cuda.py."""

import pytest
import torch

from flash_attn_3.hybrid import window_memory_attention
from flash_attn_3.recurrent import WindowRNNAttention, recurrent_reference


def dense_oracle(q, k, v, mv, ml, window, sinks):
    """One explicit softmax over the union plus a per-query memory element."""
    groups = q.shape[2] // k.shape[2]
    scores = (
        torch.einsum("bthd,bshd->bhts", q, k.repeat_interleave(groups, 2))
        * q.shape[-1] ** -0.5
    )
    t = q.shape[1]
    i, j = (
        torch.arange(t, device=q.device)[:, None],
        torch.arange(t, device=q.device)[None, :],
    )
    mask = (j <= i) & ((j < sinks) | (j > i - (window - sinks)))
    scores = scores.masked_fill(~mask, -torch.inf)
    mass = ml.transpose(1, 2).masked_fill(
        torch.arange(t, device=q.device) < window, -torch.inf
    )
    scores = torch.cat((scores, mass.unsqueeze(-1)), -1)
    p = scores.softmax(-1)
    out = torch.einsum("bhts,bshv->bthv", p[..., :-1], v.repeat_interleave(groups, 2))
    out = out + p[..., -1].transpose(1, 2).unsqueeze(-1) * mv
    return out, scores.logsumexp(-1).transpose(1, 2)


@pytest.mark.parametrize(
    "window,sinks", [(1, 0), (4, 0), (4, 2), (6, 5), (16, 3), (16, 12)]
)
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_union_output_and_all_input_gradients(window, sinks, kv_heads):
    torch.manual_seed(4)
    shapes = [
        (2, 9, 4, 6),
        (2, 9, kv_heads, 6),
        (2, 9, kv_heads, 8),
        (2, 9, 4, 8),
        (2, 9, 4),
    ]
    xs = [torch.randn(s, dtype=torch.double, requires_grad=True) for s in shapes]
    actual = window_memory_attention(
        *xs,
        window_size=window,
        num_sink_tokens=sinks,
        backend="reference",
        return_lse=True,
    )
    expected = dense_oracle(*xs, window, sinks)
    weights = [torch.randn_like(y) for y in actual]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-12, atol=1e-12)
    ga = torch.autograd.grad(sum((a * w).sum() for a, w in zip(actual, weights)), xs)
    ge = torch.autograd.grad(sum((e * w).sum() for e, w in zip(expected, weights)), xs)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=1e-11, atol=1e-11)
    assert ga[3][:, :window].count_nonzero() == 0
    assert ga[4][:, :window].count_nonzero() == 0


def test_count_mass_recovers_uniform_full_attention():
    torch.manual_seed(8)
    t, window, sinks = 13, 5, 2
    q = torch.zeros(1, t, 2, 4, dtype=torch.double)
    k = torch.zeros_like(q)
    v = torch.randn(1, t, 2, 4, dtype=torch.double)
    mv, ml = torch.zeros_like(v), torch.full(q.shape[:3], -torch.inf, dtype=q.dtype)
    for i in range(window, t):
        middle = v[:, sinks : i - (window - sinks) + 1]
        mv[:, i] = middle.mean(1)
        ml[:, i] = torch.tensor(middle.shape[1], dtype=q.dtype).log()
    out = window_memory_attention(
        q, k, v, mv, ml, window_size=window, num_sink_tokens=sinks, backend="reference"
    )
    expected = v.cumsum(1) / torch.arange(1, t + 1)[None, :, None, None]
    torch.testing.assert_close(out, expected)


def test_empty_memory_and_extreme_mass():
    q = torch.randn(1, 8, 2, 4, dtype=torch.double)
    v = torch.randn_like(q)
    mv = torch.randn_like(q, requires_grad=True)
    ml = torch.tensor(
        [[-torch.inf, -torch.inf, 1e4, -1e4, 1e4, -1e4, 1e4, -1e4]], dtype=q.dtype
    )
    ml = ml[:, :, None].expand(-1, -1, 2).clone().requires_grad_()
    out, lse = window_memory_attention(
        q, q, v, mv, ml, window_size=2, backend="reference", return_lse=True
    )
    assert out.isfinite().all() and lse.isfinite().all()
    out.sum().backward()
    assert mv.grad.isfinite().all() and ml.grad.isfinite().all()
    torch.testing.assert_close(out[:, 2], mv[:, 2])


@pytest.mark.parametrize("kind", ["gdn", "kda", "gdn2"])
@pytest.mark.parametrize("window,sinks", [(1, 0), (5, 2), (5, 4), (20, 12)])
def test_prefill_matches_bounded_decode(kind, window, sinks):
    torch.manual_seed(19)
    layer = WindowRNNAttention(
        12,
        4,
        4,
        num_kv_heads=2,
        value_dim=6,
        window_size=window,
        num_sink_tokens=sinks,
        rnn_type=kind,
        backend="reference",
    ).double()
    with torch.no_grad():
        layer.mass_q.normal_(std=0.1)
        layer.mass_value.normal_(std=0.1)
    x = torch.randn(2, 13, 12, dtype=torch.double)
    expected = layer(x)
    cache, pieces = None, []
    for token in x.split(1, dim=1):
        out, cache = layer.step(token, cache)
        pieces.append(out)
        assert len(cache.sinks) <= sinks
        assert len(cache.recent) <= window - sinks
        assert len(cache.sinks) + len(cache.recent) <= window
        assert cache.state.shape == (2, 2, 4, 6)
    torch.testing.assert_close(torch.cat(pieces, 1), expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("kind", ["gdn", "kda", "gdn2"])
def test_packed_gradients_and_causality(kind):
    torch.manual_seed(13)
    layer = WindowRNNAttention(
        8, 2, 4, window_size=4, num_sink_tokens=1, rnn_type=kind, backend="reference"
    ).double()
    x = torch.randn(1, 15, 8, dtype=torch.double, requires_grad=True)
    y = layer(x, cu_seqlens=torch.tensor([0, 7, 15]))
    expected = torch.cat([layer(x[:, :7]), layer(x[:, 7:])], 1)
    torch.testing.assert_close(y, expected)
    params = [x, *layer.parameters()]
    a = torch.autograd.grad(y.square().sum(), params)
    b = torch.autograd.grad(expected.square().sum(), params)
    for ga, gb in zip(a, b):
        torch.testing.assert_close(ga, gb)
        assert ga.isfinite().all()
    assert all(g.abs().sum() > 0 for g in a)
    changed = x.detach().clone()
    changed[:, 7:] += 50
    torch.testing.assert_close(layer(changed)[:, :7], layer(x)[:, :7])
    torch.testing.assert_close(
        layer(changed, cu_seqlens=torch.tensor([0, 7, 15]))[:, :7], y[:, :7]
    )


def test_recurrence_against_matrix_definition_and_gradcheck():
    torch.manual_seed(43)
    q, k, v = [
        torch.randn(1, 3, 1, 2, dtype=torch.double, requires_grad=True)
        for _ in range(3)
    ]
    g = (-torch.rand_like(q)).requires_grad_()
    erase, write = [torch.rand_like(q, requires_grad=True) for _ in range(2)]
    out = recurrent_reference(q, k, v, g, erase, write)
    state, expected = torch.zeros(2, 2, dtype=q.dtype), []
    for i in range(3):
        ki, vi, gi, bi, wi = [z[0, i, 0] for z in (k, v, g, erase, write)]
        state = (torch.eye(2) - torch.outer(ki, bi * ki)) @ torch.diag(
            gi.exp()
        ) @ state + torch.outer(ki, wi * vi)
        expected.append(q[0, i, 0] @ state)
    torch.testing.assert_close(out[0, :, 0], torch.stack(expected))
    assert torch.autograd.gradcheck(recurrent_reference, (q, k, v, g, erase, write))


def test_invalid_configuration_and_no_implicit_cpu_fallback():
    with pytest.raises(ValueError, match="recent includes self"):
        WindowRNNAttention(8, 2, 4, window_size=4, num_sink_tokens=4)
    layer = WindowRNNAttention(8, 2, 4, window_size=4)
    with pytest.raises(ValueError, match="CUDA"):
        layer(torch.randn(1, 6, 8))
    ref = WindowRNNAttention(8, 2, 4, window_size=4, backend="reference")
    with pytest.raises(ValueError, match="strictly partition"):
        ref(torch.randn(1, 6, 8), cu_seqlens=torch.tensor([0, 3, 2, 6]))
