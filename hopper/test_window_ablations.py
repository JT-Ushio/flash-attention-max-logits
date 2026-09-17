import copy
import pytest
import torch
from torch.nn import functional as F
from flash_attn_3.hybrid import window_memory_attention
from flash_attn_3.recurrent import WindowRNNAttention, _rotary


@pytest.mark.parametrize("window,sinks", [(6, 2), (6, 0), (6, 5), (20, 3)])
def test_cache_positions_match_explicit_reindexed_attention_and_gradients(
    window, sinks
):
    torch.manual_seed(81)
    t, h, d = 13, 2, 8
    q, k, v, mv = [
        torch.randn(1, t, h, d, dtype=torch.double, requires_grad=True)
        for _ in range(4)
    ]
    ml = torch.randn(1, t, h, dtype=torch.double, requires_grad=True)
    p = torch.arange(t)
    qr, kr = [_rotary(x, p, 10000.0, 4) for x in (q, k)]
    sq = _rotary(q, p.clamp(max=window - 1), 10000.0, 4)
    actual = window_memory_attention(
        qr,
        kr,
        v,
        mv,
        ml,
        window_size=window,
        num_sink_tokens=sinks,
        backend="reference",
        sink_q=sq,
    )
    expected = []
    for i in range(t):
        ids = list(range(min(sinks, i + 1))) + list(
            range(max(sinks, i - (window - sinks) + 1), i + 1)
        )
        qi = _rotary(q[:, i : i + 1], torch.tensor([len(ids) - 1]), 10000.0, 4)
        ki = _rotary(k[:, ids], torch.arange(len(ids)), 10000.0, 4)
        scores = torch.einsum("bthd,bshd->bhts", qi, ki) * d**-0.5
        values = v[:, ids].transpose(1, 2)
        if i >= window:
            scores = torch.cat(
                (scores, ml[:, i : i + 1].transpose(1, 2).unsqueeze(-1)), -1
            )
            values = torch.cat((values, mv[:, i : i + 1].transpose(1, 2)), 2)
        expected.append(torch.einsum("bhts,bhsv->bthv", scores.softmax(-1), values))
    expected = torch.cat(expected, 1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    weight = torch.randn_like(actual)
    xs = (q, k, v, mv, ml)
    ga = torch.autograd.grad((actual * weight).sum(), xs, retain_graph=True)
    ge = torch.autograd.grad((expected * weight).sum(), xs, allow_unused=True)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(
            a, torch.zeros_like(a) if e is None else e, atol=1e-11, rtol=1e-11
        )


@pytest.mark.parametrize(
    "mode,gate", [("absolute", False), ("window", False), ("absolute", True)]
)
def test_layer_decode_matches_prefill(mode, gate):
    torch.manual_seed(82)
    layer = WindowRNNAttention(
        16,
        2,
        8,
        window_size=6,
        num_sink_tokens=2,
        rotary_dim=4,
        backend="reference",
        sink_position_mode=mode,
        memory_output_gate=gate,
    ).double()
    x = torch.randn(1, 13, 16, dtype=torch.double, requires_grad=True)
    out = layer(x)
    cache, steps = None, []
    for i in range(x.shape[1]):
        y, cache = layer.step(x[:, i : i + 1], cache)
        steps.append(y)
    torch.testing.assert_close(out, torch.cat(steps, 1), atol=1e-10, rtol=1e-10)
    out.square().sum().backward()
    assert x.grad.isfinite().all()
    if gate:
        assert layer.output_gate_proj.weight.grad.abs().sum() > 0


def test_gate_changes_only_memory_value_not_mass_or_prefix(monkeypatch):
    import flash_attn_3.recurrent as recurrent

    torch.manual_seed(83)
    base = WindowRNNAttention(
        16, 2, 8, window_size=6, num_sink_tokens=2, backend="reference"
    ).double()
    gated = WindowRNNAttention(
        16,
        2,
        8,
        window_size=6,
        num_sink_tokens=2,
        backend="reference",
        memory_output_gate=True,
    ).double()
    gated.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(1, 13, 16, dtype=torch.double)
    captured = []
    original = recurrent.window_memory_attention

    def capture(*args, **kwargs):
        captured.append((args[3].clone(), args[4].clone()))
        return original(*args, **kwargs)

    monkeypatch.setattr(recurrent, "window_memory_attention", capture)
    a, b = base(x), gated(x)
    torch.testing.assert_close(a[:, :6], b[:, :6])
    torch.testing.assert_close(captured[0][1], captured[1][1])
    gate = F.silu(gated.output_gate_proj(x[:, 6:])).unflatten(-1, (2, 8))
    torch.testing.assert_close(captured[1][0][:, 6:], captured[0][0][:, 6:] * gate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode,gate", [("window", False), ("absolute", True)])
def test_cuda_ablation_parameter_gradients(mode, gate):
    torch.manual_seed(84)
    layer = (
        WindowRNNAttention(
            64,
            2,
            128,
            window_size=37,
            num_sink_tokens=5,
            rotary_dim=32,
            backend="fa3",
            sink_position_mode=mode,
            memory_output_gate=gate,
        )
        .cuda()
        .bfloat16()
    )
    reference = copy.deepcopy(layer).float()
    reference.backend = "reference"
    x = torch.randn(1, 131, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    out, expected = layer(x), reference(xr)
    grad = torch.randn_like(out)
    (out * grad).sum().backward()
    (expected * grad.float()).sum().backward()
    torch.testing.assert_close(out.float(), expected, atol=0.03, rtol=0.03)
    torch.testing.assert_close(x.grad.float(), xr.grad, atol=0.04, rtol=0.04)
    for (name, p), (_, pr) in zip(
        layer.named_parameters(), reference.named_parameters()
    ):
        assert p.grad is not None and p.grad.isfinite().all(), name
        torch.testing.assert_close(
            p.grad.float(), pr.grad, atol=0.08, rtol=0.05, msg=name
        )
