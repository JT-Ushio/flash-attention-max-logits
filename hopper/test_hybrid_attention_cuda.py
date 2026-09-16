"""Run on Hopper after rebuilding FA3 from this branch; never a CPU substitute."""

import pytest
import torch

from flash_attn_3.hybrid import window_memory_attention
from flash_attn_3.recurrent import WindowRNNAttention
from test_hybrid_attention import dense_oracle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA FA3"
)


def _interface():
    from flash_attn_3 import flash_attn_interface as fa

    assert "dsoftmax_lse" in str(torch.ops.flash_attn_3.bwd.default._schema), (
        "rebuild FA3 from this branch"
    )
    return fa


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "window,sinks", [(1, 0), (37, 5), (128, 0), (400, 7), (400, 300)]
)
def test_cuda_unified_softmax_gradients(dtype, window, sinks, head_dim):
    _interface()
    torch.manual_seed(5)
    shapes = [
        (2, 259, 4, head_dim),
        (2, 259, 2, head_dim),
        (2, 259, 2, head_dim),
        (2, 259, 4, head_dim),
        (2, 259, 4),
    ]
    xs = [
        torch.randn(
            s,
            device="cuda",
            dtype=dtype if i < 4 else torch.float32,
            requires_grad=True,
        )
        for i, s in enumerate(shapes)
    ]
    refs = [x.detach().float().requires_grad_() for x in xs]
    out, lse = window_memory_attention(
        *xs, window_size=window, num_sink_tokens=sinks, return_lse=True
    )
    ref, lr = dense_oracle(*refs, window, sinks)
    do, dl = torch.randn_like(out), torch.randn_like(lse)
    grads = torch.autograd.grad((out, lse), xs, (do, dl))
    ref_grads = torch.autograd.grad((ref, lr), refs, (do.float(), dl))
    atol = 0.003 if dtype == torch.float16 else 0.025
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=atol)
    torch.testing.assert_close(lse, lr, atol=0.005, rtol=0.005)
    for grad, expected in zip(grads, ref_grads):
        assert grad.isfinite().all()
        torch.testing.assert_close(grad.float(), expected, atol=atol * 3, rtol=atol * 3)


def _dense_stats(q, k, v, *, left=-1, causal=True, softcap=0.0):
    scores = torch.einsum("bthd,bshd->bhts", q, k) * q.shape[-1] ** -0.5
    if softcap:
        scores = (scores / softcap).tanh() * softcap
    i = torch.arange(q.shape[1], device=q.device)[:, None] + k.shape[1] - q.shape[1]
    j = torch.arange(k.shape[1], device=q.device)[None, :]
    mask = j <= i if causal else torch.ones_like(i + j, dtype=torch.bool)
    if left >= 0:
        mask = mask & (j >= i - left)
    scores = scores.masked_fill(~mask, -torch.inf)
    return torch.einsum("bhts,bshv->bthv", scores.softmax(-1), v), scores.logsumexp(-1)


@pytest.mark.parametrize("mode", ["out_only", "lse_only", "both"])
@pytest.mark.parametrize("api", ["dense", "qkvpacked", "raw", "varlen", "varlen_raw"])
@pytest.mark.parametrize("left,softcap", [(-1, 0.0), (31, 5.0)])
def test_cuda_lse_backward_all_interfaces(mode, api, left, softcap):
    fa = _interface()
    torch.manual_seed(24)
    dtype = torch.float16
    xs = [
        torch.randn(1, 133, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    refs = [x.detach().float().requires_grad_() for x in xs]
    if api == "dense":
        out, lse = fa.flash_attn_func(
            *xs,
            causal=True,
            window_size=(left, 0),
            softcap=softcap,
            return_attn_probs=True,
        )
    elif api == "qkvpacked":
        out, lse = fa.flash_attn_qkvpacked_func(
            torch.stack(xs, dim=2),
            causal=True,
            window_size=(left, 0),
            softcap=softcap,
            return_attn_probs=True,
        )
    elif api == "raw":
        out, lse, *_ = fa._flash_attn_forward(
            *xs,
            causal=True,
            window_size_left=left,
            window_size_right=0,
            softcap=softcap,
        )
    else:
        cu = torch.tensor([0, 63, 133], device="cuda", dtype=torch.int32)
        if api == "varlen":
            out, lse = fa.flash_attn_varlen_func(
                *(x[0] for x in xs),
                cu,
                cu,
                70,
                70,
                causal=True,
                window_size=(left, 0),
                softcap=softcap,
                return_attn_probs=True,
            )
        else:
            out, lse, *_ = fa._flash_attn_forward(
                *(x[0] for x in xs),
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=70,
                max_seqlen_k=70,
                causal=True,
                window_size_left=left,
                window_size_right=0,
                softcap=softcap,
            )
        out, lse = out[None], lse[None]
    if api.startswith("varlen"):
        parts = [
            _dense_stats(*(x[:, a:b] for x in refs), left=left, softcap=softcap)
            for a, b in [(0, 63), (63, 133)]
        ]
        ref, lr = (
            torch.cat([p[0] for p in parts], 1),
            torch.cat([p[1] for p in parts], -1),
        )
    else:
        ref, lr = _dense_stats(*refs, left=left, softcap=softcap)
    do, dl = torch.randn_like(out), torch.randn_like(lse)
    if mode == "out_only":
        loss, ref_loss = (out * do).sum(), (ref * do.float()).sum()
    elif mode == "lse_only":
        loss, ref_loss = (lse * dl).sum(), (lr * dl).sum()
    else:
        loss = (out * do).sum() + (lse * dl).sum()
        ref_loss = (ref * do.float()).sum() + (lr * dl).sum()
    actual = torch.autograd.grad(loss, xs)
    expected = torch.autograd.grad(ref_loss, refs, allow_unused=True)
    for a, e in zip(actual, expected):
        if e is None:
            e = torch.zeros_like(a)
        torch.testing.assert_close(a.float(), e.float(), atol=0.008, rtol=0.01)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("kind", ["gdn", "kda", "gdn2"])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_cuda_fla_layer_output_and_parameter_gradients(kind, kv_heads, head_dim):
    _interface()
    pytest.importorskip("fla")
    torch.manual_seed(50)
    kwargs = dict(
        d_model=64,
        num_heads=2,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        window_size=19,
        num_sink_tokens=3,
        rnn_type=kind,
    )
    layer = WindowRNNAttention(**kwargs).cuda()
    ref_layer = WindowRNNAttention(**kwargs, backend="reference").cuda()
    # Exercise the learned state/query-dependent mass, not just its initialization.
    with torch.no_grad():
        layer.mass_q.normal_(std=0.1)
        layer.mass_value.normal_(std=0.1)
    ref_layer.load_state_dict(layer.state_dict())
    x = torch.randn(2, 97, 64, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(x)
        expected = ref_layer(xr)
    do = torch.randn_like(out)
    actual_grads = torch.autograd.grad(out, [x, *layer.parameters()], do)
    expected_grads = torch.autograd.grad(expected, [xr, *ref_layer.parameters()], do)
    torch.testing.assert_close(out.float(), expected.float(), atol=0.02, rtol=0.03)
    for a, e in zip(actual_grads, expected_grads):
        assert a.isfinite().all()
        torch.testing.assert_close(a, e, atol=0.07, rtol=0.05)
