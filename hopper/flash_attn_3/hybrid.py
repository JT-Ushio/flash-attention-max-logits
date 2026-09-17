"""Causal sink + recent + query-conditioned memory attention.

The FA3 backend needs this branch's differentiable LSE backward. The explicit
``reference`` backend is a small-sequence correctness oracle, not a GPU fallback.
All partition functions use natural logarithms. See ../HYBRID_ATTENTION.md.
"""

import math

import torch


def _acc_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _validate_window(window_size, num_sink_tokens):
    if not isinstance(window_size, int) or not isinstance(num_sink_tokens, int):
        raise TypeError("window_size and num_sink_tokens must be integers")
    if not 0 <= num_sink_tokens < window_size:
        raise ValueError(
            "require 0 <= num_sink_tokens < window_size (recent includes self)"
        )


def merge_partitions(out_a, lse_a, out_b, lse_b):
    """Merge disjoint nonempty/empty partitions; LSE layout is [B, T, H].

    Empty partitions use -inf LSE and finite zero values. At least one partition
    per row must be nonempty. Accumulate in FP32 (FP64 for reference gradchecks).
    """
    dtype = _acc_dtype(out_a)
    lse_a, lse_b = lse_a.to(dtype), lse_b.to(dtype)
    lse = torch.logaddexp(lse_a, lse_b)
    out = (lse_a - lse).exp().unsqueeze(-1) * out_a.to(dtype) + (
        lse_b - lse
    ).exp().unsqueeze(-1) * out_b.to(dtype)
    return out, lse


def _attention_stats(q, k, v, *, causal, left, scale, backend, deterministic):
    if backend == "fa3":
        from flash_attn_3.flash_attn_interface import flash_attn_func

        out, lse = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=causal,
            window_size=(left, 0 if causal else -1),
            deterministic=deterministic,
            return_attn_probs=True,
        )
        return out, lse.transpose(1, 2)

    groups = q.shape[2] // k.shape[2]
    dtype = _acc_dtype(q)
    k = k.repeat_interleave(groups, dim=2).to(dtype)
    v = v.repeat_interleave(groups, dim=2).to(dtype)
    scores = torch.einsum("bthd,bshd->bhts", q.to(dtype), k) * scale
    if causal:
        tq, tk = q.shape[1], k.shape[1]
        i = torch.arange(tq, device=q.device)[:, None] + tk - tq
        j = torch.arange(tk, device=q.device)[None, :]
        mask = j <= i
        if left >= 0:
            mask = mask & (j >= i - left)
        scores = scores.masked_fill(~mask, -torch.inf)
    lse = scores.logsumexp(-1)
    out = torch.einsum("bhts,bshd->bthd", scores.softmax(-1), v)
    return out, lse.transpose(1, 2)


def window_memory_attention(
    q,
    k,
    v,
    memory_value,
    memory_log_mass,
    *,
    window_size,
    num_sink_tokens=0,
    softmax_scale=None,
    backend="fa3",
    deterministic=False,
    return_lse=False,
    sink_q=None,
):
    """Self-attention with exactly one memory component per query head.

    q/k/v: [B,T,Hq/Hkv,D/Dv]; memory_value: [B,T,Hq,Dv];
    memory_log_mass: [B,T,Hq], already a log-partition (never QK-scaled).
    q and k must already have the desired positional encoding. Optional
    ``sink_q`` supplies a different query rotation against sinks;
    this does not change recent-token scores or the causal sink prefix.
    The caller must
    construct memory from ONLY tokens N <= j <= t-(window_size-N). Empty memory
    rows are forcibly masked here. Sink/recent overlap is counted exactly once.
    Returns [B,T,Hq,Dv], and optionally differentiable LSE [B,T,Hq].
    """
    _validate_window(window_size, num_sink_tokens)
    if backend not in ("fa3", "reference"):
        raise ValueError("backend must be 'fa3' or 'reference'")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [batch, sequence, heads, dim]")
    if min(q.shape) <= 0 or min(k.shape) <= 0 or min(v.shape) <= 0:
        raise ValueError("q, k, v dimensions must be nonzero")
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
        raise ValueError("self-attention requires matching batch/sequence and KV heads")
    if q.shape[-1] != k.shape[-1] or q.shape[2] % k.shape[2]:
        raise ValueError("Q/K dimensions must match and Hq must be divisible by Hkv")
    if memory_value.shape != (*q.shape[:3], v.shape[-1]):
        raise ValueError("memory_value must have shape [B,T,Hq,Dv]")
    if memory_log_mass.shape != q.shape[:3]:
        raise ValueError("memory_log_mass must have shape [B,T,Hq]")
    if any(x.device != q.device for x in (k, v, memory_value, memory_log_mass)):
        raise ValueError("all inputs must be on the same device")
    if not q.is_floating_point() or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, v must have the same floating dtype")
    if not memory_value.is_floating_point() or not memory_log_mass.is_floating_point():
        raise ValueError("memory inputs must be floating point")
    if backend == "fa3" and (
        q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError(
            "FA3 requires CUDA FP16/BF16; select reference explicitly for CPU"
        )
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("softmax_scale must be finite and nonnegative")

    t = q.shape[1]
    if sink_q is not None and (
        sink_q.shape != q.shape or sink_q.dtype != q.dtype or sink_q.device != q.device
    ):
        raise ValueError("sink_q must match q shape, dtype and device")
    n = min(num_sink_tokens, t)
    recent = window_size - num_sink_tokens
    kwargs = dict(scale=scale, backend=backend, deterministic=deterministic)
    parts_o, parts_l = [], []
    # The prefix attends only to causal sink tokens. Removing it from the recent
    # call avoids all-masked FA3 rows and the +/-inf LSE sentinel convention.
    if n:
        prefix_o, prefix_l = _attention_stats(
            q[:, :n],
            k[:, :n],
            v[:, :n],
            causal=True,
            left=-1,
            **kwargs,
        )
        parts_o.append(prefix_o)
        parts_l.append(prefix_l)
    if t > n:
        out, lse = _attention_stats(
            q[:, n:],
            k[:, n:],
            v[:, n:],
            causal=True,
            left=recent - 1,
            **kwargs,
        )
        if n:
            sink_o, sink_l = _attention_stats(
                q[:, n:] if sink_q is None else sink_q[:, n:],
                k[:, :n],
                v[:, :n],
                causal=False,
                left=-1,
                **kwargs,
            )
            out, lse = merge_partitions(out, lse, sink_o, sink_l)
        parts_o.append(out)
        parts_l.append(lse)
    exact_o, exact_l = torch.cat(parts_o, dim=1), torch.cat(parts_l, dim=1)
    has_memory = (
        torch.arange(t, device=q.device)[None, :, None] >= num_sink_tokens + recent
    )
    mass = memory_log_mass.masked_fill(~has_memory, -torch.inf)
    # Also sanitize absent memory values: zero mass times NaN is still NaN.
    value = memory_value.masked_fill(~has_memory.unsqueeze(-1), 0)
    out, lse = merge_partitions(exact_o, exact_l, value, mass)
    out = out.to(q.dtype)
    return (out, lse) if return_lse else out
