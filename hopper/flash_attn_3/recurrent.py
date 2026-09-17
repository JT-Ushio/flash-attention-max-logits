"""Shared-projection window/RNN layer with delayed GDN, KDA or GDN2 writes."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .hybrid import _acc_dtype, _validate_window, window_memory_attention


def _normalize(x):
    y = x.to(_acc_dtype(x))
    return (y * torch.rsqrt(y.square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


def _rotary(x, positions, theta, rotary_dim):
    if rotary_dim == 0:
        return x
    dtype = _acc_dtype(x)
    freq = theta ** (
        -torch.arange(0, rotary_dim, 2, device=x.device, dtype=dtype) / rotary_dim
    )
    angle = positions.to(dtype)[:, None] * freq[None, :]
    cos, sin = angle.cos()[None, :, None], angle.sin()[None, :, None]
    a, b = x[..., :rotary_dim].to(dtype).chunk(2, dim=-1)
    rotated = torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(x.dtype)
    return torch.cat((rotated, x[..., rotary_dim:]), -1)


def _update(state, k, v, g, erase, write):
    """GDN2 update; scalar erase=write recovers KDA and scalar decay GDN."""
    state = state * g.exp().unsqueeze(-1)
    prediction = ((erase * k).unsqueeze(-1) * state).sum(-2)
    return state + k.unsqueeze(-1) * (write * v - prediction).unsqueeze(-2)


def recurrent_reference(q, k, v, g, erase, write):
    """Differentiable sequential oracle on already-normalized Q/K [B,T,H,D]."""
    dtype = _acc_dtype(q)
    q, k, v, g, erase, write = (x.to(dtype) for x in (q, k, v, g, erase, write))
    state = k.new_zeros(k.shape[0], k.shape[2], k.shape[3], v.shape[3])
    outputs = []
    for i in range(q.shape[1]):
        state = _update(state, k[:, i], v[:, i], g[:, i], erase[:, i], write[:, i])
        outputs.append((q[:, i].unsqueeze(-1) * state).sum(-2))
    return torch.stack(outputs, dim=1)


@dataclass
class WindowRNNCache:
    """One cache per layer/sequence batch, holding at most X exact KV tokens."""

    position: int
    state: torch.Tensor
    sinks: list
    recent: list
    window_size: int
    num_sink_tokens: int


class WindowRNNAttention(nn.Module):
    """LM-only layer: shared Q/K/V/O, exact RoPE, recurrent NoPE.

    ``backend='fa3'`` uses CUDA FA3 and FLA chunk kernels for training/prefill.
    ``backend='reference'`` is an explicit small-input oracle. No auxiliary loss,
    short convolution, independent recurrent QKV or learned fusion gate is used.
    GQA training repeats the recurrent KV/gates across query groups for FLA;
    step() keeps one persistent state per KV head. Packed forward currently calls
    each document separately (correctness first; no cross-document states/RoPE).
    ``sink_position_mode='window'`` uses cache-relative RoPE distances to sinks.
    ``memory_output_gate=True`` gates only the memory value with SiLU(W_g x);
    log-mass continues to use the ungated readout. Defaults preserve the baseline.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        head_dim,
        *,
        window_size,
        num_sink_tokens=0,
        rnn_type="gdn",
        num_kv_heads=None,
        value_dim=None,
        rotary_dim=None,
        rope_theta=10000.0,
        backend="fa3",
        deterministic=False,
        sink_position_mode="absolute",
        memory_output_gate=False,
    ):
        super().__init__()
        _validate_window(window_size, num_sink_tokens)
        if rnn_type not in ("gdn", "kda", "gdn2"):
            raise ValueError("rnn_type must be gdn, kda or gdn2")
        if backend not in ("fa3", "reference"):
            raise ValueError("backend must be fa3 or reference")
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        value_dim = head_dim if value_dim is None else value_dim
        rotary_dim = head_dim if rotary_dim is None else rotary_dim
        if min(d_model, num_heads, num_kv_heads, head_dim, value_dim) <= 0:
            raise ValueError("model, head and value dimensions must be positive")
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if rotary_dim < 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError("rotary_dim must be even and between 0 and head_dim")
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        self.d_model, self.num_heads, self.num_kv_heads = (
            d_model,
            num_heads,
            num_kv_heads,
        )
        self.head_dim, self.value_dim = head_dim, value_dim
        self.window_size, self.num_sink_tokens = window_size, num_sink_tokens
        self.rnn_type, self.backend = rnn_type, backend
        self.rotary_dim, self.rope_theta = rotary_dim, rope_theta
        self.deterministic = deterministic
        if sink_position_mode not in ("absolute", "window"):
            raise ValueError("sink_position_mode must be absolute or window")
        self.sink_position_mode = sink_position_mode
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * value_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * value_dim, d_model, bias=False)
        self.gate_dim = 1 if rnn_type == "gdn" else head_dim
        self.erase_dim = head_dim if rnn_type == "gdn2" else 1
        self.decay_proj = nn.Linear(d_model, num_kv_heads * self.gate_dim, bias=False)
        self.erase_proj = nn.Linear(d_model, num_kv_heads * self.erase_dim, bias=False)
        self.write_proj = (
            nn.Linear(d_model, num_kv_heads * value_dim, bias=False)
            if rnn_type == "gdn2"
            else None
        )
        self.A_log = nn.Parameter(torch.zeros(num_kv_heads, 1))
        self.dt_bias = nn.Parameter(torch.full((num_kv_heads, self.gate_dim), -4.0))
        # Query/state-conditioned log-mean-exp correction; count prior at init.
        self.mass_q = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.mass_value = nn.Parameter(torch.zeros(num_heads, value_dim))
        self.mass_bias = nn.Parameter(torch.zeros(num_heads))
        self.output_gate_proj = (
            nn.Linear(d_model, num_heads * value_dim, bias=False)
            if memory_output_gate
            else None
        )

    @property
    def recent_size(self):
        return self.window_size - self.num_sink_tokens

    def _project(self, x):
        q = self.q_proj(x).unflatten(-1, (self.num_heads, self.head_dim))
        k = self.k_proj(x).unflatten(-1, (self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).unflatten(-1, (self.num_kv_heads, self.value_dim))
        return q, k, v

    def _gates(self, x):
        dtype = _acc_dtype(x)
        raw = (
            self.decay_proj(x)
            .unflatten(-1, (self.num_kv_heads, self.gate_dim))
            .to(dtype)
        )
        g = -self.A_log.to(dtype).exp() * F.softplus(raw + self.dt_bias.to(dtype))
        erase = (
            self.erase_proj(x)
            .unflatten(-1, (self.num_kv_heads, self.erase_dim))
            .sigmoid()
        )
        write = (
            self.write_proj(x)
            .unflatten(-1, (self.num_kv_heads, self.value_dim))
            .sigmoid()
            if self.write_proj is not None
            else erase
        )
        return g, erase, write

    def _log_mass(self, q, value, count):
        dtype = _acc_dtype(q)
        correction = (q.to(dtype) * self.mass_q.to(dtype)).sum(-1)
        correction = correction + (value.to(dtype) * self.mass_value.to(dtype)).sum(-1)
        return count.to(dtype).log() + correction + self.mass_bias.to(dtype)

    def _recurrent(self, q, k, v, gates):
        groups = self.num_heads // self.num_kv_heads
        k, v, g, erase, write = (
            x.repeat_interleave(groups, dim=2).contiguous() for x in (k, v, *gates)
        )
        if self.backend == "reference":
            return recurrent_reference(q, k, v, g, erase, write).to(q.dtype)
        # Read q[t] immediately after updating with evicted k[t-R]. FLA's query
        # at a step need not be the query of the token being written that step.
        common = dict(
            q=q.contiguous(),
            k=k,
            v=v,
            g=g.squeeze(-1) if self.rnn_type == "gdn" else g,
            scale=1.0,
            use_qk_l2norm_in_kernel=False,
            output_final_state=False,
        )
        if self.rnn_type == "gdn":
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            out, _ = chunk_gated_delta_rule(**common, beta=erase.squeeze(-1))
        elif self.rnn_type == "kda":
            from fla.ops.kda import chunk_kda

            out, _ = chunk_kda(**common, beta=erase.squeeze(-1))
        else:
            from fla.ops.gdn2 import chunk_gdn2

            out, _ = chunk_gdn2(**common, b=erase, w=write)
        return out

    def forward(self, x, *, cu_seqlens=None):
        if x.ndim != 3 or x.shape[-1] != self.d_model or min(x.shape[:2]) <= 0:
            raise ValueError("x must have nonempty shape [B,T,d_model]")
        if cu_seqlens is not None:
            if (
                x.shape[0] != 1
                or cu_seqlens.ndim != 1
                or cu_seqlens.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("packed input requires B=1 and integer 1D cu_seqlens")
            bounds = cu_seqlens.detach().cpu().tolist()
            if (
                len(bounds) < 2
                or bounds[0] != 0
                or bounds[-1] != x.shape[1]
                or any(b <= a for a, b in zip(bounds, bounds[1:]))
            ):
                raise ValueError("cu_seqlens must strictly partition [0,T]")
            return torch.cat(
                [self(x[:, a:b]) for a, b in zip(bounds, bounds[1:])], dim=1
            )
        if self.backend == "fa3" and (x.device.type != "cuda"):
            raise ValueError("fa3 backend requires CUDA")
        q, k, v = self._project(x)
        qn, kn = _normalize(q), _normalize(k)
        b, t = x.shape[:2]
        # First evicted token is j=N, first query with memory is t=N+R=X.
        start = self.window_size
        mv = q.new_zeros(b, t, self.num_heads, self.value_dim)
        ml = q.new_full((b, t, self.num_heads), -torch.inf, dtype=_acc_dtype(q))
        if t > start:
            source = slice(self.num_sink_tokens, t - self.recent_size)
            read = self._recurrent(
                qn[:, start:], kn[:, source], v[:, source], self._gates(x[:, source])
            )
            count = torch.arange(1, t - start + 1, device=x.device)[None, :, None]
            memory_value = read
            if self.output_gate_proj is not None:
                gate = F.silu(self.output_gate_proj(x[:, start:])).unflatten(
                    -1, (self.num_heads, self.value_dim)
                )
                memory_value = read * gate
            mv = torch.cat((mv[:, :start], memory_value), dim=1)
            ml = torch.cat(
                (ml[:, :start], self._log_mass(qn[:, start:], read, count)), dim=1
            )
        positions = torch.arange(t, device=x.device)
        qr = _rotary(q, positions, self.rope_theta, self.rotary_dim)
        kr = _rotary(k, positions, self.rope_theta, self.rotary_dim)
        # Cache reindexing preserves recent relative distances. Only the sink
        # partition needs a query with position capped at X-1 (includes self).
        sink_q = (
            _rotary(
                q,
                positions.clamp(max=self.window_size - 1),
                self.rope_theta,
                self.rotary_dim,
            )
            if self.sink_position_mode == "window"
            else None
        )
        out = window_memory_attention(
            qr,
            kr,
            v,
            mv,
            ml,
            window_size=self.window_size,
            num_sink_tokens=self.num_sink_tokens,
            backend=self.backend,
            deterministic=self.deterministic,
            sink_q=sink_q,
        )
        return self.o_proj(out.flatten(-2))

    @torch.no_grad()
    def step(self, x, cache=None):
        """One-token decode oracle with bounded KV/state storage (PyTorch ops).

        x: [B,1,d_model]. Returns (output, new_cache). This intentionally uses
        ordinary tensor operations, not a fused decoding kernel. Reset cache for
        each new document, and never reuse a cache after changing layer weights.
        """
        if x.ndim != 3 or x.shape[1:] != (1, self.d_model):
            raise ValueError("step expects [B,1,d_model]")
        if cache is None:
            cache = WindowRNNCache(
                0,
                x.new_zeros(
                    x.shape[0],
                    self.num_kv_heads,
                    self.head_dim,
                    self.value_dim,
                    dtype=_acc_dtype(x),
                ),
                [],
                [],
                self.window_size,
                self.num_sink_tokens,
            )
        if (cache.window_size, cache.num_sink_tokens) != (
            self.window_size,
            self.num_sink_tokens,
        ):
            raise ValueError("cache window configuration differs from this layer")
        if (
            cache.state.shape
            != (x.shape[0], self.num_kv_heads, self.head_dim, self.value_dim)
            or cache.state.device != x.device
        ):
            raise ValueError("cache shape/device differs from the input/layer")
        q, k, v = self._project(x)
        qn, kn = _normalize(q), _normalize(k)
        pos = torch.tensor([cache.position], device=x.device)
        qr, kr = (_rotary(y, pos, self.rope_theta, self.rotary_dim) for y in (q, k))
        sinks, recent = list(cache.sinks), list(cache.recent)
        state = cache.state
        if cache.position < self.num_sink_tokens:
            sinks.append((kr, v))
        else:
            recent.append((kr, kn, v, *self._gates(x)))
            if len(recent) > self.recent_size:
                _, old_k, old_v, g, erase, write = recent.pop(0)
                state = _update(
                    state,
                    *(y[:, 0].to(state.dtype) for y in (old_k, old_v, g, erase, write)),
                )
        exact_k = torch.cat(
            [kv[0] for kv in sinks] + [entry[0] for entry in recent], dim=1
        )
        exact_v = torch.cat(
            [kv[1] for kv in sinks] + [entry[2] for entry in recent], dim=1
        )
        groups = self.num_heads // self.num_kv_heads
        scores = (
            torch.einsum(
                "bthd,bshd->bths",
                qr.to(state.dtype),
                exact_k.repeat_interleave(groups, 2).to(state.dtype),
            )
            * self.head_dim**-0.5
        )
        count = cache.position + 1 - self.window_size
        if self.sink_position_mode == "window" and sinks:
            sink_q = _rotary(
                q, pos.clamp(max=self.window_size - 1), self.rope_theta, self.rotary_dim
            )
            sink_k = exact_k[:, : len(sinks)].repeat_interleave(groups, 2)
            sink_scores = (
                torch.einsum(
                    "bthd,bshd->bths", sink_q.to(state.dtype), sink_k.to(state.dtype)
                )
                * self.head_dim**-0.5
            )
            scores = torch.cat((sink_scores, scores[..., len(sinks) :]), dim=-1)
        if count > 0:
            grouped_q = (
                qn[:, 0]
                .to(state.dtype)
                .reshape(x.shape[0], self.num_kv_heads, groups, self.head_dim)
            )
            read = (
                torch.einsum("bhgk,bhkv->bhgv", grouped_q, state)
                .reshape(x.shape[0], 1, self.num_heads, self.value_dim)
                .to(q.dtype)
            )
            mass = self._log_mass(qn, read, x.new_tensor(count))
            if self.output_gate_proj is not None:
                gate = F.silu(self.output_gate_proj(x)).unflatten(
                    -1, (self.num_heads, self.value_dim)
                )
                read = read * gate
            scores = torch.cat((scores, mass.unsqueeze(-1)), dim=-1)
            values = torch.cat(
                (
                    exact_v.repeat_interleave(groups, 2).transpose(1, 2),
                    read.transpose(1, 2),
                ),
                dim=2,
            )
        else:
            values = exact_v.repeat_interleave(groups, 2).transpose(1, 2)
        out = torch.einsum(
            "bths,bhsv->bthv", scores.softmax(-1), values.to(state.dtype)
        ).to(q.dtype)
        new_cache = WindowRNNCache(
            cache.position + 1,
            state,
            sinks,
            recent,
            self.window_size,
            self.num_sink_tokens,
        )
        return self.o_proj(out.flatten(-2)), new_cache
