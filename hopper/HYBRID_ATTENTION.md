# Window attention + delayed recurrent memory

This branch implements an LM-only within-layer hybrid. Each layer has its own
`window_size=X` and `num_sink_tokens=N`, with `R=X-N >= 1`. `R` **includes the
current token**. With zero-based query position `t`, the disjoint sets are:

* sink: `0 <= j < N` and `j <= t`;
* recent: `max(N, t-R+1) <= j <= t`;
* recurrent middle: `N <= j <= t-R`.

The middle first becomes nonempty at `t=X` (the `(X+1)`th token). Sink tokens
never enter the recurrent state. A non-sink token is written once, when evicted.
Sink and recent Q/K both use RoPE; recurrent Q/K use their shared pre-RoPE
representations. There are no auxiliary losses, full-attention teachers,
curriculum schedules, or eviction-consistency objectives in this implementation.

## Unified softmax and the FA3 change

For each query head the recurrent read predicts a value `mu_M` and log-mass
`ell_M = log(number_of_middle_tokens) + Delta(q, state_read)`. The count is the
actual number of evicted tokens, without decay. Empty memory has mass zero
(`ell_M=-inf`). The learnable correction is initially zero.

For exact attention statistics `(o_E, ell_E)` the output is

```text
ell = logaddexp(ell_E, ell_M)
o   = exp(ell_E - ell) * o_E + exp(ell_M - ell) * mu_M
```

This is precisely one softmax over `[sink logits, recent logits, memory logit]`.
It does not imply that the learned memory exactly reproduces full attention.
Exact equivalence would additionally require true middle LSE and conditional
value, which an LM-only recurrent model must learn to approximate.

The implementation uses up to three existing FA3 calls: a causal sink prefix,
a causal recent suffix with left window `R-1`, and attention from the suffix to
all sinks. These partitions are disjoint and are merged in FP32. The attention
score matrices are never materialized on the FA3 path. No new sparse-mask tile
scheduler is needed for this version. The CPU reference backend does materialize
dense scores and is only intended for small correctness tests.

**The CUDA backward is modified.** Previously `return_attn_probs=True` exposed
LSE but ignored its gradient. Merging differentiable softmax partitions requires
that gradient. FA3 now accepts optional FP32 `dsoftmax_lse` and preprocesses

```text
delta = dot(dO, O) - dLSE
dS    = P * (dO @ V.T - delta)
```

Dense, QKV-packed, varlen, and the registered low-level forward propagate LSE
gradients, including losses using only LSE. Both `flash_api.cpp` and the PyTorch
2.9+ stable-ABI binding are updated. LSE uses natural logarithms; no extra `ln(2)`
factor belongs in the new subtraction. Existing output-only and max-logit paths
retain their behavior. ROCm LSE differentiation is explicitly unsupported.

## Layer API

```python
import torch
from flash_attn_3.recurrent import WindowRNNAttention

layer = WindowRNNAttention(
    d_model=1024,
    num_heads=8,
    num_kv_heads=8,          # GQA also supported
    head_dim=128,
    value_dim=128,
    window_size=1024,       # total exact budget, including sinks
    num_sink_tokens=16,     # recent size is 1008, including the current token
    rnn_type="gdn",        # "gdn", "kda", or "gdn2"
    rope_theta=10000.0,
    rotary_dim=128,         # even; 0 disables RoPE, partial RoPE also supported
    backend="fa3",         # explicit "reference" for small CPU tests
).cuda()

x = torch.randn(2, 8192, 1024, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = layer(x)            # [2, 8192, 1024]; shared output projection included
loss = y.float().square().mean()  # replace with the model's next-token CE
loss.backward()
```

Keep parameters FP32 and use autocast for mixed precision, including decay and
mass parameters. Q/K/V/O projections are shared between paths. Recurrent Q/K
are L2-normalized (`rsqrt(sum(x*x)+1e-6)`), with read scale `1.0`; exact attention
uses ordinary `1/sqrt(head_dim)` QK scaling. There is no separate recurrent QKV,
short convolution, SiLU transform of shared KV, or independently projected output.
By default, only decay/erase/write gates and the log-mass readout add parameters.

Two independent ablation options are available (both disabled by default):

- `sink_position_mode="window"`: exact attention uses the positions obtained by
  reindexing the retained cache, including the current token. Sink keys keep
  positions `0..N-1`, while the query position against sinks is `min(t, X-1)`.
  Recent query/key relative distances are unchanged by a common position shift,
  so that partition retains ordinary RoPE. Recurrence and memory mass are unchanged.
- `memory_output_gate=True`: apply `SiLU(W_g x_t)` elementwise to the recurrent
  readout before softmax fusion. This adds a bias-free `d_model -> Hq * Dv`
  projection, without RMSNorm. Memory log-mass uses the **ungated** readout.
  Gate projection is evaluated only for queries with nonempty memory.

Both options support packed-document resets and the reference `step()` path.
`test_window_ablations.py` checks explicit cache reindexing, gradients, decoding,
gate isolation, and CUDA FA3/FLA output and parameter gradients.

Let `k` be normalized, `g` log-decay, `b` erase, and `w` write. The common update is

```text
S_decay = diag(exp(g)) S
S_new   = S_decay + k (w * v - (b * k)^T S_decay)^T
mu      = q^T S_new
Delta   = dot(q, mass_q) + dot(mu, mass_value) + mass_bias
```

GDN uses scalar `g` and scalar `b=w=beta` per head. KDA uses channel-wise `g`
and scalar `b=w=beta`. GDN2 uses channel-wise `g`, key-channel erase `b`, and
value-channel write `w`. Gates come from the **evicted token**, not the current
query. During training, shifted query sequence `q[:,X:]` is paired with
`k/v[:,N:T-R]` in FLA chunk kernels; this reads the correct prefix state without
materializing every token's state matrix. Sink tokens are never in that scan.

For grouped query attention, training repeats KV and gates across query groups
because these FLA operators require aligned query/write head counts. Parameters
remain shared, but recurrent computation and transient state storage are repeated.
The decoding cache stores only one state per KV head.

## Different windows by layer

```python
from torch import nn

windows = [1024, 1024, 1024, 4096] * 4
mixers = nn.ModuleList([
    WindowRNNAttention(
        1024, 8, 128, window_size=window, num_sink_tokens=16,
        rnn_type="kda",
    )
    for window in windows
])
```

Insert each mixer into the surrounding model's existing norm/residual/FFN block.
This repository provides the attention layer, not an OLMo training recipe.

## Packed documents and decoding

`layer(x, cu_seqlens=...)` accepts flattened `[1,total_tokens,d_model]` input with
strictly increasing boundaries starting at 0 and ending at `total_tokens`.
Each document resets sinks, RoPE positions, and recurrent state. The first
implementation loops over documents and reads boundaries on the CPU, so it has
synchronization/launch overhead; it is not a fused varlen hybrid implementation.
Padding masks and zero-length documents are not accepted by this interface.

`layer.step(x[:,t:t+1], cache)` returns `(output, new_cache)` for one-token
inference. Start with `cache=None` and keep a separate cache for every layer.
The cache contains at most `X` exact KV records plus a `[B,Hkv,Dk,Dv]` state.
Pending recent records also store the gates needed at eviction. This is a
bounded-memory PyTorch decoding implementation, not a fused/paged-cache kernel.
It is inference-only; do not reuse caches after updating model parameters.
Changing window sizes during an existing cache's lifetime is rejected.

## Lower-level API

```python
from flash_attn_3.hybrid import window_memory_attention

out, lse = window_memory_attention(
    q_rope, k_rope, v, memory_value, memory_log_mass,
    window_size=1024, num_sink_tokens=16, return_lse=True,
)
```

Q/K/V have shape `[B,T,Hq/Hkv,Dk/Dv]`. Memory values are `[B,T,Hq,Dv]` and
log-masses `[B,T,Hq]`; returned LSE is `[B,T,Hq]` (the wrapper layout).
Memory must already be computed from the allowed evicted prefix. The wrapper
masks absent memory rows but cannot validate the provenance of supplied states.
Self-attention is supported; the wrapper does not expose cross-attention, dropout,
softcap, FP8, paged caches, context parallelism, or higher-order gradients.
Use head dimensions supported by the installed FA3 build (multiples of 8,
at most 256, subject to its QK/V dimension combinations).

## Install and validate on the GPU server

Rebuild the native extension: copying Python files on top of an older `_C` is
insufficient. Preserve the server's existing compatible PyTorch/CUDA/Triton stack.

```bash
git clone --branch codex/window-rnn-unified-softmax --recursive \
  https://github.com/JT-Ushio/flash-attention-max-logits.git
cd flash-attention-max-logits
git submodule update --init --recursive
cd hopper
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
  python -m pip install --no-build-isolation --no-deps .

# FLA reference API pinned to the same revision used by the AHN backend.
# Install its declared requirements in the server environment first if absent;
# --no-deps intentionally does not replace the existing CUDA/Triton stack.
python -m pip install --no-build-isolation --no-deps \
  'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@35dceaee5408e69a555fec34cb215c93c375dabe'
PYTHONPATH="$PWD" python -m pytest -q test_hybrid_attention.py test_hybrid_attention_cuda.py

# Existing max-logit regression tests in this fork:
PYTHONPATH="$PWD" python -m pytest -q test_flash_attn.py -k max_logits
```

The CUDA suite checks the extension's new operator schema, FP16/BF16 hybrid
outputs and all five input gradients against a dense oracle, LSE-only/mixed/output
losses across the public and raw APIs, varlen offsets, softcap/window derivatives,
and FLA layer input/parameter gradients for all three recurrent variants. FLA
tests skip if FLA is absent; a server validation is complete only when they run.
Run on Hopper with a compatible CUDA toolkit as required by FA3.

Local CPU checks, from repository root:

```bash
PYTHONPATH=hopper python -m pytest -q \
  hopper/test_hybrid_attention.py hopper/test_hybrid_attention_cuda.py
python -m ruff check hopper/flash_attn_3/hybrid.py hopper/flash_attn_3/recurrent.py \
  hopper/test_hybrid_attention.py hopper/test_hybrid_attention_cuda.py
git diff --check
```

### GPU validation record (2026-09-16)

Commit `1768aabf5e8e2241df6b04302ad5fd103d98e63a` was compiled from source
and tested on one NVIDIA H200 (SM90), with PyTorch `2.10.0+cu128`, CUDA toolkit
12.8, Triton 3.6.0, and the server's installed FLA 0.5.2. The loaded extension's
path and `dsoftmax_lse` operator schema were checked before running the tests.
The installed FLA package's originating Git revision was not recorded; the
reference revision in the installation example above is not a claim about that
server artifact.

| Coverage | Passed |
| --- | ---: |
| CPU partition, recurrence, packed-document and decoding checks | 37 |
| CUDA unified softmax output/input gradients, FP16/BF16, head dimensions 64/128 | 20 |
| CUDA differentiable LSE, five APIs, output-only/LSE-only/mixed losses | 30 |
| FLA GDN/KDA/GDN2 layer output/input/parameter gradients, BF16, MHA/GQA | 12 |
| Existing max-logits regression (excluding `max_logits_qv`) | 10 |

Both pytest runs finished with **zero failures, errors, or skips**: 99 hybrid
tests in 332.64 seconds (including initial FLA compilation), then 10 max-logits
tests in 3.79 seconds. The scheduler reported success with exit code 0.

This was a targeted Hopper build with equal QK/V dimensions 64 and 128.
SM80, FP8, paged/append KV, other head dimensions, and unequal QK/V dimensions
were disabled for the build; consequently `max_logits_qv` was excluded from
the regression command. The PyTorch 2.9+ stable-ABI binding was exercised; the
older binding was not compiled in this run. These results establish numerical
and gradient agreement for the tested configurations, not long-run training
convergence, throughput, or a measured speedup. This is still a composition of
FA3 calls and FLA kernels, rather than a single fused hybrid kernel.
