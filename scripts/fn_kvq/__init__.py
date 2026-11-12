"""4-bit K and V for the QSA layers of Qwen3.8-Flash-Next on the SGLang fork (task P5.5).

Candidate 1 of ``docs/tasks/P5.5.md``: the paged FP8 K and V cache of the 12 QSA layers becomes
INT4 with group-wise FP16 scales (symmetric, groups of 32, a switchable 64-point Hadamard
rotation), quantized on write and dequantized where the backend gathers the selected top-k
tokens into the BF16 FlashAttention 2 scratch (and where chunked prefill gathers a request's
full context). Mode ``v`` keeps K in FP8 and packs V only.

Modules:

- ``layout``: the packed row format, the modes and the byte accounting (pure Python, no torch).
- ``reference``: the pure-PyTorch rotate, quantize, dequantize, write and gather (CPU or GPU).
- ``triton_kernels``: the sm89 Triton kernels with the same semantics, and dispatchers that fall
  back to the reference off the GPU.
- ``sglang_kv4``: the pool subclass and the two gather shims the hash-guarded edits of
  ``configs/P5.5-01.yaml`` import (needs sglang; imported only inside the patched engine).
- ``bench``: the standalone one-GPU benchmark and correctness check.

Shipping: the patched engine imports ``fn_kvq`` as a top-level package, so the folder holding
``fn_kvq/`` must be on the engine's ``sys.path`` (``PYTHONPATH`` of the server process; on Kaggle
the code dataset beside ``fn_probe.py``, on Modal the job's code mount). The arc26 wheel is not
changed: the competition path's wheel stays byte for byte what it is.
"""

from fn_kvq.layout import (
    DEFAULT_GROUP,
    ENV_GROUP,
    ENV_HADAMARD,
    ENV_MODE,
    HADAMARD_BLOCK,
    MODES,
    KV4Layout,
    bytes_per_token,
    group_from_env,
    hadamard_from_env,
    head_dim_from_row,
    mode_from_env,
    row_bytes,
)

__all__ = [
    "DEFAULT_GROUP",
    "ENV_GROUP",
    "ENV_HADAMARD",
    "ENV_MODE",
    "HADAMARD_BLOCK",
    "MODES",
    "KV4Layout",
    "bytes_per_token",
    "group_from_env",
    "hadamard_from_env",
    "head_dim_from_row",
    "mode_from_env",
    "row_bytes",
]
