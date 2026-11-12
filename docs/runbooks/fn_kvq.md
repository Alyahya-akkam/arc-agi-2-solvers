# Runbook: the KV4 cache kernels and their benchmark (`scripts/fn_kvq/`)

Task P5.5, candidate 1 (`docs/tasks/P5.5.md`): INT4 K and V for the 12 QSA layers of
Qwen3.8-Flash-Next on the SGLang fork `jpezzulli/sglang-rtxpro6000` at `d00d88e`. The package
`scripts/fn_kvq/` holds the format (`layout.py`), the PyTorch reference (`reference.py`), the sm89
Triton kernels (`triton_kernels.py`), the pool and the two gather shims the engine imports
(`sglang_kv4.py`), and the standalone one-GPU benchmark (`bench.py`). The hash-guarded edits that
wire it into the engine are `configs/P5.5-01.yaml`, `patches.items`; the tests are
`tests/test_fn_kvq.py`. The Modal run that executes the benchmark and the model-level A/B is
`P5.5-01` (`docs/runbooks/fn_kvab.md`, `docs/prerun/P5.5-01.md`).

## 1. Why this script exists

Revision 15's FP8 pool on one L4 is about 169,728 tokens at 6,912 bytes per token per GPU: K 3,072
and V 3,072 in FP8 (12 layers x 1 KV head x 256) plus the indexer cache's 768
(`docs/research/fn_l4_memory.md` on `task/ideas`). The planner's stream count and thinking budget
follow the pool. INT4 K and V with group-wise scales cut the K and V part to 3,456 bytes, 4,224 in
all, so the same memory holds about 277,696 tokens (1.64x). The decode gather then reads 144 bytes
per selected token instead of 256 and writes the same BF16 scratch as today.

## 2. What it does, step by step

Format (`layout.py`). One row per token and KV head: $R = D/2 + 2D/G$ bytes, 144 at $D = 256$,
$G = 32$. Bytes $[0, 128)$ hold the nibbles (byte $j$: element $2j$ low, $2j+1$ high, unsigned
$u = q + 8$); bytes $[128, 144)$ hold the 8 FP16 scales. Symmetric per group:
$s_g = \mathrm{fp16}(\max_i |x_{g,i}| / 7)$, $q = \mathrm{clip}(\mathrm{round}(x / s_g), -8, 7)$,
round half to even, so $|x - s_g q| \le s_g / 2$ up to the FP16 rounding of $s_g$.

Rotation (default on, `ARC26_KV4_HADAMARD`). Before quantizing, each block of 64 channels is
multiplied by $H_{64}/8$, the normalized Sylvester Hadamard matrix (symmetric, its own inverse);
the gather multiplies the dequantized block by it again. Attention is unchanged since
$q^\top k = (Hq)^\top(Hk)$ holds for every orthogonal $H$ and the output is $V$'s rows combined
linearly; a per-channel outlier spreads over 64 channels so one group's absmax no longer wastes the
other channels' levels.

Modes (`ARC26_KV4`): `1` or `kv` packs K and V; `v` packs V only and keeps K in the engine's KV
dtype (FP8), 5,568 bytes per token, about 210,688 tokens; `0` or unset boots the FP8 path of
revision 16 byte for byte (nothing of `fn_kvq` is imported).

Kernels (`triton_kernels.py`), all plain loads and stores plus one 64 x 64 FP32 `tl.dot` per
block for the rotation:

1. `kv4_write`: the pool's `set_kv_buffer`. Grid (tokens / 16, heads); per block of 64 channels:
   load, rotate, group absmax, FP16 scale, round, clip, pack two nibbles per byte, store 32 bytes
   and the block's scales. Slots outside `[0, size + page_size)` are skipped as the fork's writer
   skips them (CUDA-graph padding rows write slot 0's page).
2. `kv4_compact_gather`: the decode gather, the fork's `_compact_kv`
   (`qsa/sparse_attn.py:369-406`) with dequantization: grid (batch, heads, top-k / 16); the same
   validity rule (`cols < cu_k[b+1] - cu_k[b]`, `0 <= position < seq_len`), the same
   `req_to_token` lookup, the same scratch rows; invalid rows are left untouched. In mode `v` the
   K row is decoded from its FP8 bytes in the kernel. Captured in the decode CUDA graph like the
   fork's kernel: no allocation, no host sync.
3. `kv4_gather_rows`: chunked prefill's staging of a request's whole context (the map note's
   second read path, `qwen_sparse_attn_backend.py:1649-1665`): the packed rows at the joined
   `req_to_token` slots dequantized into a new BF16 tensor, 1 KiB per token for K plus V.

Every dispatcher runs the reference instead when the tensors are not on CUDA; under
`TRITON_INTERPRET=1` the kernels run on CPU tensors through Triton's interpreter, which is how
they were checked bit for bit against the reference before any GPU time.

Engine edits (`configs/P5.5-01.yaml`), nine on four files, chained by hash after revision 16's
five. The hashes are the wheel's files (`runs/wheel-peek`, read out of pennyroyal-v253's
`sglang-0.5.19+gd00d88efc8d6` on Kaggle): `kv_cache_configurator.py` differs there from the
clone's `d00d88e` (its `no_intermediate_ssm` lines), the other files are equal. Every edit is
inert with `ARC26_KV4` unset or `0`: the pool-class and cell-size helpers return the fork's own
class and term, the shims are the fork's own calls, the scratch dim and `get_v_head_dim` give
256 as the FP8 buffers do; `tests/test_fn_kvq.py` checks each.

- `kv_cache_configurator.py`: the QSA pool's full-attention sub-pool is `KV4MHATokenToKVPool`
  when `ARC26_KV4` is on. The subclass overrides `_kv_buffer_shapes` (rows of 144 bytes),
  `store_dtype` (`uint8`), the raw getters and `set_kv_buffer`; the base allocates, measures and
  moves rows unchanged.
- `qwen_sparse_attn_backend.py`: the FA2 scratch's head dim from `layer.head_dim` (a packed row
  is not the head dim); both `qwen_sparse_kv_extraction_compact_triton` call sites through
  `_arc26_kv_extract`; the extend `index_select` pair through `_arc26_kv_gather`. The two shims
  read `ARC26_KV4` once at import: off, each is the fork's own call whatever the buffers' dtype;
  on, a `uint8` buffer (the packed rows) goes to `fn_kvq`, a `float8` or BF16 one to the fork.
- `pool_configurator.py`: the full-attention term of the cell size prices the packed rows, so
  `max_total_num_tokens` grows by itself.
- `memory_pool.py`: `HybridLinearKVPool.get_v_head_dim` returns the sub-pool's `v_head_dim`
  (256) rather than the V buffer's last dim (144), which the Triton backend of the linear layers
  reads at init.

Benchmark (`bench.py`): random K and V at the real shapes (169,792 slots, 1 head of 256, 3
streams of 56,576 tokens, top-k 2,051, a 2,048-token prefill chunk), the kernels against the
reference (exact), timings of the FP8 path's write and gathers against the KV4 ones at both
rotation settings and both modes, max, mean and relative RMS error of INT4 and of FP8 against the
BF16 data, one JSON file.

## 3. Why it is built this way

- **Per-token groups of 32 along the head dimension, symmetric, FP16 scales** (llama.cpp
  `q4_0`): the methods note's primary (`docs/research/fn_kv_compress_methods.md` on `task/ideas`,
  `6bb4e15`; KIVI Table 1, QuaRot-KV, arXiv 2504.04823). Per-channel K (KIVI, KVQuant) lost: it
  needs scales across a group of tokens, so a page's late tokens would wait or need an FP16 window;
  the decode write is one token at a time. Asymmetric lost: a zero point per group costs 160
  bytes per row against 144 for under half a bit of resolution on data that is not skewed per
  token (K is RMS-normalized before RoPE; a group of V mixes signs). Group 64 (136 bytes, 1.71x)
  stays one environment variable away (`ARC26_KV4_GROUP`) for the A/B.
- **Hadamard-64 on write and in the gather, not on q and the output**: the gather already
  materializes BF16 rows, so the inverse rotation is one `tl.dot` on data in registers; rotating
  q and the output would touch the model's attention code. The block is 64 so a group never
  straddles one and a 256-wide head is 4 blocks. Evidence: llama.cpp PR 21038 (Qwen3.5-35B
  `q4_0` PPL 6.61 unrotated, 6.60 rotated; gpt-oss-20b AIME25 2.0 unrotated, 21.7 rotated).
- **Dequantization inside the gather kernels, not a dequantized layer view**: the fork's quantized
  pools dequantize a whole layer on every `get_key_buffer` (`memory_pool.py:2440-2478`), about
  3.9 GB of traffic per decode step on this pool (the map note, `fn_kv_compress_map.md`,
  `e6f3e9b`, section 4). The gather reads only the 2,051 selected tokens per row.
- **One packed row per token (nibbles then scales), not the fork's `fp4_mx_block16` buffers**
  (the map note's candidate 1a). The row layout makes every base-class mechanism that counts or
  moves rows correct without an override (`get_kv_size_bytes`, `move_kv_cache`, `get_cpu_copy`,
  the PD descriptors) and the gather reads one contiguous 144-byte row per token. Reusing the
  FP4 recipe would set `is_quantized_kv_cache`, which routes `set_kv_buffer`, the getters, the
  FlashInfer backend and the sizing through the FP4 code (`memory_pool.py:2096-2175`,
  `2495-2527`; `pool_configurator.py:335-343` over-charges 512 bytes per token), and would change
  `--kv-cache-dtype` for the B boot, so the A/B would differ in a flag as well as in the switch.
  Here both boots carry revision 16's flags; only `ARC26_KV4` differs.
- **A subclass selected by one environment variable, edits that stay small**: `apply_patch`
  (`scripts/fn_probe.py`, ADR of recipe R4) applies regex edits with hash guards, so the function
  bodies live in `fn_kvq` and each edit is a call-site rename or a few inserted lines; with
  `ARC26_KV4` unset the shims call the fork's own names and `fn_kvq` is never imported.
- **Round half to even in the kernel** (`_round_half_even`), so the kernel and `torch.round` agree
  bit for bit and the tests can demand equality rather than a tolerance.
- **Triton, not CUDA C++**: the fork's own QSA kernels are Triton, the sm89 wheel builds nothing
  at run time, and `TRITON_INTERPRET=1` gave a CPU check of the kernels before the first GPU
  minute.

## 4. How to run it

CPU smoke (BRIEF A8), about 20 s, no GPU:

```bash
.venv/bin/python scripts/fn_kvq/bench.py --smoke --out runs/P5.5/tmp/bench_smoke.json
.venv/bin/pytest tests/test_fn_kvq.py -m "not gpu"
TRITON_INTERPRET=1 .venv/bin/pytest tests/test_fn_kvq.py -m "not gpu" -k interpreter
```

On a GPU (Modal L40S or L4, through `P5.5-01`; never locally):

```bash
python scripts/fn_kvq/bench.py --out runs/P5.5/bench.json          # the real shapes, 50 iterations
pytest tests/test_fn_kvq.py -m gpu                                  # kernels vs reference, BF16
```

The engine: revision 16's flags unchanged (`--kv-cache-dtype fp8_e4m3`, `--page-size 64`,
`--attention-backend triton`), the `patches.items` of `configs/P5.5-01.yaml` applied at the
install, and `engine.env`: `ARC26_KV4=1` (B), `ARC26_KV4=0` (A, the FP8 baseline),
`ARC26_KV4=v` (the fallback), `ARC26_KV4_HADAMARD=0|1`, `ARC26_KV4_GROUP=32|64`. The folder that
holds `fn_kvq/` must be on the server's `PYTHONPATH` (the config's header says where on Kaggle
and on Modal).

## 5. Outputs

`bench.json`: `device`, `capability`, `torch`, `triton`; `shapes`; `bytes_per_token` per mode
(`off` 6,912, `kv` 4,224, `v` 5,568) and `pool_tokens_at_fp8_pool` (169,728 scaled, page
aligned: 277,696 and 210,688); `checks` per rotation setting: for K and for V
`write_*_rows_equal`, `write_*_nibble_diff_share`, `write_*_nibble_diff_max`,
`write_*_scale_diff_share`, `write_*_dequant_diff_max_steps` (the kernel's rows against the
reference's, nibble by nibble), then `gather_rows_max_diff`, `compact_k_max_diff`,
`compact_v_max_diff`, `mode_v_k_max_diff`;
`errors` (`fp8`, `hadamard0`, `hadamard1`, each with `k` and `v`: `max_abs`, `mean_abs`,
`rel_rms`); `timings` (`write_prefill`, `write_decode`, `gather_decode`, `gather_decode_mode_v`,
`gather_extend`, median and mean microseconds); `ok`. Exit code 0 when every check passed.

The engine's log at boot: `arc26 KV4 cache: mode=kv group=32 hadamard=True` (from
`fn_kvq.sglang_kv4`) and the fork's `KV Cache is allocated ... K size ... V size` line, whose K
and V sizes must be `(size + 64) x 12 x 144` bytes each in mode `kv`; `patches.json` with
`applied` for all fourteen edits.

## 6. How to read the results

- `ok` is the write within one quantization step of the reference on at most 0.5% of the
  nibbles (`nibble_diff_max` at most 1, `dequant_diff_max_steps` at most 1, `nibble_diff_share`
  at most `--max-nibble-diff-share`) and the gathers exact. Without the rotation the write is
  bit-identical (`write_*_rows_equal` true) since both divide in IEEE; with it the 64-term
  `tl.dot` sums in another order than torch's matmul, so a rare element at a rounding boundary
  lands one step away, which is the quantizer's own noise, not an error. P5.5-01 revision 2's
  phase K (L40S, Triton 3.7.1) had the gathers exact and `write_rows_equal` false at both
  settings: Triton's default `div.full.f32` (2 ulp) against torch's IEEE quotient; the kernel
  now uses `tl.fdiv(..., ieee_rounding=True)`. Many nibbles differing, or a difference above one
  step: a kernel bug; do not boot the engine with the switch on.
- `errors.hadamard1.k.rel_rms` below `errors.hadamard0.k.rel_rms`: the rotation helps on the
  benchmark's outlier channels (expected about 0.08 against 0.12 on its synthetic data; FP8 is
  about 0.035). Real keys are measured only by `P5.5-01`'s teacher-forced comparison.
- `timings.hadamard1.gather_decode` against `timings.fp8.gather_decode`: the decode cost of the
  gather, per layer; 12 layers per step. Measured on the L40S (revision 2, phase K): 22.6 us
  FP8, 28.5 us KV4, 37.9 us KV4 with the rotation, so at most $12 \times 15 = 180$ us per
  decode step, 0.7% of a 27 ms step at 36 tokens per second. `gather_extend` runs once per
  prefill chunk and layer: 71 us FP8, 48 us KV4, 266 us with the rotation (the FP32 `tl.dot`
  over 56,576 rows), $12 \times 266 = 3.2$ ms per chunk against a chunk's own seconds of MoE
  prefill, so it is acceptable; the write kernels are at or under the FP8 copy's time.
- The pool: the engine's `max_total_num_tokens` at `ARC26_KV4=1` against the FP8 boot's, expected
  about 1.64x; a 1.0x pool means the cell-size edit did not apply (read `patches.json`).

## 7. Failure modes

- An edit reports `sha256 differs` or `match count`: the wheel is not the fork's `d00d88e`, or
  another edit changed the file first; the KV4 edits add no fallback flag, so boot with
  `ARC26_KV4=0`.
- `ModuleNotFoundError: fn_kvq` in the engine log: the folder is not on the server's
  `PYTHONPATH` (section 4); the FP8 boot is unaffected since the shims import lazily.
- A Triton compile error on the GPU (`tl.split`, `tl.interleave`, `input_precision`): the wheel's
  Triton is older than 3.0; the kernels were written against 3.9 and the interpreter.
- Memory: the extend staging of a long request doubles today's transient (88 MB against 44 MB per
  layer call at 86k tokens, freed at once); the margin after capture was estimated at 0.5 GiB
  (`fn_l4_memory.md`). A `--mem-fraction-static` of 0.95 may need 0.94 if a prefill OOMs.
- Quality: 4-bit K on long reasoning is the open risk (the methods note: length inflation of 43
  to 61% on another model at MXFP4). The decision rule is `P5.5-01`'s teacher-forced top-1
  agreement and the DEV tasks; the fallbacks in order are `ARC26_KV4_GROUP=16`, then
  `ARC26_KV4=v`, then `ARC26_KV4=0`.
- A pool written with one rotation or group setting cannot be read with another: both are read
  once at the engine's import, so they cannot change within a server's life.
