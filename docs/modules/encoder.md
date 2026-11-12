# `arc26/hyper/encoder.py`: frozen-base pair encoder and embedding cache

Task P3.2 (BRIEF B2 Q1, B5 P3.2), ADR-001. Our code. Rendering and augmentation call the moved
notebook's own functions in `arc26.nvarc.arc_loader` (`QwenFormatter`, `convert_grid_to_string`,
`ArcDataset.forward_mod`, `permute_mod` through `forward_mod`); nothing of the notebook is
reimplemented. GPU script: `scripts/cache_embeddings.py`
([runbook](../runbooks/cache_embeddings.md)).

## Purpose

H reads a task through a fixed-size embedding of its demonstration pairs (ADR-001). No encoder is
trained: the grid-SFT base model already encodes grid structure, so one prefill of each augmented
pair through the frozen base, pooled over the output-grid tokens, gives the pair embedding. The
module returns the pair embeddings $\bar{e}_i$ and the task embedding $\bar{z}$; the MLP that
maps $\bar{z}$ to $z$ belongs to H (P3.3). Embeddings are cached by content, so meta-training
(P3.5) and the refinement loop (P4) encode each pair once.

## Math

A pair $i$ is rendered in the notebook's training format under each view $a$ of a fixed view set
$A$:

```text
<|im_start|>user\n{input}<|im_end|><|im_start|>assistant\n{output}<|im_end|>
```

where a grid is its rows of digits joined by `\n`. Let $h^{(L)}_t \in \mathbb{R}^{2560}$ be the
residual stream at position $t$ after $L$ decoder layers (Hugging Face's `hidden_states[L]`), and
$O_{i,a}$ the positions of the output-grid tokens of pair $i$ under view $a$. Then

$$e_{i,a} = \frac{1}{|O_{i,a}|} \sum_{t \in O_{i,a}} h^{(L)}_t, \qquad
\bar{e}_i = \frac{1}{|A|} \sum_{a \in A} e_{i,a}, \qquad
\bar{z} = \frac{1}{n} \sum_{i=1}^{n} \bar{e}_i .$$

- $\bar{z}$ is a symmetric function of the pairs, so it is invariant to their order. In floating
  point, a different pair order changes the summation order and the batch composition; the CPU
  test bounds the change by $10^{-5}$ (measured: $1.9 \times 10^{-9}$ on the tiny fp32 model).
- Every sum accumulates in fp32, whatever the model's dtype.
- `pool="input"` replaces $O_{i,a}$ by the input-grid positions $I_{i,a}$: P3.7's overturn test
  (ADR-001).
- $A$ has $8 \cdot n_{\text{perm}}$ views. The 8 geometric transforms are the dihedral group of
  the square: the identity, transpose, and one to three quarter turns of each, in the order
  `ArcDataset.augment` builds them. Each transform $g$ is paired, in each of $n_{\text{perm}}$
  rounds $r$, with its own color permutation $\pi_{r,g}$ of $\{0, \dots, 9\}$, drawn from
  `RandomState(seed)` in augment's order. The default ($n_{\text{perm}} = 2$, seed 2) is the
  decode path's set (`augment(n=2, seed=2)` in `arc_solver.worker`); for a task with one test
  input the two sets are identical, which run A of `P0.1-01` confirms on Kaggle (test
  `test_default_views_are_the_views_run_a_decoded_on_kaggle`).

**Token positions.** Let $s$ be the pair text, $\tau$ the tokenizer, and $c$ a character offset.
The prefix $s_{<c}$ ends on a token boundary when $\tau(s_{<c})$ is a prefix of $\tau(s)$; its
token count $n(c) = |\tau(s_{<c})|$ is then the token offset of $c$. With $c_1, c_2$ the start and
end of the input-grid text and $c_3, c_4$ those of the output-grid text,

$$I = [\,n(c_1),\, n(c_2)\,), \qquad O = [\,n(c_3),\, n(c_4)\,).$$

`render_pair` raises `ValueError` when a prefix is not a token prefix, when either span fails
to decode back to its grid through the notebook's `QwenFormatter.convert_tokens_to_array`, and
when a span's decoded text, spaces removed, differs from the grid's text. The last check is
needed because the notebook's decoder strips whitespace and skips non-digits: a tokenizer whose
normalizer drops the newline at the end of a prefix starts both spans one token early, on the
newline, and only the exact text shows it. No assumption of one token per character is made.
With the notebook's vocabulary (one token per digit and per newline), a $30 \times 30$ pair is
$8 + 2 \cdot (900 + 29) = 1866$ tokens.

## Interfaces

```python
ArcPair = Mapping[str, Any]          # {"input": grid, "output": grid}; grid: 2-D ints 0..9

def pair_hash(pair: ArcPair) -> str  # sha256 hex of b'{"input":[[..]],"output":[[..]]}'
def view_descriptors(n_perm: int = 2, seed: int = 2) -> list[str]
def view_set_id(views: Sequence[str]) -> str                    # 16 hex characters
def apply_view(pair: ArcPair, descriptor: str) -> dict[str, np.ndarray]
def augmented_views(pair, n_perm=2, seed=2) -> list[tuple[str, dict[str, np.ndarray]]]

class RenderedPair(NamedTuple):
    token_ids: list[int]             # [seq]
    output_positions: range          # O
    input_positions: range           # I
def render_pair(pair: ArcPair, formatter: QwenFormatter) -> RenderedPair

class PairEncoder:
    def __init__(self, model, tokenizer, layer=24, pool="output", views=None,
                 batch_tokens=16384, timing=None)
    def hidden_states(self, input_ids, attention_mask) -> Tensor   # [batch, seq, hidden]
    def encode_views(self, pairs, puzzle_id=None) -> Tensor        # [n, n_views, hidden], e
    def encode_pairs(self, pairs, puzzle_id=None) -> Tensor        # [n, hidden], e_bar
    def task_embedding(self, pairs, puzzle_id=None) -> Tensor      # [hidden], z_bar

class EmbeddingCache:
    def __init__(self, root, *, model_id: str, layer: int, pool: str, view_set_id: str)
    def missing(self, pair_hashes) -> list[str]
    def put(self, means: Mapping[str, Tensor], views: Mapping[str, Tensor] | None = None) -> int
    def get(self, pair_hashes) -> Tensor          # [n, hidden], fp32
    def get_views(self, pair_hashes) -> Tensor    # [n, n_views, hidden], fp32
    # also: len(), `in`, has_views, hidden_size, n_views, size_bytes
```

- `pair_hash` canonicalizes lists and integer arrays of any dtype alike and ignores keys other
  than `input` and `output`; a ragged, empty, non-2-D, non-integer, or out-of-range grid raises
  `ValueError`. P3.4's leakage check can reuse it.
- `PairEncoder` validates $0 \le L <$ `num_hidden_layers` (the last `hidden_states` entry is after
  the final norm, not the residual stream, so it is excluded), `pool`, `batch_tokens`, and every
  view descriptor. Outputs are fp32 on the CPU, ordinary tensors (not inference tensors), so H
  can train on them. A non-finite embedding raises `FloatingPointError`.
- Timing: each `encode_views` call sends two records to the optional `timing` callback,
  `{"puzzle_id", "op", "seconds", "tokens"}` (BRIEF B4) with `op` in `encode_render` (CPU:
  views, text, tokens, span checks) and `encode_prefill` (forward and pooling, synchronized by
  the copy of the result to the CPU); `tokens` counts real tokens over all views. The default is
  no callback. `arc26/timing.py` (P0.3) is on another branch; once it merges, its JSONL logger is
  the callback to pass. The GPU script passes its own small appender (`TimingLog`) until then.

## Data flow

```text
pairs -> apply_view (|A| views each) -> render_pair -> sort by length
      -> batches with rows x width <= batch_tokens, right-padded, attention mask
      -> decoder layers 1..L (the hook on layer L stops the forward)
      -> fp32 mean over each sequence's own span -> e [n, |A|, hidden]
      -> mean over views -> e_bar [n, hidden] -> mean over pairs -> z_bar [hidden]
e_bar, e -> EmbeddingCache.put -> root/<pool>-L<layer>-<digest>/shard-NNNNNN.safetensors
                                  and manifest.json
```

- **First $L$ layers only.** A forward hook on decoder layer $L$ (the embedding layer when
  $L = 0$) stores its output and raises a private exception, which ends the forward: later
  layers, the final norm, and the LM head never run, and the hook is removed in a `finally`.
  Output recording is passed as off, because transformers 4.55's `check_model_inputs` restores the
  forwards it patches for recording only on a normal return.
- **Batching.** Sequences are sorted by length, longest first, and grouped greedily so that rows
  times the longest length stays within `batch_tokens`; a longer sequence runs alone. Rows are
  right-padded and masked. Right padding keeps every real token's position id and attention
  context identical to an unbatched run, and pooling slices each row's own span, so padded
  positions are never read.
- **Cache layout.** The namespace (schema, model id, layer, pool, view-set id) is stored in
  `namespace.json` and its sha256 names the directory, so another layer or view set never mixes
  with this one. Each `put` writes one immutable shard (tensors `<hash>`, `[hidden]`, and
  optionally `<hash>/views`, `[n_views, hidden]`, fp32), then rewrites `manifest.json` (pair hash
  to shard, shard byte sizes, `hidden_size`, `n_views`). Both writes go to a temporary file,
  `fsync`, and `os.replace`; the manifest goes last, so a crash leaves at most an unreferenced
  shard that the next `put` overwrites.
- **Size.** At hidden size 2560: 10,240 bytes of tensor data per pair for $\bar{e}_i$ and 163,840
  for the 16 per-view rows. Measured with this code at that width: 174,354 bytes per pair in the
  shard with views (10,371 without), plus about 118 bytes of manifest.

## Tests

`tests/test_encoder.py` (48 tests, skipped without torch, transformers, or safetensors), on the
tiny random Qwen3 (hidden 64, 4 layers, 4 heads, 2 KV heads, head size 16, vocabulary 16, eager)
and the stand-in tokenizer with the notebook's 16 ids, both built in code by
`scripts/cache_embeddings.py`:

- the stand-in tokenizer gives the notebook's ids and refuses text outside the format;
- `pair_hash`: the documented encoding, container and dtype independence, swapped, transposed,
  and reshaped pairs differ, invalid grids raise;
- views: equal to `ArcDataset.augment(n=2, seed=2)` (descriptors and grids) without touching
  numpy's global generator; equal to the views run A decoded on Kaggle; counts and ids;
- `render_pair`: spans decode to the grids for $1 \times 1$, $3 \times 4$, and $30 \times 30$
  pairs, sit between `assistant\n` and `<|im_end|>`, and the token count follows the formula; a
  tokenizer that breaks the prefix property, misdecodes a span, or strips the newline before a
  grid (spans one token early) raises;
- forward: the captured tensor equals the full forward's `hidden_states[L]` bitwise for every
  $L$ on a padded batch; no later layer, norm, or head runs; no hook is left behind;
- pooling: each $e_{i,a}$ equals the mean of the full forward's `hidden_states[L]` over $O$ (or
  $I$); input and output pooling differ; batched equals one-at-a-time (tolerance $10^{-5}$,
  measured $2.6 \times 10^{-8}$); padding replaced by NaN changes nothing; non-finite output
  raises;
- task embedding: order invariance (tolerance $10^{-5}$); $k = 2$; $30 \times 30$ pairs; the
  timing records' schema and token counts; settings validation;
- cache: exact round trip after reopening, resume and idempotent `put`, means without views,
  separate namespaces and a tampered `namespace.json`, refused keys, shapes, and NaN, a failed
  manifest write leaves the old manifest and the next `put` succeeds, encoder output stored
  exactly;
- one GPU test (`@pytest.mark.gpu`): bf16 with SDPA on CUDA, truncation equality and batched
  versus one-at-a-time within bf16 tolerance. Never run locally (BRIEF A0).

The script's tests are in `tests/test_cache_embeddings.py`; section 11 of the
[pre-run brief](../prerun/P3.2-01.md) lists the smoke evidence.

## Known limits

- The real tokenizer is on Kaggle only; its handling of the format is checked first by the
  `P3.2-01` preflight, which renders every view before the model loads.
- In bf16, an embedding depends slightly on the other sequences in its batch (padding width
  changes kernel shapes): cache values are reproducible within bf16 tolerance, not bitwise,
  across different batchings.
- The model is used as given: frozen, in eval mode, with no active adapter. P4.1 must disable
  the TTT adapter around encoding, or the embedding changes.
- One writer per cache namespace; no lock. `put` rewrites the manifest, $O(\text{entries})$ bytes
  per call, so callers write in batches (the script every 256 pairs).
- Grids are at most $30 \times 30$: `convert_tokens_to_array` truncates longer outputs, so
  `render_pair` refuses them.
- Importing the module imports torch, transformers, and safetensors, so `pair_hash` needs them
  too; a torch-free P3.4 would move it to its own module.
- The stand-in model and tokenizer live in the script (`smoke_model`, `smoke_tokenizer`); P3.1
  builds its own tiny Qwen3 in its tests. A shared test-helper module would be their home.
