# Runbook: `scripts/agent_spike.py` and `scripts/build_download_notebooks.py`

Task P6.3 (ADR-020; BRIEF A8, A9; `docs/tasks/P6.3.md`). Configs: `configs/P6.3-download.yaml`
(the two download notebooks), `configs/P6.3-01.yaml` and `configs/P6.3-01-script.yaml` (run
`P6.3-01`, stage 1), `configs/P6.3-02.yaml` and `configs/P6.3-02-script.yaml` (run `P6.3-02`,
stage 2, on the merged harness of P6.1 and P6.2). Pre-run briefs:
[`docs/prerun/P6.3-01.md`](../prerun/P6.3-01.md), [`docs/prerun/P6.3-02.md`](../prerun/P6.3-02.md).
Sources: "models note" is `docs/research/qwen38_and_small_models.md`, "design note"
`docs/research/agent_library_design.md`, "first-runs note" `docs/research/kaggle_first_runs.md`.
vLLM facts were read from its `v0.31.0` sources on 9 Oct (`vllm/engine/arg_utils.py`,
`vllm/entrypoints/launchers/cli_args.py`, `vllm/entrypoints/openai/chat_completion/protocol.py`,
`vllm/config/{cache,attention,multimodal}.py`, `vllm/v1/core/kv_cache_utils.py`,
`vllm/v1/metrics/loggers.py`, `vllm/envs.py`).

## 1. Why this script exists

ADR-020 moves the entry toward an agent that writes `transform(grid)` programs with Qwen3.8-27B
and keeps only programs that reproduce every demonstration. Its gate has two halves that no one
has measured on Kaggle's 4x L4: at least about 300 generated tokens per second, and at least
about 5 of the tasks NVARC misses solved by verified programs. The token rate decides the plan
more than any harness choice (design note, 5.5: 64k tokens per task at 400 tokens per second,
322k at 2,000).

- **Run `P6.3-01` (stage 1)** answers the first half and gives an early solve signal: does the
  engine boot offline on the pinned image, how large is its KV pool, how many tokens per second
  does it generate at the agent's concurrency (1 to 64) and contexts (8k and 32k), and does the
  model write verified programs for ARC-AGI-2 training tasks at low effort?
- **Run `P6.3-02` (stage 2)** answers the second half with the real harness (P6.1, P6.2) on
  DEV40, which holds 29 tasks NVARC scored 0 on in `P2.6-01`.

Both feed ADR-020's gate and its overturn rule (if the gate fails, Phase 6 stops at its CPU
parts). Before either can run, the weights (19.47 GB) and the engine (about 4 GB of wheels) must
reach Kaggle's offline sessions without passing through the human's machine (BRIEF A0): that is
the two download notebooks' job.

## 2. What it does, step by step

### 2.1 The download notebooks (CPU, internet on)

`scripts/build_download_notebooks.py` writes two private script kernels, each a generated `.py`
file with its config section baked in as `SETTINGS`, and a `kernel-metadata.json` with internet
on, no accelerator in any field, and the reference image pinned.

1. **`alyahya737/arc26-dl-qwen38-int4`**: checks that `/kaggle/working` has room for the 12
   files (19,472,990,100 bytes) plus a margin; lists the revision's files through the Hub API and
   compares them with the pinned list (names, sizes, LFS sha256, git blob sha1); runs
   `huggingface_hub.snapshot_download` of `RedHatAI/Qwen3.8-27B-INT4` at
   `91bd022d5b49442a868bc35008f6c21e1860edfa` into `/kaggle/working/qwen38-27b-int4/`, up to 3
   attempts (it resumes partial files); removes its `.cache/` bookkeeping; hashes every file once
   for both sha256 and git blob sha1 and compares them with the pins; writes `MANIFEST.json`.
   Any mismatch deletes the folder, so no unverified weights are ever saved.
2. **`alyahya737/arc26-dl-agent-wheels`**: checks that its interpreter is Python 3.11 on x86_64
   (the GPU run's); fetches the pinned `pip` wheel and runs pip from it; writes
   `requirements.txt` (vLLM 0.31.0, httpx 0.28.1, pip 26.2.1) and `constraints.txt` (199 pins);
   runs `pip download --only-binary :all:` under those constraints into
   `/kaggle/working/wheelhouse/`; fetches the optional extra (FlashInfer's sm_89 kernel cache,
   a provider wheel whose shim was left out: section 2.2, step 3) with `--no-deps`; writes `install_wheelhouse.py` (generated from the same module) into the
   wheelhouse; runs it into a scratch venv as the **install check**, exactly as the GPU run will;
   writes `MANIFEST.json` with every file's name, version, size, and sha256.

`install_wheelhouse.py` (both runs): `python -m venv --without-pip`, then pip from the
wheelhouse's own wheel (`python pip-*.whl/pip install`, a form pip supports), then `pip install
--no-index --find-links <wheelhouse> -r requirements.txt -c constraints.txt --no-compile`, each
extra (optional), `pip check`, an import of each module in a subprocess (`torch` reports its CUDA
version), and `pip freeze`; every command runs with `PYTHONPATH`, `PYTHONHOME`, and `PIP_*`
removed.

### 2.2 Run `P6.3-01`

The script notebook (`scripts/build_kaggle_notebook.py`, script mode) checks the inputs and the
GPUs (bf16 needed), installs the `arc26` wheel, and runs `python agent_spike.py --config
P6.3-01.yaml --commit <sha> --model-root <model output> --wheelhouse-root <wheels output>` with
the hard stop. The script then:

1. **Inputs.** Reads the model output's `MANIFEST.json` (it must say `ok` and name the pinned
   revision) and checks each file's size; reads the wheelhouse's (it must list vLLM 0.31.0 and
   hold `install_wheelhouse.py`); finds the training challenges and solutions under the
   competition folder by search.
2. **Install.** Runs the shipped installer with `/usr/bin/python3.11` into
   `/tmp/agent-spike/venv`; its report goes to `install.json`.
3. **Engine preparation** (`prepare_engine`, record in `engine_prep.json`). (a) If
   `flashinfer_jit_cache` imports without a `__version__`, a FlashInfer jit-cache provider was
   installed without its shim: `pip uninstall` the `flashinfer-jit-cache-*` providers and delete
   the empty namespace folders pip leaves inside the venv, then check that the name no longer
   imports. The wheelhouse's `flashinfer-jit-cache-sm89` is such a provider; FlashInfer reaches
   providers only through the shim `flashinfer-jit-cache`, which the wheelhouse lacks, so the
   provider was never usable, and it made `import flashinfer.comm` raise an `AttributeError`
   that vLLM does not catch (`P6.3-01` attempt 2, `docs/experiments/P6.3-01.md`). (b) An empty
   `sitecustomize.py` in scratch goes first on the engine's `PYTHONPATH`: the image's
   `/usr/lib/python3.11/sitecustomize.py` imports `log` from the image's packages, fails in the
   venv at its second statement, and only printed an error in every process. (c) The venv's
   python, in the engine's environment, imports the modules of `wheels.engine_imports` (the
   worker, the device communicators, the model, its layers, the attention backends, the
   sampler); the first failure stops the run with its error and traceback.
4. **Preflight.** From the venv: torch's CUDA version, every GPU's name, capability, and memory,
   a bf16 matmul on each, the NCCL version, which GPU pairs allow peer access (P2P), and which
   `libnccl`, `libcudart`, `libcublas`, and `libcudnn` the process loaded. Fewer than 4 GPUs
   stops the run.
5. **Boot.** `vllm serve <model> --tensor-parallel-size 4 --kv-cache-dtype fp8
   --enable-prefix-caching --attention-backend TRITON_ATTN --language-model-only
   --reasoning-parser qwen3 --max-model-len 65536 --max-num-seqs 64 --max-num-batched-tokens
   8192 --gpu-memory-utilization 0.90 --generation-config auto --seed 0`, no speculative
   decoding, in its own process group, with the image's `PYTHONPATH` and Triton's tool overrides
   removed. The script waits for `/health` and parses the log: the KV pool in tokens, the maximum
   concurrency at 65,536 tokens, weight loading, compile, and graph capture times, the attention
   backend, and NCCL's transports. A sampler then records `nvidia-smi` (memory, utilization,
   power) and the server's counters every 5 s.
6. **Sanity.** One short request ("What is 17 + 25?") at low effort; an error, an empty answer, or
   a degenerate output stops the run; a wrong answer is recorded.
7. **Throughput cells** (phase `throughput-8k`, then `throughput-32k` last). Each context starts
   with one unmeasured warmup request at its prompt length (32 tokens; its cold TTFT is kept), so
   kernels Triton tunes per shape on first use stay out of the first cell. Then, for each
   concurrency $c \in \{1, 8, 16, 32, 64\}$, $c$ streaming requests start at once, each with its
   own synthetic ARC-like prompt (digit-row grids, about $P$ tokens: 7,168 or 31,744), thinking
   on, `ignore_eos`, and $O = 2048$ generated tokens. With $t^{s}_r$, $t^{f}_r$, $t^{l}_r$ the
   send, first-token, and last-token times of request $r$ and $n_r$ its tokens:

   $$R_\text{agg} = \frac{\sum_r n_r}{\max_r t^{l}_r - \min_r t^{s}_r}, \qquad
   R_\text{steady} = \frac{\sum_r \sum_{k \in r} n_k \, \mathbb{1}[t_0 < t_k \le t_1]}{t_1 - t_0},
   \quad t_0 = \max_r t^{f}_r,\ t_1 = \min_r t^{l}_r,$$

   where $k$ runs over the chunks of $r$. $R_\text{steady}$ counts only the window in which every
   request is decoding: no prefill, no tail with fewer sequences. The mean context over the
   decode is $\bar{L} = \bar{P} + O/2$: 8,192 and 32,768. Also recorded: time to first token
   (TTFT, its median and 90th percentile), each request's decode rate, the server's preemptions
   and prefix-cache hits, and, per request, the longest run of one token (special tokens break
   runs, and only tokens before the first end-of-sequence id count, since with `ignore_eos` a
   finished model keeps generating), whose length 64 marks the output degenerate, the symptom of
   NaN logits. A wave that reaches its cell timeout is cut: its sockets are shut down, each
   request's own thread closes its connection, and the open requests are recorded as
   `cancelled`, which the kill criteria skip. Any other failure of a request (an HTTP status, a
   broken or refused connection, a stream that ends without `[DONE]`) is recorded as that
   request's error, never raised.
8. **Probe** (phase `probe`). A seeded sample (seed 0) of 10 ARC-AGI-2 training tasks with one test
   input and at most 6,000 characters of text; all at once, thinking on, low effort, up to 16,384
   tokens per round. Round 1 asks for `transform(grid)`; the last fenced block that defines it is
   run on every demonstration and test input in a fresh subprocess (`python -I`, 2 GiB address
   space, CPU limit, wall timeout, one BLAS thread). A program is **verified** when
   $\forall i: f(x_i) = y_i$ on the demonstrations, and **correct** when also $f(x^\ast) =
   y^\ast$ on the test input (the training solution). Round 2, for the unverified, sends the
   checker's report (each example right or wrong, cell counts, the first wrong grid beside the
   expected one, errors) and asks again.
9. **End.** The server is stopped (SIGTERM, then SIGKILL after 60 s), the head and tail of its
   log are copied, and `metrics.json`, `summary.md`, and `run.json` are written whatever happened.

### 2.3 Run `P6.3-02`

The same install, preflight, boot (with `--max-model-len 32768`), sampler, and sanity request,
then two phases with the merged harness (`task/P6`: P6.1 and P6.2). Revision 3 of the brief sets
the budgets from `P6.3-01`'s Borderline result: about 300 tokens per second effective at 32
requests, every attempt's context capped at 32,768 tokens, phase 1 before phase 2. Revision 4
adds the harness preflight and the harness rule of the gate, after revision 3's run failed every
attempt in its sandbox (below). Revision 5 keeps the reasoning across turns, sends a forcing turn
after every reply cut at its cap, and runs the forcing check before the cascade in place of the
preserve check, after revision 4's run read Bad with 156 of its 201 engine calls cut at the cap
while the model was still reasoning (`docs/experiments/P6.3-02.md`).

0. **Harness preflight** (stage `harness`, right after the inputs, before the install; seconds).
   With the agent phase's own REPL factory and verifier pool (the config's limits): a
   `SubprocessRepl` for DEV40's smallest task runs a cell that imports numpy and counts the
   helpers in `lib`'s index, both equal to the parent's; a second REPL, its Python-level policy
   off, reports the child's interpreter, import path, and where it loaded `arc26`, `numpy`, and
   `scipy` from (`arc26` and `numpy` must be the parent's files); the pool verifies the identity
   program on the task, and every input must come back a grid. Any failure stops the run (exit 1)
   with the full traceback printed and in `harness_preflight.json`, which also records the
   parent's interpreter, `sys.path`, user id, temporary folder, and the child's command line. The
   factory and the pool are reused by the agent phase. Why: run `P6.3-02` (12 Oct) learned only
   after 14 minutes of install, boot, and preserve check that no sandbox process could start:
   P6.1 passed the parent's site folder as the child's `PYTHONPATH`, ahead of the standard
   library, and Kaggle's image keeps the obsolete `pathlib` 1.0.1 backport there, which fails on
   import under Python 3.11. P6.1 now starts each child without `site` and with the standard
   library first (`docs/modules/agent_sandbox.md`).
1. **Forcing check** (phase `forcing`, revision 5; seconds to minutes). On DEV40's 3 smallest
   tasks at once, one P6.2 attempt each with the agent's own loop, engine, REPL, and verifier:
   one thinking round capped at 1,024 tokens at low effort, reasoning kept, so it is cut; then
   its forcing turn (thinking off, `agent.forcing_tokens`). Per task: whether turn 1 was cut,
   whether the forcing call had thinking off, whether its reply holds a complete python block
   defining `transform`; the growth $P_2 - P_1$ of the two calls' `prompt_tokens` against $R +
   T$, the counts (vLLM's `/tokenize`) of the cut reasoning and of turn 1's text and the forcing
   message, kept when within 64; and the render check on **the prompt the forcing call fed the
   model** (revision 6): the call's own response carries its prompt's token ids
   (`prompt_token_ids`, under the `return_token_ids` the engine always sends) and the templated
   text (`prompt_text`, under `return_prompt_text`, asked only in this check; vLLM 0.31's
   `ChatCompletionResponse`); the ids are decoded by `/detokenize` (a tokenizer call, no
   template), else the text is taken. It must hold the reasoning's first and last 200
   characters, the last followed by the template's `\n</think>\n\n`, and end with the empty think
   block `<think>\n\n</think>\n\n`; whether it starts with turn 1's prompt (`prefix_kept`, by
   token ids, the prefix cache's share) is reported. Revision 5 re-rendered the request through
   `/tokenize` instead, whose path dropped the assistant turn's reasoning that the chat call had
   rendered: all three renders failed while the real prompts had grown by the reasoning's 1,024
   tokens. **Pass:** at least 2 forced programs, at least one task with its reasoning kept and
   none dropped, no failed render (a response without its prompt, or one that cannot be decoded,
   leaves the render `unavailable`: reported, not judged; a failed `/tokenize` count judges the
   growth from the cut reply's generated count, $P_2 - P_1 \ge G - 64$). **Fail:**
   `forcing_check.json` and `agent_metrics.json` record the gate row Fault, the run stops with
   exit 1 before the agent's phases, and the problem names the reason.

   The **preserve check** (phase `preserve`) was revision 4's first phase (pass on 12 Oct); the
   script keeps it, `configs/P6.3-02.yaml` no longer runs it. On DEV40's 3 smallest tasks, through P6.2's
   `OpenAIEngine`: turn 1 is the agent's own opening (P6.2's system prompt and task message) with
   thinking on at low effort; turn 2 resends it with the reply and a follow-up, generating one
   token, in four variants: the reply without its reasoning (`off`, P6.2's default), with
   `Message.reasoning` sent as `reasoning_content` and `preserve_thinking` true (`on`), the same in
   the `reasoning` field, and reasoning sent with `preserve_thinking` false. With $R$ the
   reasoning's token count from vLLM's `/tokenize` and $P_v$ turn 2's `prompt_tokens`,
   preserved thinking works when $|P_\text{on} - P_\text{off} - R| \le 16$ and the last
   variant's growth is within 16 of 0; a reasoning of 32 tokens or fewer is inconclusive.
   `preserve_check.json`; a fail is recorded, not a kill, and so is a failure: a task whose
   check raises (a broken connection, an engine or `/tokenize` error) is inconclusive with its
   `error`, and the next task runs; then the agent's phases run whatever happened.
2. **Agent cascade** (phase `agent`). DEV40 through P6.2's loader (demonstrations and test inputs
   only), smallest first. Phase 1 runs P6.2's `run_tasks` on every task, one short attempt each
   (low effort, up to 3 thinking turns, 16,384 tokens per turn, 24,000 in all), 32 at a time,
   each in a fresh P6.1 `SubprocessRepl`, every program checked by P6.1's `PoolVerifier` (16
   workers, its guards); phase 2 runs it again on the tasks no verified program covers (xhigh, up
   to 4 thinking turns, 24,576 per turn, 49,152 in all), with their failed programs shown. Since
   revision 5 both phases keep each reply's reasoning in the conversation
   (`preserve_thinking`), and a reply cut at its cap gets one forcing turn (P6.2's
   `Budget.forcing_tokens`, 4,096; `docs/modules/agent_loop.md`, "A turn"). Every attempt's
   context is capped at 32,768 tokens (`agent.context_tokens`, P6.2's `Budget.context_tokens`):
   a reply's `max_tokens` is lowered so that an upper bound of the prompt plus the reply, plus a
   thinking reply's forcing reserve (6,021 tokens), fits; the first prompt is bounded by the
   server's count of it (vLLM's `/tokenize`, plus 64; one token per character if the count
   fails), later ones from the engine's counts. On DEV40 turn 1 gets 16,384 tokens in phase 1 on
   every task, and in phase 2 19,392 to 24,189 without earlier programs, about 16,200 to 21,000
   with two median failed programs shown, about 12,800 to 17,600 with two of the largest; then
   its forcing turn, then a short repair turn after the verifier's report if room is left
   (phase 1 about 3k; phase 2 little or none). Each phase stops at its budget (3,600 s each) or the run's deadline: attempts
   not started are skipped, running ones stop at their next engine call (the call in flight is
   cut at the deadline). P6.2's `build_submission` chooses two attempts per test
   input; after every attempt has ended the evaluation solutions are read and
   `arc26.evaluate.score_submission` scores them.
3. **Records.** Every engine call appends a row to `engine_calls.jsonl`: the attempt, where its
   reasoning arrived (`reasoning`, vLLM 0.31's field; `reasoning_content`; or inline in `content`
   before `</think>`), how many turns it sent back with reasoning and with which
   `preserve_thinking`, its thinking switch (`thinking`: false on a forcing turn), its tokens and
   finish reason; `metrics.json`'s `engine_calls` counts the finish reasons of all calls and of
   the forcing turns. Each phase's report and `agent_metrics.json` count forcing turns and forced
   programs (in all and per task). The measures of the design note's section
   6.2: the score, verified rate, precision on test (the share of distinct verified programs right
   on every test output), tokens per verified task, turns, guard rejections (programs with a fatal
   guard finding), sandbox failures (programs with a timeout, memory, or crash on some input),
   verifier workers replaced, and wall time per phase. Each call also records the engine's health
   (vLLM's token ids: a run of 64 identical non-special tokens before the first end-of-sequence
   id, token-0 counts, empty replies; failed calls).
4. **The gate** (ADR-020 as amended 9 Oct). $M$ counts NVARC's DEV40 misses solved by verified
   programs only: every test output's right attempt (P6.2's selection) comes from a verified
   program; misses scored by direct answers or unverified programs, or solved in part, are listed
   apart. The row: Good at $M \ge 7$ with a precision of at least 0.8; Bad at $M \le 4$ with a
   healthy engine (failed calls under 10%, no degenerate output) and a healthy harness (every
   phase that started attempts finished at least one, and under 10% of its attempts raised);
   Borderline at 5 or 6, or at 7 or more with a precision from 0.6 to 0.8; Fault otherwise
   (`agent_metrics.json`, key `gate`). A call the run's own deadline cancelled (a
   `CancelledError` row, in flight at a phase's end) is counted apart (`cancelled`), not as a
   failed call: revision 4's 5 failed calls were all such, and a phase that ends with 32 calls in
   flight would otherwise read an unhealthy engine. In turn a hung engine would end every call
   that way (P6.2's engine has no read timeout), so a phase whose deadline cancelled calls also
   needs at least `agent.min_tokens_per_s` (10) tokens per second of its wall time and an answered
   call within `agent.stall_s` (1,200 s) of its end; a stalled phase makes the engine unhealthy,
   so the row reads Fault, not Bad (`agent_metrics.json`, `engine.stalls`; a problem of the run). A harness fault is also a problem of the run (exit 1). A
   failed forcing check is a Fault before phase 1 (step 1).
5. **Faults made visible.** The script sends the `arc26` package's log records to standard
   output (warnings and errors; each exception kind's traceback once) and to `harness.log` (every
   record from INFO up, every traceback), whatever the process's root logger does. Each phase's
   report counts the attempts that raised, by kind (the exception's type and its message's last
   line), and `agent_metrics.json`'s `attempt_errors` keeps the first traceback of each kind.

### 2.4 Run `P6.3-04`: two other routes

One notebook, one engine boot (`max_model_len` 65,536), the same install, preflight, sanity
request, and forcing check as `P6.3-02` revision 6, then two arms in sequence
(`configs/P6.3-04.yaml`; brief `docs/prerun/P6.3-04.md`). Why: the organizers' 42.4% (xhigh)
and 22.8% (low) for Qwen3.8-27B are direct-answer scores (the model writes the grids, pass@2);
this harness had measured only program synthesis.

1. **Arm D, direct answers** (phase `direct`). Every DEV40 task twice through P6.2's loop with
   `DIRECT_SYSTEM_PROMPT` (no REPL, no `transform`; one ```answer block per test input) and
   `NoRepl` (a code block gets a note, never runs): the task, and the task through one fixed view
   (`arc26.agent.views`: a quarter turn and a color permutation fixing 0, drawn from the run's
   seed; NVARC's `forward_mod` operations). One thinking turn at low effort, capped at 16,384
   tokens in a 32k context; a cut turn gets one answer-forcing turn with thinking off
   (`Budget.forcing_kind` `answer`: "Reply with exactly one answer block per test input ...",
   cap 6,144). 32 attempts in flight, the two samples of a task side by side, smallest tasks
   first, 3,600 s. The view's answers are mapped back by its inverse; the two samples are the
   two attempts per test output (pass@2), scored by `arc26.evaluate`. `direct_metrics.json`: the
   score; $M_\text{direct}$, the NVARC misses a task score of 1 solves; per sample the test
   outputs answered and right and the tasks fully right, and their difference (the memorization
   control: DEV40 is public, and a model that recalls its outputs scores far higher on the
   original than on the view); agreement of the two samples and precision when they agree; tokens
   per task; solves per hour of 4x L4; forcing turns; the route reading.
2. **Arm L, one long generation** (phase `agent`, `agent.only_nvarc_misses`). P6.2's program
   harness on the 29 NVARC misses at once: reasoning kept, one thinking turn at xhigh capped at
   40,960 tokens (the most 29 sequences fit in the 1.42M-token KV pool), then its forcing turn,
   then stop (rounds 1); 6,600 s (the forcing turns still reached at 220 tokens per second; below that the reading guard reads Fault (rate)). `agent_metrics.json` as for `P6.3-02` (ADR-020's $M$ and
   precision; direct answers apart), plus `long_context` (the rate and the calls' context
   lengths) and the route reading.
3. **The readings** (pre-registered, `reading` section): a route enters P6.5's hybrid design when
   it solves at least 5 of the 29 misses (direct answers for D, verified programs for L) at no
   more than 54,000 generated tokens per task (what a 12 h run affords at about 300 tokens per
   second); Fault when its engine or harness health fails (errors, degenerate output, a
   stall); arm D counts only the misses its view solved when the original's output accuracy is
   0.10 or more above the view's; arm L reads Fault (rate) when more than 10% of its attempts were
   cut by the deadline before their forcing turn.

### 2.5 Run `P6.3-05`: the long shot

The regime of the organizers' 42.4% (direct answers at xhigh, about 172k tokens per task) on the
share of tasks a hybrid's scored run can afford it for (about 3 h of 12 after NVARC):
`configs/P6.3-05.yaml`, brief `docs/prerun/P6.3-05.md`. One boot at `max_model_len` 131,072
(Qwen3.8's native length is 262,144, rope default: no scaling flag), the forcing check, then the
`direct` phase with `task_ids` (the 15 DEV40 tasks NVARC is least sure of in `P2.6-01`: the top
selected grid's `score_kgmon` share per output from `reports/P2.8/P2.6-01.csv`, the minimum over
a task's outputs, ascending, ties by id), `with_view` false (one sample per task), and
`answers_per_test` 2 (`DIRECT_TWO_SYSTEM_PROMPT`: one or two answer blocks per test input, best
first; the forcing kind `answer2`). One thinking turn of up to 94,208 tokens at xhigh, all 15 in
flight, 10,800 s. `shot_metrics.json`: the score, the tasks solved (and by the second attempt
alone), NVARC's solves on the same tasks, tokens per task, forcing turns, the cut count, and the
reading: enters at 3 or more solves and 2 or more above NVARC; Fault on the engine's health;
Fault (rate) when more than 10% of the tasks were cut before their forcing turn. Since this run
an empty reply is reported (`empty`), not a degenerate output: `P6.3-04`'s Fault came from one
empty forced reply; token runs and token-0 runs stay degenerate. The smoke:
`.venv/bin/python scripts/agent_spike.py --config configs/P6.3-05.yaml --smoke --out-dir /tmp/s5`.

### 2.6 Run `P6.3-06`: the long shot's confirmation

`P6.3-05` solved 4 of its 15 against NVARC's 0; one run on 15 tasks is thin evidence for a
submission slot. `configs/P6.3-06.yaml` (brief `docs/prerun/P6.3-06.md`) repeats it unchanged,
same command and same kernel (`alyahya737/arc26-p6-3-shot`, its version 2), on the 15 DEV120 tasks
outside DEV40 NVARC is least sure of (`P6.3-05`'s rule on the other 80 tasks); only `run_id`,
`direct.task_ids`, and comments differ (`tests/agent/test_spike_confirm.py` checks it). The
`direct` phase now loads `task_ids` from the whole challenges file (`selected_tasks`, P6.2's
loader, demonstrations and test inputs only), labelled `DEV120, 15 selected` when an id lies
outside `agent.subset`; `agent.subset` stays DEV40, so the forcing check and the harness
preflight run on `P6.3-05`'s three DEV40 tasks. An id missing from the file stops the run at the
inputs stage, before the install and the boot. `P6.3-05`'s tasks and renderings are unchanged
(the test compares both paths). The smoke:
`.venv/bin/python scripts/agent_spike.py --config configs/P6.3-06.yaml --smoke --out-dir /tmp/s6`.

### 2.7 Run `P6.9-01`: the measurement wave of the re-targeted, tightened shot

Task P6.9 (`docs/tasks/P6.9.md`; `docs/research/beyond_h_options.md` on `task/ideas`, option 2c)
changes Submission 3's shot in three ways and measures the two that carry risk on one wave of 15
DEV120 tasks: `configs/P6.9-01.yaml`, brief `docs/prerun/P6.9-01.md`, the same kernel
(`alyahya737/arc26-p6-3-shot`, its version 3) and code dataset as `P6.3-05` and `P6.3-06`. It is
`P6.3-06`'s config with three new optional `direct` keys, the shot's tasks, and a shorter budget:

1. **The tasks** (`direct.task_ids`, in bucket order): with NVARC's task confidence $c$ of
   `P2.6-01` (`P6.3-05`'s rule on `reports/P2.8/P2.6-01.csv`), the 4 tasks of $0 < c < 0.5$ no
   shot ran before; 6 of the 23 in $0.5 \le c < 0.7$, the task brief's uniform draw
   (`random.Random(0).sample` of the sorted ids; NVARC 2.0 of 6, both by attempt 1, against the
   bucket's 7.67 of 23); and 5 of the 27 tasks NVARC solved at $c \ge 0.85$, the harness check.
   21 test outputs; NVARC scores 7.0 of 15 on them.
2. **In-request forcing** (`direct.forcing_in_request`, `direct.thinking_end_note`, and
   `server.extra_args`' `--reasoning-config`): each thinking request carries
   `thinking_token_budget` $= C = 94{,}208$ and `max_tokens` $= C + F$ ($F = 6{,}144$). vLLM
   v0.31.0 counts the reasoning from the last `<think>` (the generation prompt's), and once it
   reaches $C$ its sampler sets the logit of the next token of `reasoning_end_str` (the note, then
   `</think>`) to $10^9$ until the string is out; the model then writes its answer in the same
   request, with the conversation's KV still in place. A reply that still ends at its cap without
   a readable answer block (a server that ignored the budget) gets a forcing turn with its cut
   reasoning left out, so nothing long is prefilled again; `token_budget` is $C + 2F$. Before
   the shot one request (the sanity question at low effort, budget 8 tokens,
   `arc26.agent.shot.budget_probe`) checks that the server forces the note
   (`budget_check.json`). If it errors (vLLM's HTTP 400 when its reasoning config is off,
   `vllm/v1/engine/input_processor.py`) or its reasoning does not end with the note, the
   in-request forcing is off for the whole run: `P6.3-05`'s request and its separate forcing
   turn, `in_request` false in `budget_check.json` and a problem line in `run.json`. A single
   request refused for the field (HTTP 400) is sent once more without it and that attempt goes on
   as `P6.3-05`'s (`budget_refused` in its record). The client-side cut (`direct.turn_cut_s`,
   6,300 s $= (C + F)/15.9$ tokens per second): a thinking request not answered by then is dropped
   (`turn_cuts`) and a forcing turn on the opening alone answers, so every attempt ends by about
   6,760 s, inside the 7,200 s budget, even when the server ignores the budget.
3. **The lenient parser** (`direct.lenient_answers`): a block the strict parser refuses is read
   again with rows as digit strings, cells as one-digit strings, and digit-string rows whose
   brackets do not balance (a row `01 02` is four cells); per test output the first attempt is
   the first answer of the shape the demonstrations imply (every output the size of its input,
   or one fixed size), the model's own order with drafts of other sizes skipped, the second the
   next answer that differs. `shot_metrics.json` adds `strict` (the same records scored with
   the strict parser's answers and `P6.3-05`'s choice), `forced_answers` (the outputs of the
   tasks whose thinking was cut, by the budget or by the cap and a forcing turn, and how many
   got an answer), and per task `budget_forced`, `lenient_answers`, `right` (each attempt right
   or not per output), and `implied`.

The reading is not the script's `reading` field (`P6.3-05`'s rule, meaningless against 6 NVARC
solves by design) but `scripts/merge_replay.py` on the pulled logs: it replays Submission 4's
merge (`arc26.agent.shot.merge_output`) with NVARC's `P2.6-01` attempts and reads the brief's rows
(section 6 below). The smoke:
`.venv/bin/python scripts/agent_spike.py --config configs/P6.9-01.yaml --smoke --out-dir /tmp/s9`
(3.0 of 3: `smoke-mirror` answered inside its cut request, `smoke-transpose` by its second
attempt, `smoke-recolor` read only by the lenient parser; `strict` 2.0).

## 3. Why it is built this way

| Choice | Alternatives considered | Why they lost |
|---|---|---|
| vLLM 0.31.0 as the engine | SGLang 0.5.21; llama.cpp; transformers `generate` | SGLang is the fallback for the official FP8 build (its Triton W8A8 path on sm_89); its Qwen3.8 cookbook was measured on H200 and consumer cards, not Ada, and it pins other transformers and FlashInfer versions, so it needs its own wheelhouse: one engine first. llama.cpp has no useful batching (13 tokens per second for one request on one L4). `generate` has no paged KV or continuous batching (models note, 4 and 5). |
| `RedHatAI/Qwen3.8-27B-INT4` at a pinned commit | official FP8 (30.89 GB); AWQ builds (21.0 to 21.6 GB); AutoRound (19.47 GB) | Only a build of at most about 20 GB fits one notebook output (the measured `/kaggle/working` is 20,957,446,144 bytes); FP8 would need two datasets or a mirror and runs weight-only on L4 anyway. RedHat's build has published recovery (98.5% to 101%, AIME 2025 98.7%) and FP8 KV scales; AutoRound has no evaluations (models note, 2.2). |
| Our own CPU download notebooks | public Kaggle copies; the Hub reference route | Copies are known by title only, with no revision or hashes; the reference route for 31 GB repositories rests on a search snippet (models note, 6). The notebook pins the commit and hashes every file. |
| `snapshot_download`, then our own hashing | `hf_transfer`; plain HTTP with resume | `snapshot_download` resumes and uses Xet when the image has it; hashing every file ourselves makes the check independent of the downloader. |
| A wheelhouse built on the pinned image | `pip install` in the GPU session (offline); the `kaggle-vllm` package (vLLM 0.18, cu128); installing into the image's packages | GPU sessions have no internet. vLLM 0.18 predates the Qwen3.8 work the models note relies on and runs on torch 2.10. The image's packages hold torch 2.8 on `PYTHONPATH` (the attached install notebook), which would shadow torch 2.13. |
| Constraints from a metadata-only resolution | letting pip resolve on Kaggle | uv resolved all 199 packages for CPython 3.11 manylinux_2_35 with binary wheels only, from PyPI metadata, before anything was pushed, so a missing cp311 wheel would have shown here; pip then reproduces exactly that set. |
| The pinned reference image (Python 3.11.13) | Kaggle's latest image | The latest image moved to Python 3.12 and its source to 3.13 (first-runs note), so a wheelhouse for it could stop matching; the pinned image is the one every script notebook runs, and its CPU sessions run the same image, so the install check proves the GPU install. |
| A venv made with `--without-pip`, pip from its own wheel | `ensurepip`; `pip install --target` with `PYTHONPATH`; uv | Debian-style Pythons may lack `ensurepip`. `--target` mixes with the image's packages and `.pth` files. uv is faster but one more tool; the install check measures pip's time first. |
| The existing script mode for the GPU notebook | a new GPU notebook builder | Script mode already checks the inputs and their sha256, the GPUs (bf16), the hard stop, and the logs, and it attaches notebook outputs by pinned version. Its cost is two unused attachments (the reference model and install notebook). |
| Tensor parallelism 4, one replica | two TP2 replicas; four TP1 replicas | The KV heads (4) allow TP 1, 2, or 4; TP1 leaves under 2 GB for KV; two TP2 replicas duplicate the weights and hold fewer sequences (models note, 5.2: 398 against 528 tokens per second at 32k, estimate). |
| FP8 KV cache | BF16 KV | Half the bytes per token, so about twice the sequences at a given context, with the checkpoint's static scales. |
| Prefix caching on | off | The agent resends each conversation every turn; the NaN bug #55766 needs speculative decoding, which is off. |
| No speculative decoding | MTP (1 to 3 tokens); DFlash2 or DSpark drafters | MTP collapses the batch to about 3 sequences on 24 GB cards (#55533); drafters plus prefix caching corrupt outputs (#60174, #58894) (models note, 4.3). |
| Triton attention | FlashInfer; FlashAttention 2 | FlashAttention 2 takes no FP8 KV cache on Ada. FlashInfer may JIT-compile kernels, which needs a CUDA 13 nvcc the image lacks; its sm_89 kernel cache, downloaded for a later comparison, also needs the shim the wheelhouse lacks, so the engine preparation removes it. |
| Remove the orphan jit-cache provider in the venv (`drop_orphan_jit_cache`) | add the shim `flashinfer-jit-cache` (4.5 KB on FlashInfer's cu130 index) to the wheelhouse; uninstall FlashInfer; `FLASHINFER_DISABLE_VERSION_CHECK`; `VLLM_ALLREDUCE_USE_FLASHINFER=0` | The shim needs a new wheels notebook run and dataset version, the next wheelhouse's change. vLLM imports FlashInfer in every worker. The version-check switch acts after the failing attribute read. vLLM imports its FlashInfer all-reduce module unconditionally, whatever the switch (`cuda_communicator.py`). With the provider gone, FlashInfer runs none of its kernels here: Triton attention, Triton DeltaNet prefill on sm_89 (FlashInfer only on sm_90, sm_100, sm_120), the PyTorch sampler, and no FlashInfer all-reduce size for capability 8.9. |
| An import check of the workers' modules before the boot | the install's `import torch, vllm` probes only | Attempt 2's probes passed and the workers then failed on an import made only inside them, after the boot had started; the check takes under a minute (estimate) and names the error. |
| The PyTorch sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`) | FlashInfer's sampler (vLLM 0.31's default) | Same JIT risk; the sampler's cost is small next to a 27B decode step. |
| `max_num_seqs` 64 | vLLM's default | The largest concurrency measured; a participant's default exceeded the hybrid model's state slots at graph capture (models note, 4.3). |
| `ignore_eos` with 2,048 tokens per request | natural stops | Natural lengths vary by request, so the batch shrinks while it is measured; fixed lengths make every cell the same workload. |
| One wave per cell (closed loop) | Poisson arrivals at a request rate (`vllm bench serve`) | The agent keeps a fixed number of conversations in flight, a closed loop; a wave of $c$ requests is that operating point. |
| The steady window beside the wave's rate | the wave's rate alone | The wave's rate mixes prefill and the tail; the steady rate is the decode rate the budget plan needs (design note, 5.5). |
| Synthetic ARC-like prompts | real tasks | The cost of a decode step depends on the context length, not on what the grids show; synthetic prompts hit each target length, and each request's prompt differs from the start, so no prefix cache hit inflates a cell. |
| 8k cells first, then the probe, the 32k cells last | the 32k cells first | The 32k cells cost the most (up to 2M prompt tokens at $c = 64$); a short budget then cuts them, not the probe. |
| A standard-library streaming client in threads | httpx, aiohttp, or openai from the image | Their versions on the image vary with the attached install notebook; `http.client` needs nothing installed, and `return_token_ids` gives exact token counts per chunk. |
| A cut shuts the sockets down; each reading thread closes its own connection; every failure is the request's `error` | closing every connection from the cancelling thread; catching only `OSError`, `HTTPException`, and `ValueError` | `http.client` objects are not thread-safe: in `P6.3-01`'s attempt 3 the cut of the 32k, 64-request cell closed connections under threads inside `readline`, `http.client` raised `AttributeError` on its own `fp`, and the uncaught error ended the run. `http.client` also ends a chunked stream cut mid-way as a clean end of file, so a stream without `[DONE]` is an error. |
| Stage 2's context cap at 32,768 (`max_model_len` and `agent.context_tokens`) | 65,536, the checkpoint's length | `P6.3-01`'s Borderline row: contexts capped where the measured rate clears 300 at 32 requests (377 steady at 32k). The KV pool (1,593,344 tokens) holds 48 sequences of 32,768 and 24 of 65,536, so 32 attempts at the cap are never preempted. |
| The probe on training tasks | DEV40 tasks | DEV40 is stage 2's set; training tasks have solutions for the test check and keep evaluation data out of a probe. They are easier than the hidden test (many are ARC-AGI-1 tasks), which the brief states. |
| A minimal subprocess checker | P6.1's sandbox and verifier | P6.1 is being built at the same time; the checker re-runs every program in a fresh, limited process, enough for a probe; stage 2 uses P6.1's verifier. |
| Kill criteria checked by the script | Kaggle's Timeout alone | The CLI cannot cancel a run; the script stops on a dead server, a failed boot, or a cell with more than half its requests failed or degenerate, and writes its logs first. |
| `--resume-from` a previous output | none | Each cell and probe task is independent, so a second session can skip the finished ones; install and boot are repeated anyway. |
| Stage 2 in the same script | a separate script | The script notebook ships one script; stage 2 reuses the install, boot, sampler, and kill logic. |
| P6.2's `run_tasks` per phase, with a deadline | the spike's own scheduler; one `run_tasks` call with every phase's attempts | `run_tasks` is the harness the submission will run (concurrency, crash-safe record log, transcripts, deadline handling); a call per phase lets phase 2 take only the unverified tasks and show their failed programs. |
| P6.2's `build_submission` and `arc26.evaluate` for the score | the spike's own vote | The score must be the one the submission would get, with the selection it would use, and the project's metric code (CLAUDE.md). |
| The smoke on P6.1's real REPL and verifier with P6.2's `ScriptedEngine` | stand-ins for the REPL and the verifier; the fake HTTP server for the agent too | Real sandbox processes are what the GPU run uses, so the smoke is the integration test of P6.1 and P6.2; the scripted engine replays each outcome (a right program, a fix after the verifier's report, a guard-rejected copy of the outputs, a direct answer) without a model. |
| The preserve check with one-token turn-2 requests and `/tokenize` | comparing whole agent runs with and without preserved thinking | One-token requests isolate the prompt's growth in seconds; `/tokenize` gives the reasoning's own count on the server's tokenizer. Four variants separate the transport (`reasoning_content` or `reasoning`) from the template's switch. |
| A recording subclass of P6.2's client | a streaming client; vLLM's server logs | The agent's own client is what the submission runs; overriding its one post method sees the raw reply, so every call's reasoning location is recorded without a second client. |
| A forcing turn with thinking off after every reply cut at its cap (revision 5), the cut reasoning kept | a longer turn cap; `continue_final_message` budget forcing (resend the cut reply as an open assistant message with `</think>` appended, so the model continues in its answer); preserved reasoning without forcing | **Longer cap:** revision 4 cut 78% of its calls at 16,384 and 24,576 tokens, and the 32k context (`P6.3-01`'s Borderline row, unchanged) leaves at most about 25k for one turn after a 2.5k to 7.3k prompt, so no cap that fits ends the cut; it only moves it. **Budget forcing** (Muennighoff et al. 2025, s1, "Wait" and end-of-thinking forcing) continues the same message, so the reasoning is kept without `preserve_thinking`, but it needs vLLM's `continue_final_message` with `add_generation_prompt` false, a server path this harness has never measured, and a hand-written `</think>` in the assistant text, which vLLM's qwen3 reasoning parser must then split from the new output; the template's own switch (`enable_thinking: false`) prefills the same empty think block through the path the sanity request and the preserve check already measured. **Preserve without forcing:** a cut turn's next turn is a thinking turn again, which at 32k has room for at most a few thousand more tokens of reasoning and in revision 4 mostly reasoned again (40 of 201 calls stopped by themselves); the forcing message asks for the program now. |
| The next prompt bounded from the engine's counts (prompt plus generated tokens, plus 16 for the think tags) | one token per character of the kept reasoning (revision 4) | With reasoning kept, a 16k-token reasoning of about 60k characters would have counted 60k against a 32k context, and every attempt would have stopped with `budget` after turn 1, before any forcing turn. The server's counts are exact; the preserve check measured the template's tags at 4 tokens. |
| A thinking turn reserves its forcing turn in the attempt's tokens and the context (6,021 tokens at a 4,096 cap) | reserve nothing and cap the forcing turn by what is left | Without a reserve a turn cut at the context's edge leaves no room for its forcing turn, which is exactly the case the forcing turn is for. |
| On the forcing call, the effort's instruction written at the head of the system message (`QWEN38_EFFORT_NOTES`), no effort field | send the effort as on other calls; send none | The template reads the effort only while thinking is on, so it cannot raise either way, but with thinking off it drops its instruction from the system message: the forcing prompt's first tokens would differ from the conversation the prefix cache holds, and every forcing turn (about 120 expected, up to 30k tokens each) would be prefilled again, about 3M tokens at the measured 1,500 tokens per second of prefill. The template's own text renders the same system block (checked with the template locally, 12 Oct, for low, xhigh, and medium; the forcing check reports `prefix_kept`). |
| The forcing check before the cascade, a Fault on failure | the preserve check (revision 4); no check | The preserve check passed on 12 Oct and measured one-token turns outside the loop; the forcing check runs the agent's own path (loop, engine, REPL, verifier) on a cut turn and its forcing turn in about 2 minutes, and a broken thinking switch, template, or parser fails it before 2 hours of quota measure a broken turn structure. |
| The forcing message's wording ("Stop reasoning now. Reply with exactly one python block that defines transform(grid) ... and nothing else: no exploration, no prints, no tests, no prose"; revision 6, after one of revision 5's three forced replies explored with prints) | resend the task; ask for a direct answer | The reasoning in context already holds the analysis; one block is what the verifier judges; a direct answer would bypass verification, which the gate's $M$ does not count. |
| Two samples per task, the task and one fixed view, for arm D | two samples of the task itself; more views | The view is the memorization control (a public task recalled scores far better as published) and still a pass@2 attempt; one view keeps the arm at 80 attempts in an hour. |
| A quarter turn and a color permutation fixing 0 (`arc26.agent.views`) | NVARC's `arc_loader` itself; a permutation of all ten colors | `arc_loader` imports transformers, which the smoke's interpreter lacks; the module copies its two operations. Color 0 is the background in most tasks: recoloring it makes a different puzzle. |
| Arm L's 29 attempts in one wave at a 40,960-token turn | 20 in flight at a full 64k (the pool's count) | Two waves of about 41k-token turns take about 3,800 s at 350 tokens per second but about 6,400 at 220, the second wave cut before its forcing turns; one wave fits the pool and ends in lockstep. |
| Direct answers as JSON answer blocks (P6.2's `parse_answer`) | rows of digits | The loop already parses and validates them; a forcing turn asks for the same format. |
| `P6.9-01`: the answer forced inside the thinking request through `thinking_token_budget` | `P6.3-05`'s separate forcing turn; `continue_final_message` budget forcing (s1, Muennighoff et al. 2025); a client-side stop at $C$ tokens and a resend | The separate turn re-prefills the whole conversation: 816 to 1,061 s of a shot, 14% to 23% (`P6.3-05`, `P6.3-06`, `P6.5-03`), prefix-cache hits only 1.2% to 2.8%. `continue_final_message` resends the cut reply and prefills it again too, through a server path never measured here. A client-side stop needs streaming and a second request. vLLM 0.31.0 forces the end in the running request (`vllm/v1/worker/gpu/sample/thinking_budget.py`, Model Runner V2, which `P6.3-05`'s boot log shows), so no token is prefilled twice. |
| `P6.9-01`: a transition phrase before `</think>` (`thinking_end_note`) | vLLM's default, `</think>` alone | With the bare tag the reply cannot tell a forced end from a natural one (vLLM reports neither), so the forcing row could not be read; the phrase also tells the model what comes next, as vLLM's own example does ("I have to give the solution based on the reasoning directly now."). V2's kernel detects the natural end by the parser's `</think>` alone, so a phrase does not break natural ends; the V1 runner would (it looks for the whole forced string), which is why the boot log's runner line matters (failure modes). |
| `P6.9-01`: the budget check before the shot; on a failure the in-request forcing off for the whole run | no check; a Fault that stops the run; keep the budget field and rely on each attempt's fallback | Without it a vLLM without the reasoning config answers every thinking request with HTTP 400, which the loop alone would end as `engine_error`, and a server that ignores the field shows only after the wave; a stop would waste the boot for a wave `P6.3-05`'s request still reads. The per-request retry without the field covers a refusal the check did not see. |
| `P6.9-01`: the client-side cut at 6,300 s | none (the 7,200 s budget alone); a cut by tokens | A server that ignores the budget generates $C + F$ before its forcing turn, about 6,009 s at 16.7 tokens per second, and a forcing turn with the 100k context then overruns 7,200 s; the cut and the forcing turn without the cut reasoning keep every attempt answered by about 6,760 s. A cut by tokens needs a streaming client the loop does not have. |
| `P6.9-01`: the lenient parser and the implied-shape choice | a stricter prompt; P6.1's Python tool | The prompt already asks for JSON rows and to check every row; 9 of `P6.3-06`'s 33 blocks were still invalid. The tool (a third party solved two of those three tasks with it) costs 1.5 to 2 agent-days and tool turns cost tokens (option 4b): after 2 Nov. On every earlier shot transcript the lenient rules read all 91 strict-valid blocks as before and recovered 9 of 24 refused ones; ragged and oversized grids stay refused. |
| `P6.9-01`: the middle bucket's 6 by the task brief's uniform seed-0 draw, the loss of a replaced attempt 2 from P2.8's rows of two runs | a sample stratified by NVARC's score (fix round 1 dropped it: chosen after seeing the draws); all 23 | The brief pre-registered 6 of 23; the draw holds NVARC's share (2.0 of 6 against 7.67 of 23) but no output right by attempt 2 alone, so the wave cannot see that loss. P2.8's rows can: 3 of 23 tasks with such an output in `P2.6-01` (2.0 task units) and 5 of 23 in `P1.1-01` (3.5), pooled $\ell = 5.5/46 = 0.12$ per task, which the reading subtracts (`scripts/merge_replay.py`). All 23 would cost two more waves. |
| `P6.9-01`: the shot's budget 7,200 s (Timeout 13,100 s) | `P6.3-05`'s 10,800 s | The answer follows inside the request, so the wave ends at about 5,000 to 5,900 s at the measured 16.7 to 19.0 tokens per second per stream; 7,200 s reaches every answer down to 13.9. About 6 quota hours are left this week (4x L4 bills at 2x), and 10,800 s would put the worst case at about 9.3. |
| The shot's `task_ids` loaded from the whole challenges file, `agent.subset` kept at DEV40 (`P6.3-06`) | `agent.subset: DEV120`; the subset's tasks filtered by id (`P6.3-05`) | The filter cannot reach P6.3-06's tasks (`KeyError '13e47133'`). DEV120 as the subset would move the forcing check and the harness preflight to DEV120's smallest tasks, whose cut at 1,024 low-effort tokens was never measured (the DEV40 three were cut in revision 4), so an uncut turn could read a false Fault after the boot. |

## 4. How to run it

### 4.1 The download notebooks (the human pushes; no brief)

```bash
cd ~/wt-P6.3
.venv/bin/python scripts/build_download_notebooks.py --config configs/P6.3-download.yaml \
    --out dist/P6.3-download                      # from a clean commit; prints the folders
.venv/bin/python scripts/kaggle_ops.py push dist/P6.3-download/arc26-dl-qwen38-int4 --timeout 7200
.venv/bin/python scripts/kaggle_ops.py push dist/P6.3-download/arc26-dl-agent-wheels --timeout 3600
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-dl-qwen38-int4
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-dl-agent-wheels
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-dl-qwen38-int4 --task-id P6.3-dl
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-dl-agent-wheels --task-id P6.3-dl
```

Both are CPU sessions (no GPU quota). Expected: the model notebook takes 10 to 40 minutes
(19.47 GB at 10 to 100 MB/s, then about 2 minutes of hashing) and leaves 19.47 GB of output; the
wheels notebook 10 to 25 minutes and about 4 GB (estimates). `run_logs/status.json` holds `ok`;
`run_logs/install_check.json` holds the install's steps and imports. A notebook whose `ok` is
false is pushed again after the fix; the GPU runs then attach version 2 (a config change, so the
briefs' `Code-commit` changes).

### 4.2 Run `P6.3-01` (after its brief is approved)

```bash
cd ~/wt-P6.3
.venv/bin/python scripts/build_submission.py --tag P6.3-01 --config configs/P6.3-01-script.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-create dist/P6.3-01/arc26-code   # first build
# wait about 5 minutes for Kaggle to process the dataset
.venv/bin/python scripts/kaggle_ops.py push dist/P6.3-01 --run-id P6.3-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-p6-3-spike
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p6-3-spike --task-id P6.3-01
```

Attach nothing by hand: the metadata attaches the competition, the code dataset, and both
download outputs at version 1. At most two batch GPU sessions per account, queued ones included:
push when a slot is free. Expected: install 3 to 5 minutes, boot 5 to 15, the 8k cells 10 to 15,
the probe up to 25, the 32k cells up to 40; about 1.5 h in all, 2 h at most (the Timeout). GPU
memory: about 20.7 GB per L4 by design (`gpu_memory_utilization` 0.90); disk: about 12 GB in
`/tmp` (the venv and caches), kilobytes to a few MB in `/kaggle/working/run_logs`.

### 4.3 On a plain GPU machine

Four GPUs of compute capability 8.0 or higher, a venv with vLLM 0.31.0 (or the wheelhouse), and
a model folder laid out as the download notebook leaves it (`MANIFEST.json` beside
`qwen38-27b-int4/`):

```bash
python scripts/agent_spike.py --config configs/P6.3-01.yaml --commit "$(git rev-parse HEAD)" \
    --model-root /data/qwen-output --venv-python /opt/vllm-venv/bin/python \
    --input-root /data/arc --out-dir runs/P6.3-01-local --scratch-dir /tmp/agent-spike
```

`--wheelhouse-root` instead of `--venv-python` installs from a wheelhouse folder. `--resume-from
<previous out_dir>` skips finished cells and probe tasks. The CPU smoke path:

```bash
.venv/bin/python scripts/agent_spike.py --config configs/P6.3-01.yaml --smoke --out-dir /tmp/spike
.venv/bin/python scripts/agent_spike.py --config configs/P6.3-02.yaml --smoke --out-dir /tmp/spike2
```

### 4.4 Run `P6.3-02` (after its brief is approved)

From `~/wt-P6.3b` (branch `task/P6.3b`; `task/P6.3` stays frozen until `P6.3-01` is pushed).
First the CPU check (no brief, no GPU quota): `configs/P6.3-02-cpu-script.yaml` builds notebook
`alyahya737/arc26-p6-3-agent-cpu`, which runs the same three dataset files as the GPU build
(`--smoke`: the harness preflight on DEV40's smallest task, then the agent phase with P6.2's
`ScriptedEngine` on the fixtures and DEV4 through P6.1's real REPL and verifier) in the pinned
image with the install notebook. Since revision 5 the smoke also runs the forcing check against
the fake server (a turn cut at its cap, its forcing turn with thinking off, the program run in a
real REPL and verified, the fake template's rendering checked) and scripts a cut reply and its
forcing turn into the agent phase (`smoke-mirror` and DEV4's four tasks). The CPU notebook
mounts no Qwen3.8 files, so the real template's rendering of the forcing prompt is checked by the
GPU run's forcing check, before its agent phases. Both builds' `arc26-code/` folders are
byte-identical, so one dataset version serves both:

```bash
.venv/bin/python scripts/build_submission.py --tag P6.3-02 --config configs/P6.3-02-script.yaml
.venv/bin/python scripts/build_submission.py --tag P6.3-02-cpu --config configs/P6.3-02-cpu-script.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P6.3-02/arc26-code -m "P6.3-02 <sha>"
# (dataset-create instead for the first build); wait until it is ready, about 5 minutes
.venv/bin/python scripts/kaggle_ops.py push dist/P6.3-02-cpu --timeout 1800
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p6-3-agent-cpu --task-id P6.3-02-cpu
# only when the CPU check passed:
.venv/bin/python scripts/kaggle_ops.py push dist/P6.3-02 --run-id P6.3-02
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p6-3-agent --task-id P6.3-02
```

The CPU check passes when the script exits 0 and `harness_preflight.json` has `ok` true (the
child's `arc26` and `numpy` are the parent's files, its path the standard library first);
`forcing_check.json` has verdict `pass` with 3 forced programs; both agent phases have `raised`
0; `agent_metrics.json` has the two right fixture programs verified (`smoke-mirror`,
`smoke-transpose`), forcing turns in phase 1 (1 for the fixtures, 1 more per DEV4 task) and in
phase 2 (1 per DEV4 task), DEV4's four tasks attempted twice each, no sandbox failure, and no
worker replaced; and `harness.log` holds no traceback.

Expected (revision 4's run measured the shared stages): the engine install about 2 minutes, its
preparation under 1, boot about 5, the forcing check about 1 to 3, phase 1 about 35 to 40
(revision 4: 36), phase 2 about 35 to 60; about 1.5 to 2.2 h, 3 h at most (the `Timeout`). The CPU smoke, the integration test of
P6.1 and P6.2, runs on the fixtures; with `--input-root <a folder holding the competition's files>`
it adds DEV4's four tasks:

```bash
.venv/bin/python scripts/agent_spike.py --config configs/P6.3-02.yaml --smoke --out-dir /tmp/s2
```

### 4.5 Run `P6.3-04` (after its brief is approved)

```bash
.venv/bin/python scripts/build_submission.py --tag P6.3-04 --config configs/P6.3-04-script.yaml
.venv/bin/python scripts/build_submission.py --tag P6.3-04-cpu --config configs/P6.3-04-cpu-script.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-create dist/P6.3-04/arc26-code
# the human checks the dataset is ready (kaggle datasets status alyahya737/arc26-code-p6-route)
.venv/bin/python scripts/kaggle_ops.py push dist/P6.3-04-cpu --timeout 1800
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p6-3-route-cpu --task-id P6.3-04-cpu
.venv/bin/python scripts/kaggle_ops.py push dist/P6.3-04 --run-id P6.3-04
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p6-3-route --task-id P6.3-04
```

The smoke: `.venv/bin/python scripts/agent_spike.py --config configs/P6.3-04.yaml --smoke
--out-dir /tmp/s4`.

### 4.6 Runs `P6.3-05` and `P6.3-06` (after the brief is approved)

`P6.3-06` is `P6.3-05`'s sequence with its tags; the brief's launch block is the one to follow.

```bash
.venv/bin/python scripts/build_submission.py --tag P6.3-06 --config configs/P6.3-06-script.yaml
.venv/bin/python scripts/build_submission.py --tag P6.3-06-cpu --config configs/P6.3-06-cpu-script.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P6.3-06/arc26-code -m "P6.3-06 <commit>"
# the human checks the dataset is ready (kaggle datasets status alyahya737/arc26-code-p6-route)
.venv/bin/python scripts/kaggle_ops.py push dist/P6.3-06-cpu --timeout 1800
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p6-3-shot-cpu --task-id P6.3-06-cpu
.venv/bin/python scripts/kaggle_ops.py push dist/P6.3-06 --run-id P6.3-06
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p6-3-shot --task-id P6.3-06
```

### 4.7 Run `P6.9-01` (after the brief is approved)

The brief's launch block is the one to follow; the orchestrator runs `kaggle_ops.py`.

```bash
.venv/bin/python scripts/build_submission.py --tag P6.9-01 --config configs/P6.9-01-script.yaml
.venv/bin/python scripts/build_submission.py --tag P6.9-01-cpu --config configs/P6.9-01-cpu-script.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P6.9-01/arc26-code -m "P6.9-01 <commit>"
# the human checks the dataset is ready (kaggle datasets status alyahya737/arc26-code-p6-route)
.venv/bin/python scripts/kaggle_ops.py push dist/P6.9-01-cpu --timeout 1800
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p6-3-shot-cpu --task-id P6.9-01-cpu
.venv/bin/python scripts/kaggle_ops.py push dist/P6.9-01 --run-id P6.9-01
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p6-3-shot --task-id P6.9-01
.venv/bin/python scripts/merge_replay.py --run-logs runs/P6.9-01/run_logs \
    --nvarc-submission ~/wt-P2.6/runs/P2.6-01/submission.json --out reports/P6.9-01
```

Expected: install about 106 to 126 s, preparation 33 to 40 s, boot 300 to 320 s, the forcing check
about 62 s, the budget check about 2 s, the shot about 5,000 to 5,900 s: about 1.6 to 1.8 h of 4x
L4 from the first cell (brief, section 6). Disk: the venv and caches in `/tmp`, about 12 GB.

## 5. Outputs

| File (in `/kaggle/working/run_logs/`) | Contents |
|---|---|
| `run.json` | run id, commit, exit code, wall time, problems, phases, stage times |
| `config.yaml` | the resolved config |
| `metrics.json` | boot facts, preflight, sanity, every cell's summary, the probe's totals (or the agent's), peak memory per GPU, log counts |
| `summary.md` | the verdict, the engine, the throughput table, the probe |
| `timing.jsonl` | one `{puzzle_id, op, seconds, tokens}` per engine request and for the boot (CLAUDE.md's timing rule) |
| `throughput.jsonl` | per request: cell, prompt and completion tokens, TTFT, end-to-end seconds, decode rate, finish reason, longest token run, degenerate |
| `throughput_cells.json` | per cell: wall, aggregate and steady rates, TTFT percentiles, mean context, errors, degenerate, the server's counters (preemptions, prefix-cache queries and hits, maximum running and waiting requests, KV usage) |
| `probe.jsonl` | per task: each round's tokens, timing, program, check, verdict |
| `boot.json` | parsed boot log: KV pool, maximum concurrency, load, compile, graph-capture times, backend, NCCL version and transports |
| `install.json`, `install.log` | the installer's steps, imports (torch's CUDA), and freeze |
| `engine_prep.json`, `engine_prep.log` | the jit-cache state before and after, what was removed, the import check (seconds per module, or the failing module, its error and traceback), the engine's `PYTHONPATH` |
| `preflight.json` | GPUs, capability, NCCL, P2P matrix, loaded CUDA libraries |
| `server_metrics.jsonl`, `gpu_samples.jsonl` | the sampler's series every 5 s |
| `vllm_server.log` | the server log's first 400 and last 2,000 lines |
| `agent_attempts.jsonl`, `agent_transcripts.jsonl` | stage 2: every `AttemptRecord` (P6.2's crash-safe log) and every conversation, reasoning included |
| `agent_submission.json`, `agent_metrics.json` | stage 2: P6.2's two attempts per test input; the score and the measures, per task and in total |
| `shot_metrics.json` | `P6.3-05`: the single shot's score, solves against NVARC's, the reading |
| `direct_attempts.jsonl`, `direct_transcripts.jsonl`, `direct_metrics.json` | `P6.3-04` arm D: every attempt (views under `<id>~v1`), every conversation, the report |
| `prompt_counts.jsonl` | stage 2 (revision 6): each attempt's opening count (or its error); `metrics.json`'s `prompt_counts` compares it with the first call's `prompt_tokens` |
| `forcing_check.json` | stage 2 (revision 5): the forcing check per task (turn 1, the forcing call, the growth, the render check, the verdict), the reasons of a fail, and the verdict |
| `preserve_check.json`, `engine_calls.jsonl` | stage 2: the preserve check's numbers and verdict (revision 4); one row per engine call with where its reasoning arrived, its thinking switch, tokens, and finish reason |
| `timing/timing-engine.jsonl` | stage 2: P6.2's engine timing records |
| `harness_preflight.json` | stage 2: the harness preflight: the parent's interpreter, `sys.path`, user id, temporary folder, package files, import roots, and child command; the task; the cell's output; the child's interpreter, path, and package files; the verification; on a failure the step, the error, and the traceback |
| `harness.log` | stage 2: every log record of the `arc26` package from INFO up, with tracebacks |
| `budget_check.json` | `P6.9-01`: the budget check (`ok`, `forced`, the answer, the reasoning's tail, tokens, finish) |
| `shot_metrics.json` (`P6.9-01`'s keys) | `strict`, `forced_answers`, and per task `budget_forced`, `lenient_answers`, `right`, `implied` (section 2.7); `direct_attempts.jsonl` rows add `budget_forced` and `answer_lenient` when not 0 or empty, and `engine_calls.jsonl` rows `thinking_token_budget` |

The download notebooks write `run_logs/status.json`, `MANIFEST.json`, `summary.md`, and
`pip.log` or `install_check.json`; their weights and wheels stay on Kaggle.

## 6. How to read the results

Read `summary.md` first, then `throughput_cells.json`. The decision rule is the brief's
(section 1); in short, with $R_{32}$ the steady rate at a 32k mean context and 32 requests:

- **Good** ($R_{32} \ge 400$ tokens per second, no kill criterion, the probe verifies at least 3
  of 10): stage 2 runs as drafted, its budgets set from the measured rate.
- **Borderline** ($300 \le R_{32} < 400$, or 1 to 2 probe tasks verified): stage 2 runs with
  shorter attempts (shorter contexts raise the rate) and phase 1 first.
- **Bad** ($R_{32} < 300$ and the 8k rate at 32 requests under 600, or no probe task verified with
  sane outputs): the throughput half of ADR-020's gate fails; the orchestrator takes the
  options to the human (the official FP8 build on SGLang, the 9B fan-out model, or the overturn).

`P6.3-01` read Borderline (attempt 3, `docs/experiments/P6.3-01.md`): $R_{32} = 377$, $R_8 =
535$, 4 of 10 probe tasks verified.

For `P6.3-02`, read `agent_metrics.json`'s `gate` first: $M$ (misses solved by verified programs),
the precision of verified programs, the engine's health, and the row (Good, Bad, Borderline,
Fault; the brief's section 1). Then `forcing_check.json` (a Fault there means no phase ran), the
forcing turns and forced programs per phase (how often a cut turn yielded a program), the misses
scored otherwise or in part (they hint at what programs miss), the phases' stop reasons,
`metrics.json`'s `engine_calls` (finish reasons of all calls and of the forcing turns), and
`engine_calls.jsonl`'s reasoning locations.

For `P6.9-01` read `reports/P6.9-01/merge.md` (`scripts/merge_replay.py`) against the brief's
rows, in order: a Fault or Fault (rate) of the script stops the reading; 0 of the 5 harness tasks
solved by the shot opens a harness investigation (prompt, quantization) before anything else;
the middle bucket's merged net (merged less NVARC, in tasks) less the expected attempt-2 loss
$6\ell = 0.72$ of at least 1 extends Submission 4's queue to $[0.5, 0.7)$, placed after
$(0, 0.5)$ and before confidence 0 (`selection.mid_bucket`), else it stays off; the in-request
forcing stays (`request.forcing_in_request`) when the cut tasks' outputs got an answer at least
$13/15$ of the time (`P6.3-05`'s separate forcing turn) and the budget check passed, else
Submission 4 keeps the separate turn. The lenient parser's effect (`score` against
`strict.score`) is reported, not a row.

Also read: the boot facts (a KV pool far under about a million tokens means fewer sequences at
32k than planned), TTFT at $c = 64$ and 32k (the queueing an agent would see), preemptions (the
pool overflowing), the probe's tokens per verified task, its length stops (answers cut at the
cap), and its rounds without a program (format failures, not reasoning failures).

## 7. Failure modes

| Symptom | First thing to change |
|---|---|
| A download notebook's `ok` is false: disk, hash, or Hub listing | Read `status.json`; a hash or listing mismatch means the revision's files changed: re-pin from the Hub API. Disk: check `free_bytes_before` against 19.47 GB. |
| The install check fails (a missing wheel, `pip check`, an import) | Read `install_check.json`; add the missing pin to the constraints (resolve again with uv, metadata only) and push the wheels notebook again. |
| Engine preparation: `flashinfer_jit_cache still imports without __version__` | Read `engine_prep.json`'s `before` and `after`: a folder with files the providers did not own, or one outside the venv; delete it or add the shim `flashinfer-jit-cache` of the same version to the wheelhouse. |
| Engine preparation: the import check fails | Read the error and traceback in `engine_prep.json`: a missing or mismatched package in the wheelhouse (resolve again, push the wheels notebook) or a module renamed in vLLM (fix `wheels.engine_imports`). |
| Preflight: CUDA unavailable, fewer than 4 GPUs | Kaggle gave another accelerator or the CUDA 13 runtime does not load on the driver: read `preflight.log` (driver 580.x supports CUDA 13.0); no retry without a fix. |
| The engine exits during boot | `run.json`'s problem quotes the log's first exception lines; then read `vllm_server.log`'s head: an unknown flag (check against the installed vLLM), the architecture, or the compressed-tensors format (vLLM pins 0.17, the checkpoint was written by 0.18.1) |
| Out of memory at boot (profiling or graph capture) | Lower `gpu_memory_utilization` to 0.85, then `max_num_seqs` to 48, then `max_model_len` to 49,152. |
| The inputs stage stops with `direct.task_ids: task ids not in the challenges: [...]` | The mounted challenges file is not the K0 one (sha256 `e7c62a4b...`, `arc26.data.DEV120`): `run.json`'s problem names the missing ids; fix the ids or the input, never the loader. |
| Boot slower than 1,800 s | Weight loading from the mount: read `weights_load_s`; raise `boot_timeout_s` and `VLLM_ENGINE_READY_TIMEOUT_S` together. |
| Degenerate outputs (runs of one token), the run stops | NaN logits: rerun with prefix caching off (`enable_prefix_caching: false`), then with the FlashInfer or default attention backend; report the cell. |
| Many requests fail (HTTP errors, resets) | The server died or overloaded: read the log's tail; lower the top concurrency. |
| A 32k cell is cut at its timeout | Prefill is slower than estimated: read `prefill_span_s`; give the 32k phase more time or drop $c = 64$. |
| The probe verifies nothing but answers make sense | Read `probe.jsonl`: rounds without a program (the format), length stops (raise `max_tokens` or the effort), checker errors. |
| Stage 2: many attempts end with `engine_error` | The engine died or rejected the requests: read the server log's tail and `engine_calls.jsonl`; the script's sanity request passed, so look for a format change. |
| The harness preflight fails (exit 1 within seconds) | Read the printed traceback, or `harness_preflight.json`'s `failed_step` and `traceback`: a REPL or verifier worker that does not start quotes its child's own traceback; compare `parent` and `child` (interpreter, path, package files). |
| Stage 2: a phase's report shows `raised` attempts, the gate reads Fault | The first traceback of each kind is on standard output and in `agent_metrics.json`'s `attempt_errors`; all of them are in `harness.log`. |
| Stage 2: verifier workers replaced, sandbox failures | Programs that crash or hang the workers: read `agent_attempts.jsonl`; P6.1's limits (memory, call time) are in the agent section. |
| The forcing check fails (exit 1 before phase 1, gate Fault) | Read `forcing_check.json` per task: turn 1 not cut (lower `forcing.turn1_max_tokens`); `forcing.thinking` not false (the loop's switch); no program with prose in `text_head` (the forcing message's wording); `growth.kept` false (the reasoning never reached the prompt: the engine's `reasoning_content` or `preserve_thinking`); `render.status` failed (the template: which part is missing); an empty forced reply with the reasoning in `engine_calls.jsonl`'s `location` (the reasoning parser filed the answer as reasoning, which P6.2's client moves only when the content is empty). |
| `P6.3-04` arm D: most answers missing | Read `direct_transcripts.jsonl`: answer blocks malformed (`[answer] not recorded`), cut replies whose forcing turn wrote prose, or the deadline. |
| `P6.3-04` arm D: the original far above the view | Recall of the public DEV40 outputs: read the arm's score through the view only. |
| Stage 2: the gate reads Fault with `engine.stalls` in `agent_metrics.json` | A phase ended on a hung engine (its calls cancelled at the deadline with too few tokens or no answer for 20 minutes): read `vllm_server.log`'s tail and `server_metrics.jsonl` (requests running, KV usage) around the phase's last answered call. |
| The forcing check passes but `prefix_kept` is false | The template's effort note differs from `QWEN38_EFFORT_NOTES`: forcing turns are prefilled in full; correct the text before the next run (no effect on the result's validity). |
| The preserve check fails | The variant that broke tells where: `on` and `reasoning` field both 0 means the template never saw the reasoning (keep `preserve_thinking` off); `sent_not_kept` above 0 means the switch is ignored. |
| Requests recorded with `EOFError: the stream ended before [DONE]` or a connection error | The server dropped streams: read the log's tail for a crash or an OOM; the cell's error share decides the kill. |
| The preserve check's tasks are inconclusive with an `error` | Read the error in `preserve_check.json` and `engine_calls.jsonl`; the agent's phases ran anyway, and the engine's health is in `agent_metrics.json`. |
| Stage 2's phase 2 ends with most attempts at `deadline` | Expected at about 300 tokens per second: 32 attempts at once get about 9 tokens per second each, one full xhigh turn and part of a second in 3,600 s; read the phase's `completion_tokens` and `wall_s` for the rate the agent actually got. |
| `P6.9-01`: `budget_check.json` `in_request` false (its `error` an HTTP 400, or `forced` false) | The server did not take the budget: the run went on with the in-request forcing off, `P6.3-05`'s request and its separate forcing turn, so the wave is read with row 5 Revert. Read the boot log for `Using V2 Model Runner`, the `--reasoning-config` value in `non-default args`, and the warning `Auto-initialization of reasoning token IDs failed`. |
| `P6.9-01`: attempts with `budget_refused` or `turn_cuts` | A refused field (HTTP 400) after a passing check: that attempt went on as `P6.3-05`'s; a cut request: answered from the opening alone. Read `engine_calls.jsonl` for the errors and the rate; a cut means the rate fell under 15.9 tokens per second per stream or the server ignored the budget. |
| `P6.9-01`: forced notes in the middle of answers | The V1 runner served the run (it detects only the whole forced string, not the natural `</think>`): set `VLLM_USE_V2_MODEL_RUNNER=1` in `server.env`, or drop the phrase. |
| `P6.9-01`: cut tasks answer less often than $13/15$ | Read `direct_transcripts.jsonl` after the note: prose instead of answer blocks means the phrase does not steer; a reply ending at `length` means the answer outran $F$. |
| Wall-time stop (exit 3) | The phases overran: the finished cells and tasks are in the outputs; rerun the rest with `--resume-from`. |
