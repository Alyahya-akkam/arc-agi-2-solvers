# Runbook: version 11 (`scripts/fn_probe.py`), `scripts/fn_env.py`, and their builder

Task P5.2 (`docs/tasks/P5.2.md`, rounds 1 and 2). Runs `P5.2-00` (CPU,
`scripts/fn_env.py`, configs `configs/P5.2-00.yaml` and `configs/P5.2-00-fa2.yaml`), the CPU twin
`P5.2-01-cpu` (`configs/P5.2-01-cpu.yaml`), and `P5.2-01` revision 2, version 11's own run (4 x
L4, `configs/P5.2-01.yaml`, brief [`docs/prerun/P5.2-01.md`](../prerun/P5.2-01.md)). Version 11 is
the notebook `alyahya737/arc26-submission` that `scripts/build_fn_notebooks.py` writes; its cells
run `scripts/fn_probe.py`. The tool loop is `arc26/agent/toolshot.py` (task P5.3). Fake server
`scripts/fn_fake_sglang.py`; local report `scripts/fn_report.py`. Tests `tests/test_fn_env.py`,
`tests/test_fn_probe.py`, `tests/test_fn_notebook.py`, `tests/test_build_fn.py`. The engine recipe
is `docs/research/flash_next_sglang.md` on `task/ideas` (`dd02113`, cited as R0 to R9); the
harness and its rule are `docs/research/flash_next_harness.md` (`4ab9d6f`).

## 1. Why this script exists

The leading ARC-AGI-2 teams run Qwen3.8-Flash-Next (125B MoE, 6B active, a 51.2B n-gram table)
on every task with long thinking, and Python tools are where the published harness gains are
(the harness note: +7 to +27 points across eight models). Our version 10 runs Qwen3.8-27B on
vLLM. The human's order (29 Oct): "we need to win score not throughput ... focus on flash-next".

- **`P5.2-00` (free, done 29 Oct, Pass):** the stack installs on Kaggle's image with a Python
  3.12, every compiled kernel carries code an L4 (sm89) can run, the fork knows every flag, and
  the FA2 wheel and a CPython 3.12 tarball are kept as a kernel output for the GPU runs.
- **Version 11:** in a scored rerun, Flash-Next on every hidden task, LLM first, with "think with
  Python" (plain if the sandbox fails at startup), filling `submission.json` as tasks finish.
- **`P5.2-01` revision 2 (about 5.3 quota hours):** the same notebook in development mode: arm A
  plain and arm B tool on `P6.9-01`'s 15 DEV120 tasks, paired on one server. It feeds which arm the
  human submits on Sat 31 Oct.

## 2. What it does, step by step

### 2.1 `fn_env.py` (`P5.2-00`, CPU)

Each step records its result in `run_logs/fn_env/report.json` (rewritten after every step); a
failed step records its error and the next one still runs; a step whose input is missing says
why it was skipped.

1. **listing:** `/kaggle/input` to 7 levels (`input_listing.txt`), each input found at its
   configured path or by its marker file, free disk, CPUs, RAM.
2. **python:** `sys.version`. With `standalone: always`, a python-build-standalone 3.12 from
   `uv python install 3.12` (the wheelhouse's uv, internet on) is saved as
   `/kaggle/working/python312.tar.gz` even when the image is 3.12, so `P5.2-01` still has a 3.12
   if the latest image drifts (R1, risk 9). The venv uses the image's Python when it is 3.12.
3. **venv:** dfranzen's launcher (cell 12): uv from the wheelhouse's uv wheel,
   `uv venv --python <python>`, `uv pip install --no-index --find-links <wheels> -r
   requirements.lock`, the SGLang wheel again `--no-deps`; `uv pip check`; `versions.txt`.
4. **fetch:** the FA2 wheel `flash_attn-2.8.3+cu130torch2.13-cp312` (R3) into
   `/kaggle/working/wheels`, installed `--no-deps` into the venv and imported.
5. **patch:** the recipe's one edit (R4) checked, not applied: the installed
   `qwen_sparse_attn_backend.py` must hash to the fork's `2ce24d66...` and the pattern must match
   once; the code around the match is saved (`patch_fa2_scratch_dtype_context.txt`) so the
   variable name `q` can be read before the GPU run.
6. **imports:** each module in its own process; without CUDA, errors are expected and recorded.
7. **server_help:** `python -m sglang.launch_server --help` and the fork's `server_args.py` give
   the flags the fork knows; every flag of `configs/P5.2-01.yaml`'s `engine.flags` (copied in
   by the builder) must be among them, or the step fails and names the unknown ones.
8. **arch:** every `.so`, `.cubin`, and `.fatbin` of the listed packages (and top-level modules
   such as `flash_attn_2_cuda*.so`) read for its CUDA architectures. A fat binary starts with the
   magic `0xBA55ED50` (version 1, header 16 bytes); each entry header names its kind (1 PTX, 2
   ELF) and its architecture; a standalone cubin's ELF header names its sm in `e_flags` (bits 0
   to 7 for ABI version 7, bits 8 to 15 from version 8). A file runs on the L4 when
   $$\exists\,\text{cubin } sm_{XY}: X = 8 \wedge Y \le 9 \quad\text{(native)}, \qquad\text{else}
   \quad \exists\,\text{PTX } compute_Z: Z \le 89 \quad\text{(JIT)}.$$
   Per package the verdict is `native`, `ptx`, `partial` (some files have neither), `none`, or
   `no device code` (JIT packages such as triton). `cuobjdump` cross-checks three files per
   package when the wheels bundle it.
9. **model:** the model's and the drafter's small JSON files, each file's size, missing shards,
   and the bytes of each component summed from the safetensors headers only (8-byte length, then
   the JSON; no weight is read): the n-gram table, routed experts, shared experts, the vision
   tower, MTP, Gated DeltaNet, QSA, embeddings and head, the router, hyper-connections, the rest.
   The chat template is saved (`model_chat_template.jinja.txt`); the tokenizer's sha256 is
   compared with the 27B's (equal means our prompts and token ids carry over).
10. **template:** the venv's tokenizer renders a two-message chat at `reasoning_effort: xhigh`,
    the default, and thinking off; the generation prompt's tail and the ids of `<think>`,
    `</think>`, `<|im_end|>` are recorded, and the forcing note's token count.
11. **package:** `wheels/MANIFEST.json` (sha256 of each produced wheel), the Python tarball, and
    `/kaggle/working/fn_env_output.json`, the marker `P5.2-01` finds the output by.

`configs/P5.2-00-fa2.yaml` (`alyahya737/arc26-fn-fa2`) runs listing, python, venv, then **fa2**:
the 2.8.3.post2 sdist from PyPI (sha256 from PyPI's JSON), `pip wheel --no-build-isolation`
against the venv's torch with `FLASH_ATTN_CUDA_ARCHS=80`, `MAX_JOBS=4`, `NVCC_THREADS=1`, the
wheel's `.so` scanned as above. It is the fallback if the prebuilt wheel fails its import test.

### 2.2 Version 11's notebook (`scripts/build_fn_notebooks.py`)

Three code cells around the solver, so the file Kaggle scores is valid whatever happens:

1. **Setup:** the session's start $t_0$ (the time plan counts from it), the embedded config, the
   code dataset found by `fn_code.json` and every file checked against the build's sha256, and an
   all-`[[0]]` `submission.json` for `arc-agi_test_challenges.json` from
   `arc26.submit.finalize_submission` (standard library only, imported from the wheel as a zip).
2. **Solve:** `fn_probe.py --config fn_config.json --session-start t0`, killed 120 s after the
   mode's `end_s` if it hangs; a failed code check skips it (the placeholders stay).
3. **Final check:** `finalize_submission` again over the file itself: every valid attempt kept,
   anything else `[[0]]`, written atomically, read back, `valid` asserted.

### 2.3 `fn_probe.py` (version 11's solver)

**Mode.** Competition when `KAGGLE_IS_COMPETITION_RERUN` is set (a scored rerun), development
otherwise (the version's own run, the twin).

An **outer process** (the image's Python 3.13, standard library only):

1. **inputs:** the wheelhouse, `P5.2-00`'s output, the code dataset (checked again), the model,
   the competition.
2. **local_inputs:** the wheelhouse and `P5.2-00`'s output copied to `/tmp/fn/inputs/` with
   version 9's warm-up pattern (8 readers, 256 MiB pieces, each piece through its own handle into
   `<name>.part`, renamed when its size matches); the copied folders replace the mount's in
   `state["paths"]` (the mount's kept in `mount_paths`). At `max_s` (1,800 s) or 120 s without a
   byte, every file not done becomes a symlink to the mount; without room (the bytes plus 40 GB)
   or on an error, the input stays on the mount. `local_inputs.json` records the bytes, seconds,
   rate, linked files, and errors. Only then the **precache**: a parallel read of the model's
   files (16 threads, 32 MiB chunks, at most 180 s) beside the install, measuring the mount's
   rate; the two never read the mount at once.
3. **python:** the image's if it is 3.12, else `P5.2-00`'s CPython 3.12.14 tarball unpacked in
   `/tmp`.
4. **install:** from the local copies, uv from the wheelhouse's uv wheel, the venv on that
   Python, `--no-index --find-links` of the lock (and `P5.2-00`'s wheels, at most 3,600 s),
   SGLang `--no-deps`, the FA2 wheel and the `arc26` wheel `--no-deps`; then the recipe's edit (R4), hash-guarded, exact-once; if it does
   not apply, `--kv-cache-dtype bfloat16` replaces the config's.
5. **cuda:** dfranzen's overlay (the venv's `nvidia/cu13` as `CUDA_HOME`, links, `libcuda.so`,
   compilers, caches in `/tmp/fn/cache`); only these overrides go to `state.json`.
6. The **inner process** runs with the venv's Python.

The **inner process**:

1. **tasks:** competition: every task of `arc-agi_test_challenges.json`, ordered by the rendered
   prompt's token count (smallest first, then id); development: `solve.task_ids` in order, from
   the evaluation challenges. Competition mode starts the **submission writer**.
2. **tool_self_test:** the tool loop on a 2 x 2 identity task with a scripted model (one
   `python` call with `check(transform)`, then an answer): the REPL, the verifier, and the call
   parser must work here. A failure drops the tool arm (competition runs plain).
3. **arg_check:** the fork's own `prepare_server_args` and `resolve_once` (the path
   `launch_server.py` takes) on the boot's flags and on the fallback boot's, each in a child
   process with the server's environment, at most `arg_check.timeout_s` (300 s) each. Verdicts:
   `accepted`; `refused` (a `ValueError` raised inside `sglang/srt/`, or the parser's exit 2);
   `device_stop` (the resolution stopped at a CUDA or device query, as on a CPU box: the flags
   resolved up to that point, `stopped_in` names where); `error` (anything else). A refusal
   fails development mode (exit 1, `arg_check_refused` in `state.json`); competition only
   records it. The boot runs either way. `arg_check.json` holds each flag set's argv and verdict.
   The twin checks the GPU flags with the fake's own flags dropped (`arg_check.drop_flags`) and
   the GPU fallback flags (`arg_check.retry_flags`).
4. **boot:** `python -m sglang.launch_server` with `engine.flags` (R6); `/health` polled; the
   pool and the request limit from `/get_server_info` or the log. A failed boot boots once more
   with `engine.boot_retry_flags` (BF16 KV, no CUDA graphs).
5. **template_check:** the first task's prompt counted by us, by the chat endpoint, and by
   `/generate`, and one generated token (where #36968 fails on sm89; 900 s for the first-use
   compiles); without it, one more boot with the fallback flags (BF16 KV,
   `--cuda-graph-backend-{decode,prefill} disabled`), then development stops and competition
   solves anyway (the placeholders are the floor).
6. **solve:** the arms (competition `solve.arm`, development `solve.dev_arms`) and the
   **budget plan** (`plan_budget`). Solves need long thinking: the 27B's 67 long-cap attempts
   (`docs/research/budget_vs_solves.md` on `task/ideas`) give $S(b)$ 1.5% at 16k, 3.0% at 24k and
   32k, 5.2% at 40k, 8.2% at 48k, 9.7% at 56k and 65.5k, 17.9% at 94.2k, and all 56 forced answers
   below 94k scored 0, so the expected solves $N(c)\,S(b(c))$ peak at long budgets. With $s$
   streams sharing the pool $P$,

   $$c(s) = \min\left(86{,}016,\ \left\lfloor \frac{P}{s} \right\rfloor - S\right), \qquad
   b(s) = c(s) - o_{\max} - F - m, \qquad g_t = \min(C_{\max},\ c - o_t - m),$$

   with $S = 0$ tokens of linear-attention state (SGLang keeps it in its Mamba slots, outside the
   KV pool), $o_{\max}$ the largest opening, $F$ the
   forcing reserve, $m = 128$ the context margin, $C_{\max} = 65{,}536$. The plan takes the most
   streams $k \le s \le \min(L, 8)$ ($k$ the arms) with $b(s) \ge 42{,}000$
   (`planner.min_thinking_tokens`), else $k$ streams at $c(k)$ with the shorter thinking
   (`budget.met` false); no plan holds more than the pool. The coverage $s\,r\,T/\bar g$ ($r = 30$ until measured, $T$ the competition's
   generation window in both modes, $\bar g$ the mean cap) is reported, not targeted, and
   re-reported once at the measured rate (`replan.json`). At the measured $o_{\max} = 6{,}896$
   (mean 1,782, the visible test file's 240 tasks): 180k gives 3 streams at 60,000 (57 to 76 tasks
   covered at 30 to 40 tokens per second), 248k 4 at 62,000 (74 to 98), 390k 7 at 55,714 (144 to
   192). Each arm's streams are a gate of $\lfloor s/k \rfloor$
   ($k$ arms). **One budget rule in both arms:** at most $g_t$ generated tokens, thinking stopping
   `forcing_tokens` short of it, the same forcing note; the tool loop's replies at most $c/5$.
   Every request keeps input plus `max_new` at most the context less $m$; a refusal (an error
   chunk) is recorded as finish `error` with its message. Each arm's workers take its queue in
   order while the forcing time $T_f = T_{\text{stop}} - 360$ s is at least `min_start_s` away.
   - **Arm A (plain):** `/generate` with the prompt's ids, `max_new_tokens` $g_t - F$; cut there
     or at $T_f$, the reply is continued from its own ids plus `thinking_end_note` (two blocks per
     test input) and `</think>\n\n`, `max_new_tokens` $\min(F, g_t - n)$; the blocks after the last
     `</think>` parsed leniently, two attempts per test input by `shot_answers`.
   - **Arm B (tool):** `arc26.agent.toolshot.solve_with_tools` with `generate(ids, max_new,
     cut_at, kind, extra)` built on the same streaming request: requests stop at
     `</tool_call>` (kept), a call runs in the task's REPL, and the next request is the previous
     ids plus the generated ids plus the template's tool turn, which the radix cache serves.
     Its `tool_config` per task (`tool_budget`): cap $g_t$, context $c - m$, reply tokens at
     most $c / 5$, the plain arm's forcing reserve and note. Slots: the first two distinct grids
     of $(A_1, P, A_2)$.
   - **Health:** a watch polls `/health` every 30 s; 3 misses mean dead and the server restarts
     (at most twice, not with under 40 minutes left); a task whose requests failed with the
     engine is queued again once; when the engine cannot come back, the workers stop and the
     placeholders stay.
   - **Writer** (competition): each finished task's attempts replace its placeholders, the file
     written atomically and validated each time; a test input already answered is never
     overwritten, and a second slot without a grid stays `[[0]]`.
   - **Stops:** once 2 attempts have finished, more than half failed or degenerate (64 identical
     tokens in a row) ends new starts.
6. **logs:** `engine.log` (at most 20 MB), `engine_tail.log`, `decode_lines.json`, `summary.md`.

**Time plan** (seconds after $t_0$, `time_plan`): competition generation stop 39,600, end
40,050 (Kaggle's 43,200 less version 10's 2,700 allowance for the time before the first cell
less 450); development 8,400 and 8,850 (the brief's Timeout 10,800 less 1,950 for the time before
the first cell and the save).

Every request appends `{puzzle_id, op, seconds, tokens}` to `timing-inner.jsonl` (ops `fn_think`,
`fn_tool`, `fn_force`) and its row (prompt, cached, and generated tokens, finish, seconds, first
token, decode rate $r = (n-1)/(e-f)$) to its attempt.

### 2.4 `fn_report.py` (local, after the pull)

Per arm, its attempts become a submission scored by `arc26.evaluate.score_submission`; on the
tasks $P$ both arms finished,

$$S_X = \sum_{t \in P} s^X_t, \qquad c_B = \frac{\sum_{\text{tool turns}} \text{cached}}
{\sum_{\text{tool turns}} \text{prompt}},$$

and the rule (`decide`): **B** when $S_B \ge S_A$, $w_B \le 1.3\,w_A$ (median wall per task) and
$c_B \ge 0.9$; **A** when $S_A - S_B \ge 2$ or $w_B > 1.5\,w_A$; a Fault means **A**; otherwise
the human's call. Also: median decode rates, forced shares, calls and failures, programs, block
counts, the 27B's `P6.9-01` scores on the same tasks. Writes `fn_report.md` and `fn_report.json`.

## 3. Why it is built this way

- **SGLang, not vLLM.** The human's order (task brief). The research note's vLLM route also
  fails at model construction for Intel's checkpoint in vLLM 0.31.0 (`flash_next_shot.md` 2.1).
- **The Pennyroyal fork's wheelhouse as is, not upstream SGLang.** Upstream v0.5.21 asserts
  `sglang-kernel >= 0.4.7` and `flashinfer_python >= 0.6.18` at start and pins the n-gram table
  at power-of-two size (64 GiB at TP 4 of 80.5 GiB free); the fork pins its exact 47.68 GiB
  (R0, sgl-project/sglang#40626). Every compiled wheel in it carries sm89 or sm80/sm86 code (R0),
  which `P5.2-00` measures instead of assuming.
- **Kaggle's latest image with `P5.2-00`'s CPython 3.12, not our pinned cp311 image or
  dfranzen's pinned image.** The wheelhouse is cp312; the latest image ran 3.12.13 in K0-latest
  but 3.13.15 in `P5.2-00` (29 Oct), which is why `P5.2-00` keeps a standalone 3.12.14 tarball
  always: the venv no longer depends on the image's Python. dfranzen's pinned image ran only on
  RTX PRO 6000 sessions.
- **The prebuilt FA2 wheel, with a source build as fallback.** FA4 (the only FlashAttention in
  the wheelhouse) fails the QSA decode at head dim 256 on sm89 (#36968); a community wheel for
  torch 2.13 + cu130 + cp312 exists (R3). Building from source takes 1 to 3 h on a CPU session
  (estimate), so it is a separate notebook that runs only if the wheel fails its import test.
- **A hash-guarded edit with a flag fallback, not BF16 KV from the start.** FP8 KV gives 1.9x
  the pool (6,912 against 13,056 bytes per token per GPU; R6), the binding limit of this model
  on 22.5 GiB GPUs (both research notes). The edit is one token, guarded by the file's sha256 and
  an exact-once match; if either check fails the boot runs with BF16 KV instead of failing.
- **TP 4 + EP 4, Triton for every linear-attention path, 44 Mamba slots, 0.92, context 86,016,
  8 requests** (R6): EP keeps the 640-wide experts on Marlin (TP alone falls back from it);
  FlashInfer's GDN kernels need sm90+; unset, SGLang gives Mamba 47% of the pool; 0.92 leaves
  1.80 GiB per GPU for activations; 86,016 is version 9's request bound rounded up.
- **No MTP.** The drafter costs about 1 GiB per GPU, about 150k tokens of a 0.25M pool (R7), and
  QSA + NEXTN has a silent-corruption report (#37111); streams are the binding limit.
- **"Think with Python" as version 11's default arm, plain as its fallback, and a paired DEV
  check, not a choice by prior alone.** The harness note's evidence (pivotools' REPL arm on
  GPT-OSS-120B 6.1% to 26.4%, five more before-and-afters, all one direction) is external; our
  own Phase 6 programs on the 27B lost to direct answers under capped turns. The paired run on
  one server measures B against A on our model, cap, and pool, with the note's rule fixed in
  advance; the alternatives (arm B alone against revision 1's A, about half the cost; the
  answer-check turn) lost because they compare across sessions or measure a weaker harness.
- **The tool loop behind `generate`, not inside `fn_probe.py`.** P5.3's module owns the loop
  (parsing, the REPL, budgets, slots) and stays free of HTTP; `fn_probe.py` hands it the same
  streaming request the plain arm uses, with `stop` and `no_stop_trim` for the tool turns, so
  both arms share the engine path, the cut, and the timing records.
- **A self-test of the sandbox before the boot.** The REPL subprocess and the verifier depend
  on the machine; a scripted two-turn task finds a broken sandbox in under a second and drops the
  tool arm, instead of failing every hidden task's first call.
- **A notebook of three cells around the solver.** A code competition submits a notebook. The
  placeholder file comes first, before anything can fail; the final check runs whatever the
  solver did; so a crash, a hang, or a failed code check leaves a valid all-`[[0]]` file (score
  0 for the unanswered, never an invalid submission). The alternative, the version 10 builder
  (`scripts/build_kaggle_notebook.py`), is written for NVARC's stack, its pinned cp311 image, and
  vLLM; adding a second engine there would touch protected and P5.1 files.
- **A live submission writer.** Kaggle scores the file at the session's end; writing after every
  task means a session killed at any moment keeps every finished answer. An answered input never
  reverts, so a requeued or late attempt can only add.
- **A health watch with restart and requeue.** The version 10 review found that a server dying
  mid-run zeroes every later task. A restart costs a boot (caches warm); its tasks run again. Two
  restarts at most: a crash loop must not eat the session.
- **One more boot with fallback flags.** BF16 KV avoids the edited FP8 path, no CUDA graphs avoids
  FA2 under graphs (recipe risk 2): together they trade speed and pool for a server that answers.
- **A long-budget plan, not full coverage, and not version 10's fixed 86,016 context.** Full
  coverage at the measured pool means about 17k to 22k of thinking, where the 27B solved 1.5% to
  3.0% and every forced answer below 94k scored 0 (`budget_vs_solves.md`); expected solves peak
  at the largest context of a stream count with thinking near 45k or more, which is the plan's
  42,000 floor. The alternatives lost: full coverage (fix round B) solves almost nothing per
  task; one stream at 86,016 wastes the pool; overcommitting streams invites retractions, each a
  re-prefill of up to 86k tokens.
- **One budget rule for both arms.** The pair compares harnesses, so the tokens must match: both
  generate at most $g_t$ with thinking stopping $F$ short of it and the same forcing note (P5.3's
  loop already stops thinking $F$ short of its cap; the plain arm used to think to the cap and
  force $F$ more).
- **Native `/generate` with our own chat template, not the chat endpoint.** The thinking cap is
  then exactly `max_new_tokens`, and the forcing turn continues the cut reply from its own token
  ids, which the radix cache already holds. The chat endpoint would need a server-side thinking
  budget (vLLM's `thinking_token_budget`, which SGLang lacks) or a second request that
  re-renders and re-prefills the reply. The template check compares our prompt's token count
  with the chat endpoint's, so a template mismatch shows.
- **Incremental streaming every 16 tokens, not SGLang's default.** The default sends the whole
  reply so far in every chunk: quadratic over a 65,536-token reply (the smoke's deadline test
  hung on it before the change). Streaming at all, rather than one blocking request, keeps a
  reply cut at the deadline, which the forcing turn then continues.
- **Threads and `http.client`, not `httpx` or `aiohttp`.** At most 8 streams; the standard
  library needs nothing from the venv beyond what the inner process imports for prompts
  (numpy). A cut closes the socket that `http.client` hands to the response; SGLang aborts the
  request and frees its KV.
- **One outer and one inner process.** The outer one must run before the venv exists (standard
  library only, the image's Python); the inner one needs the venv's `transformers` for the
  template and `arc26` for the prompts and the parser, which are the shot's own
  (`arc26.agent.prompts`, `arc26.agent.loop.parse_answer`, `arc26.agent.shot.shot_answers`), so
  the solves measure the same request as `P6.9-01`.
- **The config embedded in the notebook, the code in a dataset.** The image's Python may lack
  PyYAML, so the config travels as JSON in cell 1; `fn_probe.py`, the fake server, and the wheel
  come from the code dataset `alyahya737/arc26-code-fn`, each file checked against the sha256 in
  the embedded `build` section. Its `fn_code.json` holds no commit, so a docs-only commit needs no
  new dataset version. The twin extends version 11's config (`extends`), so the two cannot drift.
- **The 15 tasks of `P6.9-01`, in its order, paired.** The only same-task comparison with the
  27B we have (6.0 of 15, at a 94,208 cap); both arms take the tasks in the same order, so a
  task's two attempts start close together and the comparison is on the tasks both finished.
- **The forcing note asks for two blocks.** In `P5.1-01` both forced replies wrote one block per
  output, leaving attempt 2 at `[[0]]`; arm B's verified program fills that slot too.
- **Kill criteria in the script.** Nothing can cancel a running kernel (TEMPLATE.md, section 7):
  the boot timeout, the first generation, and the error and degenerate shares stop the run
  before it spends its whole budget.
- **The wheelhouse copied to local disk before the install, not installed from the mount with a
  longer timeout alone.** Twin 6 (`P5.2-01-cpu6`, 30 Oct) passed the lock's 1,800 s reading the
  6.6 GB straight from GCSFuse (its SGLang wheel alone took 54.6 s), where twins 2 to 5 took 104
  to 121 s; `P6.9-02` read 24 MB/s with one reader, version 9's copier 469.2 MB/s with 8 readers
  of 256 MiB pieces (`P4.7-02`). A longer timeout alone keeps uv's reads on the mount; a
  single-threaded `shutil.copytree` is the one-reader case. The copy links what it cannot finish
  to the mount, so a slow mount costs time, not the install. The precache waits for the copy:
  twin 6 had it off, so it did not cause that fault, but on the GPU it would read the same mount
  at the same time. Install timeout 3,600 s: the competition's worst case that still solves then
  leaves a generation window of 27,880 s instead of about 36,900 (the config's comment).
- **TP collectives on PCIe: NCCL only.** `P5.2-04` (Modal, 2 x L40S, TP 2, 30 Oct) hung in the
  decode graph capture: the fork's `LogitsProcessor` builds a `MultimemAllGatherer`
  (`triton_symm_mem_ag.py`) whenever TP > 1, whose first eager call waits on a store rendezvous
  for a CUDA multicast buffer; it timed out after 600 s and the server never became healthy.
  No flag or variable turns it off (`--enable-symm-mem`, `--enable-torch-symm-mem`,
  `--enable-nccl-nvls` are opt-in and unset), so a hash-guarded edit sets its `enabled` to
  False, and `--disable-custom-all-reduce` keeps the IPC custom all-reduce off at TP 2 (the
  fork already drops it at TP 4 without NVLink). Alternatives: `NCCL_P2P_DISABLE=1` (the fork's
  own hint) stages every transfer through host memory and does not touch the rendezvous;
  `--enable-tp-lm-head-all-to-all` changes the head's layout, not the gatherer.
- **The Marlin MoE's top-k sum in torch.** `P5.2-04`'s fallback boot died on its first forward
  in `sgl_kernel`'s `moe_sum_reduce` ("small-token launch failed", from
  `fused_marlin_moe.py:423`). The edit replaces that call with vLLM's
  `torch.sum(x, dim=1, out=output)` (and the scale when it is not 1.0).
  `--moe-runner-backend triton` lost: its path calls the same kernel
  (`triton_utils/fused_moe.py:878` and `907`). The kernel's check reads `cudaGetLastError`, which
  also returns an earlier launch's failure; the same reduce ran in the first boot's decode
  warm-up (8 tokens) before the hang, so the fault may sit earlier in the prefill path, and
  torch's launch check then names the CUDA error.
- **`--language-model-only` left out, not replaced.** `P5.2-03` (Modal, 2 x L40S, 29 Oct) lost
  both boots in 15 s to the fork's `ValueError: --language-model-only does not support
  ['Qwen4ExpForConditionalGeneration']. Supported: ['MuseGlimmerForConditionalGeneration']`
  (`server_args.py:8496`, `_handle_language_model_only`, called from
  `_handle_encoder_disaggregation`). Alternatives: `--language-only` ("for VLM, load weights for
  the language model only", `server_help.txt`) sits in the same encoder-disaggregation handler,
  and its checks are unread (the fork's source is not in our logs), so it lost to the one setting
  dfranzen's launcher ran this model with, the vision tower loaded. Its cost: 0.836 GiB of BF16
  weights, 897,862,112 bytes (`P5.2-00`'s `model.json`), at most
  $\lfloor 897{,}862{,}112 / 6{,}912 \rfloor = 129{,}899$ FP8 KV tokens of pool if every GPU
  holds the whole tower, $\lfloor 897{,}862{,}112 / (4 \cdot 6{,}912) \rfloor = 32{,}474$ if TP 4
  shards it; the planner reads the pool at boot, so the streams follow.
- **An argument check before the boot, not only the fallback boot.** The fallback boot changes
  KV and graphs, so a refused flag fails it too (`P5.2-03`'s second boot). The fork's own
  resolution in a child process costs about a minute, runs on CPU in the free twin, and leaves no
  CUDA context behind for the server; a static list of known refusals was the alternative and
  lost because it only knows the refusals already seen. It never replaces the boot: a check that
  errs or stops at a device query on a GPU box must not cost a run the boot would have served.
- **A fake server for the smoke and the twin, not a tiny model.** The script loads no weights
  itself; what can break is the protocol (streaming, ids, cuts, the forcing continuation, tool
  calls and their continuations, the log patterns, a dying engine), which the fake exercises end
  to end on CPU in about a minute; with the model's own tokenizer in the twin.

## 4. How to run it

**On Modal first** (task P5.2m, run `P5.2-02`): the same notebook and config, three paired tasks
and a 3,900 s plan, on Modal's 4 x L4 through `scripts/modal_ops.py` (a probe job; runbook
modal_ops.md, "Probe jobs"); its CPU path is
`python scripts/modal_ops.py --smoke --config configs/modal_P5.2-02.yaml`.

### Kaggle (the orchestrator runs every gateway call)

`P5.2-00` (done 29 Oct, Pass; kept for a rerun):

```bash
python scripts/build_fn_notebooks.py --config configs/P5.2-00.yaml --out dist/P5.2-00
python scripts/kaggle_ops.py push dist/P5.2-00 --timeout 3600
python scripts/kaggle_ops.py pull alyahya737/arc26-fn-env --task-id P5.2-00
```

The FA2 source build (`configs/P5.2-00-fa2.yaml`, `--timeout 42600`) is needed only if the
prebuilt wheel fails its import; `P5.2-00`'s imported.

Version 11 (the twin first, free; then the GPU run on approval, after Sat 31 Oct 03:00):

```bash
python scripts/build_fn_notebooks.py --config configs/P5.2-01.yaml --out dist/P5.2-01
python scripts/build_fn_notebooks.py --config configs/P5.2-01-cpu.yaml --out dist/P5.2-01-cpu
python scripts/kaggle_ops.py dataset-create dist/P5.2-01/arc26-code-fn   # once; wait until processed
python scripts/kaggle_ops.py push dist/P5.2-01-cpu --timeout 5400
python scripts/kaggle_ops.py pull alyahya737/arc26-v11-cpu-check --task-id P5.2-01-cpu
python scripts/kaggle_ops.py push dist/P5.2-01 --run-id P5.2-01
python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id P5.2-01
python scripts/fn_report.py --run-logs runs/P5.2-01/run_logs/fn --config configs/P5.2-01.yaml \
    --solutions <arc-agi_evaluation_solutions.json or ARC-AGI-2 data/evaluation> --out reports/P5.2-01
```

A later code change needs `dataset-version` of `arc26-code-fn` (the twin and version 11 share
one build's bytes). Inputs attached by the metadata: `dfranzen/pennyroyal-v253`,
`alyahya737/arc26-code-fn`, the FP8PLE model, the competition, and `alyahya737/arc26-fn-env` at
version 1. The human submits the version from the website; with arm A the config edit is
`solve.arm: plain`, a new build, and a short check of its own.

### A plain GPU machine (4 x 24 GB, sm89 or newer, Python 3.12)

```bash
python scripts/fn_probe.py --config configs/P5.2-01.yaml --root /    # paths from the config
```

with the config's `paths` and `inputs` pointed at local copies of the wheelhouse, the model, and
the challenges, `install.enabled` true, and `mode` set (`development` or `competition`).

### CPU smoke

```bash
.venv/bin/python scripts/fn_env.py --config configs/P5.2-00.yaml --smoke --root runs/P5.2/tmp/s0
.venv/bin/python scripts/fn_probe.py --config configs/P5.2-01.yaml --smoke --root runs/P5.2/tmp/s1
.venv/bin/pytest tests/test_fn_env.py tests/test_fn_probe.py tests/test_fn_notebook.py tests/test_build_fn.py
```

### Time, memory, disk (estimates unless measured)

| Run | Wall | GPU memory | Host RAM | Disk |
|---|---|---|---|---|
| `P5.2-00` | 10 to 25 min: install 3 to 8, scan of about 10 GB of libraries 1 to 3, FA2 download | none | about 4 GB | venv about 15 GB in `/tmp`; output about 0.3 GB |
| `P5.2-00-fa2` | 1 to 3 h | none | about 3 GB per nvcc job, 4 jobs | about 18 GB |
| `P5.2-01-cpu` (twin) | install about 15 min (903 s measured on the CPU box), the fake's solve minutes; Timeout 5,400 s | none | about 4 GB | venv about 15 GB in `/tmp` |
| `P5.2-01` (version 11, development) | 8,850 s of notebook time at most, plus the time before the first cell and the save; Timeout 10,800 s | about 17.9 GiB weights per GPU, 0.92 of 22.49 GiB used | 47.68 GiB n-gram table plus about 12 to 16 GiB of workers, of 80.5 | venv 15 GB, caches, the engine log |
| Version 11 (scored rerun) | 40,050 s of notebook time at most | as above | as above | as above |

## 5. Outputs

`P5.2-00`, `run_logs/fn_env/` (pulled):

| File | Contents |
|---|---|
| `report.json` | every step's record, the resolved config |
| `summary.md` | one line per step; Python, imports, sm89 verdicts, model sizes |
| `input_listing.txt` | the input tree with sizes |
| `versions.txt` | `uv pip freeze` of the venv |
| `arch_files.json` | per file: cubin and PTX architectures, the sm89 path, `cuobjdump`'s list |
| `server_help.txt` | the server's `--help` |
| `patch_fa2_scratch_dtype_context.txt` | the fork's code around the edit |
| `model.json`, `drafter.json`, `*_files.json`, `*_chat_template.jinja.txt` | configs, sizes, components, templates |
| `template.json` | renders, marker ids, the note's token count |
| `fn_env_output.json` | wheels and their sha256, the Python tarball |

Kept on Kaggle (not pulled): `/kaggle/working/wheels/` (the FA2 wheel), `python312.tar.gz`.

Version 11 and its twin, `run_logs/fn/` (pulled): `notebook_setup.json` (cell 1: the code check,
the placeholder file's report), `final_check.json` (cell 3: the final file's report, the solver's
exit, the notebook's seconds), `run.json` (the resolved config, commit, mode), `state.json`
(every stage), `inputs.json`, `install.json`, `versions.txt`, `patches.json`,
`engine_commands.jsonl`, `local_inputs.json`, `arg_check.json`, `boot.json`, `server_info.json`, `memory.jsonl`, `server_metrics.jsonl`,
`template_check.json`, `solve_plan.json` (arms, streams, order), `attempts.jsonl` (per task and
arm: every request's row, tokens, the forcing, calls and the program for arm B, block counts, the
answers as grids), `transcripts.jsonl` (the replies' text), `restarts.jsonl`,
`engine_failures.jsonl`, `timing-inner.jsonl`, `decode_lines.json`, `engine.log`,
`engine_tail.log`, `summary.md`; and `submission.json` at the root (placeholders in development
mode). Locally `fn_report.py` writes `reports/P5.2-01/fn_report.md` and `fn_report.json`.

## 6. How to read the results

`P5.2-00`:

- **Good:** Python 3.12; venv installed and `pip check` clean; `flash_attn` 2.8.3 imports;
  `server_help` lists every probe flag; every package `native` or `ptx` (or `no device code` for
  the JIT ones); the edit `checked` with `q` in scope in its context file; the model about 124.75
  GB with `auto-round`, 4 bits, FP8 table, no missing shard. Then the brief goes to the page.
- **Borderline:** an unknown flag or a failed edit check: fix the config (the flag) or rely on the
  BF16 fallback (the edit), rebuild, and say so in the brief.
- **Bad:** the venv does not install, or a package the server needs has no sm89 path: no GPU run;
  report to the orchestrator.

The twin (`P5.2-01-cpu`): **Pass** when every stage is `ok`, both arms finished their 3 tasks,
every tool row has an `ok` call, and `final_check.json` is valid (the brief's first table);
anything else stops the GPU push.

`P5.2-01` (the brief's section 1 holds the registered rule; `fn_report.json` `decision`):

- **B:** $S_B \ge S_A$ with B's median wall at most 1.3 times A's and at least 90% of its
  continuations cached: the human submits this version.
- **A:** A ahead by 2 tasks, B over 1.5 times slower, or a Fault: `solve.arm: plain`, a new build,
  a short check of its own, then the submission.
- **Human:** anything else; the readings (pool, streams, decode rates, forced shares, calls,
  programs) and the external prior go to the page.
- **Coverage of a scored rerun:** `solve_plan.json` `budget.coverage` and `replan.json` give the
  tasks reached at the long budget (about 74 to 98 of 240 at a 248k pool); the expected solves
  are the budget note's (about 7 to 12 of 240 at 248k, 2.9 to 5.0 points; NVARC alone about 31).

## 7. Failure modes

| Symptom | First thing to change |
|---|---|
| OOM at load or graph capture | `--mem-fraction-static` 0.90, fewer `--cuda-graph-bs-decode` sizes, `--chunked-prefill-size` 2048 |
| OOM on a long prefill (the indexer's logits) | `--chunked-prefill-size` 2048 |
| Pool far below 0.25M | `--disable-radix-cache` with `--max-mamba-cache-size 9` (costs the forcing turn's cached prefix), or 0.94 if the boot shows headroom |
| First generation fails (`template_check`, the #36968 point) | the solver boots once more with BF16 KV and no CUDA graphs by itself; if that fails too, the FA2 import in `versions.txt` and `patches.json` |
| Degenerate attempts (the run's own stop; NaN logits or a broken kernel) | BF16 KV, then `--disable-cuda-graph`, then `--expert-parallel-size 1` |
| Boot over 2,700 s (slow model mount) | the precache rate in `state.json`: below about 100 MB/s, copy the weights to local disk first (P6.9c's route) |
| TileLang indexer compile fails | uninstall `tilelang` in the venv (the torch fallback; slow, correct) |
| Install timed out or slow (twin 6) | `local_inputs.json`: `mb_per_s`, `stopped`, `linked`; a copy near 0 MB/s is the mount itself; files linked to the mount were read by uv there |
| Unknown server flag | `server_help.txt` from `P5.2-00`; remove or rename the flag in `engine.flags` |
| No health, the log's last line in the graph capture or a `multimem all-gather disabled (wait timeout ...)` | `patches.json`: `logits_gather_nccl` must read `applied`; then `NCCL_P2P_DISABLE=1` in `engine.env` (the fork's hint) |
| `RuntimeError: CUDA error: ...` in `torch.sum` from `fused_marlin_moe.py` (after the `marlin_moe_torch_sum` edit) | an earlier kernel's launch failed; rerun that boot with `CUDA_LAUNCH_BLOCKING=1` in `engine.env` (a diagnostic run, slow) to see which |
| `arg_check` reads `refused` (a flag the fork refuses for this model, as `--language-model-only` in `P5.2-03`) | `arg_check.json` names the flag set, the error, and `stopped_in`; remove or change that flag in `engine.flags` and rebuild |
| `arg_check` reads `device_stop` in the twin | expected on CPU: `stopped_in` shows how far the resolution got; the GPU run checks the rest before its boot |
| `arg_check` reads `error` | the child's `output_tail` in `arg_check.json`; the boot ran anyway, its log decides |
| The edit does not apply | the fallback runs BF16 KV by itself; `patch_*_context.txt` shows why |
| The engine dies mid-run | the health watch (or the stream that saw it) restarts it, twice at most; an attempt that ran across the restart is queued again once, never forced on the new server; `restarts.jsonl` and the engine log's tail show why |
| A run cut by the Timeout | `fn_report.py` reads Fault (no finished solve stage): partial rows decide nothing |
| The tool self-test fails | the tool arm is dropped (competition runs plain); `state.json` `tool_self_test` holds the call rows |
| Timeout | the solver stops itself at the mode's `end_s` (8,850 s in development, 40,050 s in competition); cell 3 always runs; Kaggle's Timeout (10,800 s) or 12 hours is the outer bound |
| Requests refused at the context edge | every request keeps a 128-token margin; a refusal shows as finish `error` with SGLang's message in the turn row |
| The thinking floor not met (`solve_plan.json` `budget.met` false) | the pool is too small even for one stream at the floor: one stream at 86,016 runs; MTP or a smaller opening bound would be the next lever |
| NaN | not observable as numbers here; it shows as degenerate output, above |

## P5.6-01: the batch sweep and the 8B boot ladder

Run `P5.6-01` (task P5.6, brief [`docs/prerun/P5.6-01.md`](../prerun/P5.6-01.md), config
`configs/P5.6-01.yaml` extending `configs/P5.2-01.yaml`, twin `configs/P5.6-01-cpu.yaml`, tests
`tests/test_p5601.py`) adds two things to `fn_probe.py`:

- **`stage_sweep`** (between the template check and the solve, in the modes `sweep.modes`
  names): at each size $n$ of `sweep.batches`, $n$ requests with one short prompt run at once to
  `sweep.max_new` tokens with `ignore_eos`; `sweep.json` holds per size the requests' rows, their
  median rate, the aggregate $A(n) = \sum_i (k_i - 1) / (\max_i e_i - \min_i f_i)$, the step time
  $1000\,n / A(n)$ ms, and the server's `gen throughput` lines in the window. A size above the
  server's running limit, or within `sweep.stop_before_s` of the generation stop, is skipped and
  named.
- **A rung's `null`** removes a flag (as before) or now a variable (`boot_rungs` keeps the null,
  `boot_environment` drops it), so a rung can return to an earlier revision's environment exactly.

| Choice | Alternative considered | Why it lost |
|---|---|---|
| The sweep inside the dev run, before the solve | A separate notebook | A second session pays the setup and the boot (about 15 minutes, 0.5 quota hours) again |
| The sweep's shared short prompt | Prompts at the solve's contexts | It measures the step's growth with the batch, which the 16-stream question needs (`fn_900_toks.md`); long contexts would add prefill to the window |
| Rung 1 as revision 20's flags and variables by `null` | A fresh rung list without 8B's keys | Cumulative rungs keep the ladder's one mechanism; the test checks rung 1 equals revision 20 exactly |
| Its own notebook and code dataset | Revision 20's | `~/wt-P5.5`'s `dist/` and `arc26-code-fn-kv4` belong to revision 20's approval; one code dataset per concurrent build lineage |

### P5.6-02: 7 streams on revision 21

Run `P5.6-02` (brief [`docs/prerun/P5.6-02.md`](../prerun/P5.6-02.md), config
`configs/P5.6-02.yaml` extending revision 21's `configs/P5.2-01.yaml`, twin
`configs/P5.6-02-cpu.yaml`, tests `tests/test_p5602.py`): P5.6-01's levers on revision 21
(`task/P5.9` merged), the fraction back to 0.95, 8 server slots for 7 streams, Mamba
$24 = 3 \times 8$, graphs `[1, 2, 4, 6, 7]`, the chunk 1,024 with at most 4 requests per prefill
batch (`--prefill-max-requests`), mixed chunks off, the sweep at 1 2 4 6 7 8, rung 1 revision 21.
Three code changes, from P5.6-01's forensics (`docs/experiments/P5.6-01-forensics.md`):

- **`stage_stress`** (after the template check, before the sweep and the solve, in
  `stress.modes`): the $n$ largest openings of the plan's tasks ($n = \min$(`solve.max_streams`,
  the server's limit)) at once, each a distinct prompt, then `stress.max_new` tokens of decode
  each with `ignore_eos`; `stress.json` holds the lengths, each request, the seconds, the
  engine's free device memory in the window, and the peak. An error, a short reply, an
  out-of-memory line, or an engine that stops answering fails it, and the boot moves one rung
  down the ladder (`boot_stress_retry`, once) before any task starts.
- **The engine instance** (`Inner.generation`, counted by `launch_server`): `restart_engine`
  takes the instance the caller saw fail and neither probes nor kills a later one; a
  `HealthWatch` belongs to one instance and exits when another starts (the restart starts the
  new instance's watch). A kill needs `health.kill_probes` failed probes in a row of
  `health.probe_timeout_s` each (`engine_responsive`; 3 at 30 s here, 1 at 10 s when unset).
- **`free_device_min`**: the engine's `free device mem: X GiB` lines (`engine.free_pattern`);
  `state.json` `monitor.free_device` holds the minimum while serving.
- **P5.2-01 revision 22** (`configs/P5.2-01.yaml`, `tests/test_r22.py`) takes the health probe and
  the stress check with revision 21's engine unchanged; its `stress.fallback_flags`
  (`--chunked-prefill-size 1024`, `--prefill-max-requests 4`) apply only if the check fails: the
  server reboots with them (`boot_stress_fallback`) and checks once more (`stress_fallback`),
  and a second failure moves the boot one rung (`boot_stress_retry`). The argument check
  resolves the fallback set as its own variant (`stress_fallback`). P5.6-02 sets
  `fallback_flags: null`: it runs the smaller prefill already, so a failure goes to rung 1.

| Choice | Alternative considered | Why it lost |
|---|---|---|
| 0.95 | P5.6-01's 0.96 | P5.6-01 had two CUDA out-of-memory restarts at a 40 MiB allocation with 32.5 MiB free (peak 22,541 MiB); version 17 at 0.95 peaked at 21,911 |
| 7 streams on 8 server slots | 8 on 9 | Revision 21's N+1 rule with the margin back: 8 streams on 9 slots need Mamba 27 and more decode rows at the margin that failed |
| Mamba 24 ($3 \times 8$) | P5.6-01's 29 ($3 \times 9 + 2$) | The fork caps the limit at $\lfloor K/3 \rfloor$ and takes slots per admitted request; 7 streams lock about 14 (P5.6-01: 2 per running request), so 10 stay free for cached states, and 5 slots (17,410 KV4 tokens) go to the pool |
| Graphs `[1, 2, 4, 6, 7]` | `[1, ..., 7]` | Five sizes keep P5.6-01's capture time (766 s boot); batches between sizes pad to the next |
| Chunk 1,024 and at most 4 requests per prefill batch | The 2,048 chunk with no batch limit | The failing allocation (`fn_p56_suspects.md` section 3) was the PLE short conv's eager `F.silu`, whose buffer pads to requests x the longest extend x channels: 40 MiB at 8 requests of 2,048 |
| A stress check at boot | Trusting the sweep | The sweep's short prompts left the long-prefill kernels unloaded; the fault came at the solve's first long prefill, which a ladder rung can no longer catch |
| A kill after 3 probes of 30 s, per instance | One 10 s probe | P5.6-01's restart 2 killed a stalled, possibly live engine on one stale 10 s probe |
