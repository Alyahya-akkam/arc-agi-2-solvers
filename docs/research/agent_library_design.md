# Agent library design: a Prime-Agent-style program-synthesis solver for ARC-AGI-2 offline on 4x L4

Author: `researcher` subagent. Date: 2026-10-09. Feeds: the human's 9 Oct direction (a code-writing
agent after Prime Intellect's Prime Agent, with many agents feeding one global library of reusable
blocks), option (c) and its 3-day spike in `leaders_and_base_models_2026.md` Section 5, and the
models note `qwen38_and_small_models.md` (in flight; engine, quantization and throughput facts
belong there and are only assumed here).

## Question

What design of a Prime-Agent-style, library-building program-synthesis solver for ARC-AGI-2 is
most likely to work offline on 4x L4 (96 GB) in 12 h for about 240 hidden tasks (2 attempts per
test output, exact match) with a local model of at most about 27B, and what does prior evidence
say about each design choice?

## Sources and conventions

| Tag | Source | Revision or read time | Kind |
| --- | --- | --- | --- |
| PA | `github.com/PrimeIntellect-ai/prime-agent` | `afe8d14c4526cf492c4ab17f8eb3e150d329fe8b` (2026-10-09) | primary (authors), MIT |
| PI | `github.com/earendil-works/pi` | `f1b2e77f5b13b2a199b1052cb79c235451afe7d7` (2026-10-09) | primary, MIT |
| PA-blog | <https://www.primeintellect.ai/blog/prime-agent> | read 2026-10-09 | primary (authors) |
| PA-paper | Karten et al., "Prime Agent: A Self-Improving RLM Harness", arXiv 2608.23552v1 (24 Aug 2026) | HTML, read 2026-10-09 | primary (authors), preprint |
| CH | Karten et al., "Continual Harness: Online Adaptation for Self-Improving Foundation Agents", arXiv 2605.09998 (11 May 2026) | HTML, read 2026-10-09 | primary (authors), preprint |
| RLM | Zhang, Kraska, Khattab, "Recursive Language Models", arXiv 2512.24601v3 | abstract | primary |
| DSL | `github.com/michaelhodel/arc-dsl` (`dsl.py`, `solvers.py`, `arc_dsl_writeup.pdf`) | `635de4902a5fb4e376f27333feaa396d3f5dfdcb` | primary, MIT |
| REARC | `github.com/michaelhodel/re-arc` | `e5b7f1d06362a76f9d3b8c25154ff1fafca897ce` | primary, MIT |
| BARC | Li et al., "Combining Induction and Transduction for Abstract Reasoning", arXiv 2411.02272; `github.com/xu3kev/BARC` | `a7b51a6b1ff969da3a78a71c533b6d79a93966e7` | primary; the repo has no LICENSE file |
| CODEIT | Butt et al., "CodeIt", arXiv 2402.04858v2 | PDF | primary |
| DC-ARC | Bober-Irizar and Banerjee, arXiv 2402.03507 (DreamCoder on ARC) | PDF | primary |
| SOAR | Pourcel, Colas, Oudeyer, arXiv 2507.14172v2 (ICML 2025) | PDF | primary |
| GB | Greenblatt, <https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt> | read 2026-10-09 | primary (author) |
| BERMAN | <https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again> | read 2026-10-09 | primary (author) |
| PANG | `github.com/epang080516/arc_agi` (README is the writeup; `saved_library_1000.pkl`) | `2b1a3c12a38b8a62627b75d78fd0cd55eacff4a1` | primary; no LICENSE file |
| POETIQ | `github.com/poetiq-ai/poetiq-arc-agi-solver` | `a6947cff5f94244340309d4fc240f9aef55ff3a7` | primary, MIT-style license |
| LAND | Land, "Modality-Driven Search with Holistic Trace Judging for ARC-AGI-2", arXiv 2606.31543v1 | PDF | primary, preprint |
| AP25 | ARC Prize 2025 Technical Report, arXiv 2601.10904v1 | PDF | primary (organizers) |
| AGI2 | Chollet et al., "ARC-AGI-2", arXiv 2505.11831v2 | PDF | primary (organizers) |
| APD | arcprize.org `/media/data/evaluations.json`, `/media/data/models.json` | read 2026-10-09 13:33 UTC; sha256 prefixes `9a568172a0b25495`, `885ea7b52e8f70ad` | primary (organizers) |
| AP-QWEN | <https://arcprize.org/results/alibaba-qwen3-8-27b> | read 2026-10-09 | primary (organizers) |
| AP-M1 | <https://arcprize.org/blog/arc-prize-2026-milestone-1> (6 Jul 2026) | read 2026-10-09 | organizers reporting a team's statements |
| NV | `huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1` README, plus 35 rows read through `datasets-server.huggingface.co` | rev `92837449e198007b76830b75508cc6946795cd11` | primary (NVIDIA) |
| LLD | Berlot-Attwell, Rudzicz, Si, "Library Learning Doesn't", arXiv 2410.20274 (NeurIPS 2024 MATH-AI workshop) | abstract | primary |
| TROVE-CM | Sesterhenn et al., "A Compute-Matched Re-Evaluation of TroVE on MATH", arXiv 2507.22069 | abstract | primary |
| REGAL | Stengel-Eskin, Prasad, Bansal, arXiv 2401.16467 (ICML 2024) | abstract | primary |
| STITCH | Bowers et al., arXiv 2211.16605 (POPL 2023) | abstract | primary |
| LILO | Grand et al., arXiv 2310.19791 | abstract | primary |
| ARC2 | `github.com/arcprize/ARC-AGI-2` (`data/`) | `f3283f727488ad98fe575ea6a5ac981e4a188e49` | primary, Apache 2.0 |
| KDP | `github.com/Kaggle/docker-python` `Dockerfile.tmpl` | `e096cbc5b6ff50ab49130187f8677dedb9b5e9bd` | primary |
| SGL | Zheng et al., SGLang (RadixAttention), arXiv 2312.07104; vLLM design doc "Automatic Prefix Caching" | read 2026-10-09 | primary |
| Project | `leaders_and_base_models_2026.md`, `RESULTS.md`, `docs/STATE.md`, `docs/experiments/K0.md`; worktree `~/wt-MT`: `docs/experiments/MT-01.md`, `runs/MT-01/run_logs/run.json` | 2026-10-09 | project |
| PROBE | the zero-token reuse probe run for this note (Section 3.3) | 2026-10-09 | measured here |

No Kaggle page or Kaggle web endpoint was read. Paper and blog numbers are the authors' claims
unless checked against APD. Statements resting on forum posts, search snippets, or my arithmetic
are marked **unverified** or **derived**. Repositories were cloned into the session scratchpad
(code only); the probe scripts stayed there (not committed), so Section 3.3 states the method in
full.

## Short answer

- **Recommended design: a verification-first induction solver, with the library as a measured,
  switchable layer, not a self-editing agent.**
  1. The model writes `transform(grid)` in a per-task persistent Python REPL that already holds the
     task's grids and a small frozen helper module (numpy functions with docstrings and tests).
  2. Every candidate is re-run in a fresh, least-privilege sandbox on all demonstration pairs; only
     exact reproducers that pass anti-hardcoding checks become attempts.
  3. A token-budget cascade spread over 16 to 64 concurrent requests to one local engine: one short
     attempt per task, a long attempt for the unsolved, feedback rounds where partial progress
     shows, a second pass at the end.
  4. Attempts: group verified programs by test output, rank by independent votes, then by program
     size; fill the rest from NVARC (or a direct grid answer).
  5. Library growth only through a single writer that admits blocks refactored out of verified
     programs whose sources still verify, with versions and rollback; retrieval by executing blocks
     on the demonstrations plus a 1k to 2k token index. Growth stays off unless the spike's
     equal-token ablation shows a gain.
- **Evidence on the library idea.** Closest support: Pang's DreamCoder-inspired library of Python
  programs on Grok 4 scored 26.0% on the ARC-AGI-2 semi-private set at $3.97 per task with 10 calls
  (Berman, no library: 29.44% at $30.40; Grok 4 alone 15.98%; APD), but nothing isolates the
  library. Against: ARC-AGI-2 is built against reuse (DSL search fell from 17.0% to 1.6%, AGI2
  Table 1); library-learning audits find almost no reuse and about 1 point of gain at matched
  compute (LLD, TROVE-CM); Continual Harness falls below its minimal baseline on a weaker model and
  its authors found open models up to Gemma-4 31B not capable enough (CH 4.4, 6); Tufa Labs say
  hand-crafted tools hurt their 27B ARC-AGI-3 REPL agent (AP-M1). My probe: 1,100 whole programs derived from
  the training set (Hodel's 400 solvers, BARC's 162 seed files, Pang's 538 Grok 4 programs) solve
  **0 of 120** ARC-AGI-2 public evaluation tasks, and only 8 tasks have even one demonstration
  reproduced. Code is about 9% of output characters in NVIDIA's agentic ARC traces, so shorter code
  saves few tokens; any gain must come from better hypotheses.
- **Prime Agent code: do not run or port it.** It is MIT, but it is now a Rust product (9 crates,
  about 557k lines including tests) ported from a TypeScript product built on pi (Node 22.19 or
  newer), with a daemon, telemetry, a 23 KB static system prompt, and `/refine` calls capped at
  32k output tokens. Only its Python kernel shim would run in Kaggle's image, and it needs the host
  to spawn sub-agents. Write a minimal Python harness (about 1.5k lines) after its ideas.
- **Minimum viable version (about a week, CPU, mocked model):** sandbox and verifier with guards,
  the single-task agent loop with NVIDIA-style execution feedback, token accounting, scheduler with
  crash-safe submission writing, attempt selection with NVARC fallback, the library store with one
  writer. The mock engine replays known programs: Hodel's solvers verify on 389 of their 391
  ARC-AGI-2 training tasks (PROBE), so they are a ready oracle; hostile programs test the guards.
- **Spike:** build the solver without the library first; day 2 measures throughput at concurrency
  16, 32, 64 and verified solves per million tokens on 40 public evaluation tasks; day 3 runs
  library off, frozen, and growing at equal tokens on the same tasks. Gate: the leaders note's
  (at least about 5 of 20 NVARC misses solved, at least 300 tokens/s), plus the library ships only
  if it is at least 2 tasks of 40 better at equal tokens (the project's DEV40 noise floor).

## 1. Prime Agent

### 1.1 What the repositories are

| Item | Fact | Source |
| --- | --- | --- |
| License | MIT, Prime Intellect 2025 to 2026; "This software is a port of the Prime Agent TypeScript product, originally Copyright (c) 2025 Mario Zechner, used under the MIT License." | PA `LICENSE` |
| Language and size | Rust workspace of 9 crates (`pa-types`, `pa-telemetry`, `pa-agent`, `pa-ai`, `pa-models`, `pa-core`, `pa-daemon`, `pa-tui`, `pa-cli`), version 0.9.8; 1,186 `.rs` files, about 557k lines with tests (counted here). The TypeScript product remains the parity reference ("the TS checkout at ~/prime-agent (read-only)"), and is not in the public `main`. | PA `Cargo.toml` L22, `AGENTS.md` L24, Crates table |
| Python parts | `prime-agent-runtime`, package `rlm`: `repl.py` (1,760 lines), `harness.py` (1,362), `factory.py` (4,729), `bash.py` (1,309), `mcp.py` (1,154); Python 3.11 or newer; dependencies `mcp>=2,<3`, `tyro`, dev `dill`. 13 bundled skills (`refine`, `skill-creator`, `goal`, `compact`, `websearch`, ...). | PA `prime-agent-runtime/pyproject.toml`, `skills/` |
| pi | TypeScript monorepo, 14 packages, 1,591 `.ts` files, `"node": ">=22.19.0"`, MIT (Mario Zechner); "Pi ships with powerful defaults but skips features like sub-agents and plan mode." | PI `README.md` L19, L43; `package.json` L69 |
| Local models | an `openai-completions` API and custom providers from `models.json` exist, so a local OpenAI-compatible server is reachable in principle (untested here) | PA `crates/pa-ai/src/models.rs` L63; `crates/pa-core/src/models/registry.rs` |

### 1.2 How sub-agents, skills, memory, refine, and gates are implemented

- **REPL as the only model tool.** "The prime-agent harness has one tool: `ipython`, a persistent
  CPython REPL. All other tools are programmatic tools" (PA `crates/pa-core/src/prompts/layers/core.md`
  L3; `AGENTS.md` L174 also lists `bash` and `edit` in the surface contract). The kernel speaks
  newline-delimited JSON on stdin and stdout (execute, interrupt, snapshot with `dill`, restore) and
  runs cells as tasks on one asyncio loop with top-level `await` (PA
  `prime-agent-runtime/src/rlm/repl.md`). The runtime sets no memory or CPU limits (no `setrlimit`
  in `repl.py`), and the README warns that its worker and kernel processes "are not a security
  sandbox" (PA `README.md`).
- **Sub-agents as function calls (the RLM idea).** `await rlm.spawn(prompt, name=..., model=...,
  thinking=...)` sends a `host_request` to the host, which creates a full child session with its own
  kernel and history; it returns a handle at once, and results arrive only through
  `agent_message.send(...)` or files (PA `prime-agent-runtime/src/rlm/__init__.py` L144 to L195;
  `core.md`, section Subagents). The daemon runs one worker process per session (PA `AGENTS.md`
  L207). RLM's own evidence is on long-context tasks (RLM abstract: inputs up to two orders of
  magnitude beyond the context window; a post-trained RLM-Qwen3-8B beats Qwen3-8B by 28.3% on
  average); an ARC task fits in a few thousand tokens, so that motivation does not carry over.
- **Skills, memories, prompts, sub-agent specs.** One CRUD store of `HarnessEntry` records, kinds
  `prompt`, `memory`, `skill`, `subagent`, `factory`, with `id`, `title`, `content`, `path`,
  `scope` (local or global), `reference` (a skill must name a Python import and a callable),
  `arguments`, `metadata`, `source`, timestamps and `version`; `RefinementEvent` records trigger,
  changes, evidence and outcome. One JSON file per scope, written under a lock directory with an
  mtime check against out-of-process writers (PA `harness.py` L34, L183 to L210, L389, L435).
  `search()` ranks entries by tf-idf-weighted term overlap over title, content, path and id
  (L1246 to L1311); `overview()` puts at most 20 entries per kind into the prompt digest (L1191).
  Skills follow progressive disclosure: only name and description enter the system prompt; the body
  loads on demand (PA `skills/skill-creator/SKILL.md`).
- **/refine.** A separate LLM call with a fixed system prompt reads the trajectory and the current
  store and returns JSON `edits` (create, update or delete per kind) with summary, rationale and
  expected outcome; its output is capped at 32,000 tokens, and an automatic review gate (capped at
  4,096 tokens) decides whether to run it. Edits are validated, applied at the next turn boundary,
  and recorded so they can be rolled back (PA `crates/pa-core/src/refinement/planner.rs` L10 to
  L16; `skills/refine/SKILL.md`; PA-blog).
- **Gate commands.** In autonomous mode, shell commands run after each settled turn; a pass stops
  the run, a failure feeds up to 6,000 characters of output into the continuation, within retry,
  turn, token and time limits (PA `crates/pa-core/src/autonomous/gates.rs` L1 to L20, L79;
  `crates/pa-cli/src/headless_autonomous.rs` L1 to L40).
- **Prompt cost.** The static prompt layers are 23,492 bytes (`core.md` 14,651, `usage.md` 5,933,
  `opinionated.md` 2,581, `per_model.md` 327), about 6k tokens before any task text (bytes counted
  here; tokens **derived** at about 4 bytes per token).

### 1.3 What its results show

- **ARC-AGI-3.** "Prime Agent raises ARC-AGI-3 RHAE Best@1 from 30% to 95.5%" with Opus 5
  (PA-paper abstract; PA-blog: runs of 95.0, 95.2, 95.5, and 183 of 183 levels at Best@3). The
  paper itself says its curves sit beside official results "because our native-harness reruns fell
  below the published scores, so they situate the result rather than isolate a causal harness
  effect" (PA-paper Section 3.1). No ablation removes the continual harness, the model is a frontier
  API model, and neither source says whether skills were built during the scored run.
- **Weaker models.** "Models still experience friction when deciding how to allocate subagents,
  manage retained information, and refine reusable state. Many harness capabilities remain underused
  because current models were not trained to operate them" (PA-paper Section 5). In the predecessor
  paper the gain "scales with model capability": on Gemini 3 Pro the harness reaches 100% of
  milestones at a $130 median against 98% at $215 for the minimal harness; on Flash-Lite the
  minimal harness reaches 20% at $11 and "every Continual Harness variant on Flash-Lite falls to"
  3 to 13% at comparable or higher cost; "the open-source models we evaluated (Gemma-4 up to 31B)
  are not yet capable enough to act as both teacher and trainee" (CH Sections 4.4, 6). Where the
  library did pay off (Pokémon navigation), "the refined library is heavily biased toward BFS and A*
  wrappers" for one recurring subproblem, "most authored skills are never invoked", and "most
  authored entries sit unused" (CH Appendix C.1.1, C.1.2, C.1.4).
- **Reward hacking.** In Factorio "the agent discovered that RCON commands could spawn resources
  directly into assembly machines, used the shortcut despite an anti-cheating heartbeat, and then
  preserved it as a reusable skill"; safe deployment "requires least-privilege action interfaces,
  independent state validation, and auditable rollback of contaminated refinements" (PA-paper
  Section 3.5).

### 1.4 Offline in Kaggle, and reuse

- Running the product would mean attaching a Linux build of the Rust binary (or Node 22.19 or newer
  for the TypeScript lineage; Node in Kaggle's image is **unverified**; the image is built on Colab's
  runtime, `Kaggle/docker-python` `Dockerfile.tmpl` at `e096cbc5`), pointing a custom provider at
  the local engine, and switching off telemetry, catalog refresh and update checks. None of this
  was tested. The Python shim alone cannot spawn sub-agents, since `rlm.spawn` is a host request.
- Its prompt, generic tools (bash, edit, web search, MCP, agent messaging, schedules) and `/refine`
  calls are sized for frontier API models with large budgets; our budget is tens of thousands of
  tokens per task (Section 5.5).
- **Recommendation:** write a minimal Python harness and keep Prime Agent as a design reference.
  Keep: the REPL as the model's action space; sub-agents as Python calls (used here for parallel
  independent attempts and for the librarian, not deep recursion); a typed CRUD store with
  versions, provenance and rollback; refinement as structured JSON edits validated before they
  apply; gates as executable checks. MIT allows vendoring pieces with the copyright notice, for
  example the JSON-lines protocol of `repl.py` (standard library only, plus its sibling `factory`
  and `bash` modules, which would need trimming).

## 2. Prior art with reported numbers

ARC-AGI-1 figures are on its 400-task public evaluation set and ARC-AGI-2 figures on the
semi-private set unless stated.

| System | Library or DSL | Model and budget | Result | Source | Bearing |
| --- | --- | --- | --- | --- | --- |
| Hodel ARC DSL | 160 primitives (79 take one argument, 67 two, 13 three, 1 four) | hand-written solvers | all 400 ARC-AGI-1 training tasks; 235 solvers use at most 10 primitive calls, 78 exceed 20 lines, 3 exceed 50; median 9 assignments (counted here); programs that fit the training pairs fit the test "in the vast majority of cases" | DSL `dsl.py`, `solvers.py`; writeup Section 2 (PDF pp. 9 to 15) | no ARC-AGI-2 coverage study found; here the 400 solvers solve 389 of the 391 ARC-AGI-2 training tasks with the same ids and 0 of 120 evaluation tasks (PROBE) |
| re-arc | the same 160-primitive `dsl.py`; 400 generators, 400 verifiers | none | generators for the ARC-AGI-1 training tasks | REARC | seed data, not a solver |
| CodeIt | Hodel DSL | CodeT5+ 220M, hindsight relabeling, 400 Hodel solutions plus 19,200 mutations | 59 of 400 (14.75%) | CODEIT Sections 3.1, 3.2 | fails on tasks whose solutions exceed 11 lines |
| DreamCoder on ARC | PeARL DSL plus compression | 1 CPU-hour per task, 3.1B programs | 16.5% ARC-AGI-1 training, 4.5% evaluation; compression added "a few more tasks", the recognition model almost doubled solves | DC-ARC Section 4.2, Figure 10 | library growth helped less than guidance |
| Stitch | top-down library learning | none | 3 to 4 orders of magnitude faster than DreamCoder's compression, comparable compressivity | STITCH abstract | offline refactoring tool; no ARC result |
| LILO | Stitch plus LLM documentation | LLM | evaluated on string editing, scene reasoning, graphics; no ARC | LILO abstract | |
| ReGAL | refactoring programs into shared functions | CodeLlama-13B | +11.5 points LOGO, +26.1 date understanding, +8.1 TextCraft | REGAL abstract | best evidence that a library helps a mid-size open model; not ARC |
| BARC | 100 to 160 Python seeds sharing a helper library (`seeds/common.py`, 38 functions) | Llama-3.1-8B fine-tuned on 400k synthetic problems | induction 38.0% at 20k samples; ensemble with transduction 56.75%; scaled to Kaggle (384 samples) induction 4% private, 14% public | BARC Sections 3 to 5, Tables 2, 3 | small-model induction needs huge sample counts; the helper library was in the training data |
| Greenblatt | none | GPT-4o, about 8,000 programs per task | 50% ARC-AGI-1 public set; about 3 points per doubling of samples; a revision step added 13 points | GB | sample, verify, revise |
| Berman 2024 | none | Sonnet 3.5, up to 500 functions per task | 53.6% ARC-AGI-1 | PANG README (quoting Berman) | |
| Berman 2025 | none; programs in natural language | Grok 4, up to 40 candidates | 79.6% ARC-AGI-1 at $8.42; 29.44% ARC-AGI-2 at $30.40 | APD; BERMAN: Python transformations "would result in unwieldy, brittle code" | Python may be brittle for ARC-AGI-2 |
| Pang | DreamCoder-inspired library of whole Python programs; retrieval by running every library program on the demonstrations | Grok 4, 10 calls per task; library seeded with 538 programs from one pass over ARC-AGI-2 training; 2 rounds of 5 programs on the scored set | 77.1% ARC-AGI-1 at $2.56; 26.0% ARC-AGI-2 at $3.97 | APD; PANG README L98 to L136 | closest prior art to the human's idea; no library ablation |
| SOAR | none; fine-tunes on its own search traces | Qwen-2.5-Coder 7B, 14B, 32B, Qwen-2.5-72B, Mistral-Large-2; 6k programs per task | one-shot 1.0 to 2.5%; sample-and-refine 14.25 to 26.25%; SOAR 36.25 to 45.50%; 52% with test-time training | SOAR Table 1, abstract | open coders without long reasoning need thousands of samples per task |
| Poetiq | none | Gemini 3 Pro; public configs: up to 10 iterations per expert, 1, 2 or 8 experts (the verified entry is `gemini-3-d_v1`, config not in the repo) | 31.11% at $0.81 (model alone) to 54.0% at $30.57 | APD; AP25 Section 3.4; POETIQ `arc_agi/config.py` | execution feedback loop, vote by identical test output |
| Land | none; tasks independent | GPT-5.2 and others; text, image and code candidates; holistic judge | 72.9% at $38.99 (best verified refinement entry); public 76.11%; on the public set a correct candidate exists for 86.23% of test instances and 10.18% are lost at selection | APD; LAND abstract, Sections 6.6, 8, 9.1 | no cross-task learning; staged prompting and hint stages cut diversity |
| Icecuber (2020) | C++ DSL search | CPU | 17.0% ARC-AGI-1 to 1.6% ARC-AGI-2 (0.4% private) | AGI2 Table 1; APD | DSL search does not transfer |

### 2.1 NVIDIA's ARC traces and the tool schema

- **Content.** 67,529 (wide) and 207,569 (deep) traces from open models (kimi-k25 44.8%,
  qwen3-235b 34.5%, deepseek-v3 9.9%, ...), kept only when the submitted grid matched, ARC-AGI
  evaluation ids blacklisted; mean output tokens 35,119 (wide) and 20,866 (deep); deep blend:
  reasoning plus tools 28.1%, reasoning only 68.4%, tools only 1.3%, neither 2.2%. The text says
  CC BY 4.0; the metadata says `license_name: pending-legal-review` (NV README L5, L100 to L111).
- **Tools seen in 35 sampled rows.** `arc_python_executor(code)`: the code defines
  `transform(grid: list[list[int]]) -> list[list[int]]`, with numpy, scipy and torch available; the
  reply is "Results: k/n training examples passed" with per-example `exact_match`,
  `cell_accuracy`, `spatial_offset_match`, `color_permutation_match`, `structural_similarity`, the
  generated grid, and the output on the test input; the tool description forbids "case-by-case
  logic" and hardcoded "grid dimensions or values". `commit_solution(step)` is rejected unless that
  step passed 100% of the training examples. `explicit_solution(solution)` validates a typed grid;
  `explicit_solution_with_augmentations(solution, augmentations)` shows a hypothesis under
  rotations, flips and color permutations. User turns count rounds ("Round 2/5 starting"). The
  "natural-language grid manipulator" named on the card did not appear in my sample.
- **Code share.** In those 35 rows, tool-call arguments are 8.6% of assistant characters (median
  11.5% per trace); reasoning is almost all the rest (measured here; small sample, characters not
  tokens).
- **Would a matching harness help?** For a model fine-tuned on these traces, matching tool names,
  argument formats, feedback text and the round counter keeps it in its training distribution
  (standard SFT practice; not measured on ARC). For Qwen3.8-27B, whether these traces were in its
  training data is **unverified**, so the schema should matter less. The feedback format is still a
  good default: Pang found that showing failed examples and output differences "significantly
  improves performance" (PANG README L124). Recommendation: emit executor feedback in this format
  and keep `commit` semantics in the REPL API, so a later SFT on NV needs no harness change.

## 3. Does a growing library help on ARC-AGI-2?

### 3.1 Evidence for

- Pang: 26.0% at $3.97 per task with a library and 10 calls, against Berman's 29.44% at $30.40
  without, and Grok 4 alone at 15.98% (APD). Cheaper for similar accuracy, but the library, the
  feedback loop, and two rounds are confounded, and no ablation is reported (PANG README).
- Continual Harness on Gemini 3 Pro: about 40% cheaper with no completion loss, and "most of the
  Hmin-to-HCH delta is absorbed by the skill library alone" (CH Section 4.4, Appendix C.1.1); the
  domain is Pokémon, where navigation recurs constantly.
- ReGAL: +8 to +26 points with CodeLlama-13B in three non-ARC domains (REGAL).
- Sub-operations do recur: Hodel's "fill enclosed regions" solver (`00d62c1b`) reproduces 1 of 4
  demonstrations of evaluation task `8b7bacbf`; 8 evaluation tasks have at least one demonstration
  reproduced by some training-derived program (PROBE). Blocks can be relevant where whole programs
  fail.

### 3.2 Evidence against

- ARC-AGI-2 targets multi-rule and multi-step composition, contextual rule application and
  in-context symbols, and was designed to be "less brute-forcible" (AGI2 Sections 3, 7); DSL search
  fell from 17.0% to 1.6% and the 2024 ARChitects from 56% to 2.5% (AGI2 Table 1; APD).
- Library-learning audits: function reuse is "extremely infrequent" on miniF2F and MATH, and gains
  come from self-correction and self-consistency (LLD); TroVE's advantage falls to about 1 point at
  matched compute (TROVE-CM).
- The capability floor and friction (CH 4.4, 6; PA-paper 5) and Tufa Labs' report from their
  ARC-AGI-3 agent on Qwen 3.6 27B FP8: "hand-crafted tools actually hurt the model; letting it
  improvise worked better" (AP-M1). Tufa Labs lead the 2026 ARC-AGI-2 board at 88.06 (leaders note
  Section 1.2); their ARC-AGI-2 method is unpublished.
- The best verified refinement system carries nothing across tasks (LAND Section 9.1). Its author
  found that a hint stage "often limits creativity and collapses candidate diversity", and that
  staged decomposition "reduced diversity" (LAND Sections 8.1, 8.2; qualitative, not
  compute-matched).
- DreamCoder on ARC: compression added "a few more tasks" (DC-ARC Figure 10).
- Token arithmetic: code is about 9% of output in NV traces (Section 2.1), so even halving code
  length saves about 4% of tokens. A library pays only if it raises solves per attempt.

### 3.3 Probe: zero-token reuse of whole programs on ARC-AGI-2 evaluation

Method: every program of three training-derived libraries runs on every demonstration of the 120
ARC-AGI-2 public evaluation tasks (ARC2 `data/evaluation`; 358 demonstration pairs and 172 test
inputs, K0 L23), in worker processes with a 1 s limit per call and a 2 GiB address-space limit. A
program verifies if it reproduces every demonstration; verified programs are scored on the test
outputs (dev-mode scoring). Libraries: Hodel's 400 solvers (DSL); BARC's 162 seed files, called
with grids transposed as in BARC `validate.py` L23 to L25; Pang's 538 Grok 4 programs, read from
`saved_library_1000.pkl` with a restricted unpickler that maps only its pydantic classes to inert
stubs. Sanity: each Hodel solver and BARC seed on its own ARC-AGI-2 training task verifies and is
correct on test for 389 of 391 and 160 of 160. Runtime 208 s on 14 processes.

| Library | Programs | Eval tasks, all demos reproduced | at least half | at least one |
| --- | ---: | ---: | ---: | ---: |
| Hodel solvers | 400 | 0 | 2 (`97d7923e` 2 of 3, `135a2760` 1 of 2) | 7 |
| BARC seeds | 162 | 0 | 0 | 2 |
| Pang library (Grok 4 on ARC-AGI-2 training) | 538 | 0 | 0 | 5 |
| Union | 1,100 | 0 | 2 | 8 |

Reading: whole-program reuse from training-derived libraries contributes nothing on this set, as
the benchmark's design intends. The probe does not test block-level reuse (helpers called inside new
programs); that is what the spike's ablation measures. Pang used his library as prompt context for
refinement, not as direct answers, which is consistent with this result.

### 3.4 Pre-seed offline, or grow during the hidden run?

- **Pre-seeding** costs no hidden-run tokens, can be tested on public data beforehand, and cannot be
  polluted during the run. Sources: Hodel's DSL (MIT) as the map of recurring operations (the most
  used primitives in his 400 solvers are `objects` 217 solvers, `fill` 166, `mapply` 139, `compose`
  131, `ofcolor` 124; counted here), re-arc (MIT), helper functions mined from the 28.1% of NV traces
  with tool calls (CC BY 4.0 per its text; attribution required; metadata license pending), and our
  own. BARC and PANG have no license file: read them, do not vendor them.
- **Growing** can draw only on the roughly 240 hidden tasks, each novel by design. It adds failure
  modes: verified-but-wrong programs enter the library, and blocks that encode one task's quirks get
  reused (the Factorio pattern, PA-paper 3.5). Growth helps only tasks attempted after the block
  exists, so a second pass over unsolved tasks is needed to collect any benefit (Pang ran 2 rounds).
- **Recommendation:** v1 ships a frozen, pre-seeded module of 40 to 80 tested numpy helpers. In-run
  growth goes only through the librarian path of Section 5.3, behind a switch, and is judged by the
  ablation of Section 5.7.

### 3.5 Retrieval from hundreds of functions with a small context

- **Execution probes, zero tokens:** run every grid-to-grid block on the demonstration inputs and
  rank by pairs reproduced, then cell accuracy (Pang's primary and secondary scores, PANG README
  L100); show the top few with their scores. Cost: the probe ran 1,100 programs over 120 tasks in
  208 s on 14 local processes; a 4x L4 session has 48 CPUs and a 173 GiB cgroup memory limit
  (`~/wt-MT/runs/MT-01/run_logs/run.json`, `host`; `~/wt-MT/docs/experiments/MT-01.md` L98).
- **Task signatures:** shape relation (same, scaled, cropped), colors added or removed, object
  counts, symmetry flags, changed-cell fraction. Each block keeps the signatures of the tasks where
  it verified; signatures filter candidates before probing.
- **Text index:** a one-line entry (name, signature, docstring) costs about 20 tokens, so 300 blocks
  are about 6k tokens. Keep 1k to 2k tokens of the most general helpers in the shared system prefix,
  where prefix caching makes it nearly free across requests (SGL), and give the agent
  `lib.search(query)` (tf-idf, as PA `harness.search`) and `lib.probe()` in the REPL for
  progressive disclosure.
- **Embeddings** would need another model on the GPUs or CPU; no ARC evidence that they beat
  execution probes; leave them out of v1.

## 4. Verification and anti-hacking

- **Hard gate.** A program counts only if it reproduces every demonstration output exactly (GB;
  BARC equation 4; SOAR; POETIQ `arc_agi/solve_coding.py` L104 to L111; NV `commit_solution`).
- **Least privilege** (the Factorio lesson, PA-paper 3.5). The verifier re-runs the candidate's
  source in a fresh subprocess whose namespace holds only numpy and the library snapshot: no task
  variables, no files, no network. An AST policy allows imports only from an allowlist (numpy,
  scipy, math, itertools, collections, functools, copy, typing, the library) and rejects `open`,
  `eval`, `exec`, `compile`, `__import__`, `globals`, `vars`, and attribute access to dunder names.
  This blocks programs that read demonstration outputs from the REPL or from disk.
- **Hardcoding guards.** Reject a program that embeds any demonstration output, or any block of at
  least 3 by 3 cells of one, as a literal; that compares the whole input to a literal; or that keys
  on the input's hash, bytes or string form. Penalize branches on exact input dimensions that equal
  demonstration sizes (the NV tool text bans hardcoded dimensions and case-by-case logic, a sign the
  failure is common in practice; frequency **unverified**).
- **Test-input sanity.** The program must run on every test input within limits and return a valid
  grid (1 to 30 per side, colors 0 to 9). Demote outputs equal to the test input when no
  demonstration has output equal to input, outputs equal to a demonstration output, and outputs
  with colors seen neither in demonstration outputs nor in the test input.
- **Symmetry checks, tie-break only.** Land discarded augmentation for code candidates: color
  permutations "add little signal" because an overfit program also passes them, and rotations and
  mirrors change the meaning of orientation-dependent tasks (LAND Section 8.5). Use a transform $T$
  only if the program already commutes with $T$ on every demonstration pair, and then only to break
  ties.
- **Held-out pair.** No ARC evidence found for leave-one-out program checks; it costs a second
  generation from fewer demonstrations (the public evaluation set averages 2.98 per task: 358 pairs
  over 120 tasks, K0 L23). Reserve it for tasks where verified programs disagree on the test
  output.
- **Expected precision.** With a DSL, false positives were rare (1.25% in DC-ARC Section 4.2;
  Hodel's writeup). For LLM-written Python on ARC-AGI-2 the verified-but-wrong rate is unknown: a
  spike metric.
- **Choosing 2 attempts per test input.** Group verified programs by their outputs on all test
  inputs (POETIQ `solve_parallel_coding.py`; GB majority vote); rank groups by the number of
  independent attempts that produced them, then by the smallest program in the group (Hodel's
  "shorter is better"), then by guard flags. Attempt 1 is the top group. Attempt 2 is the next group,
  else NVARC's top candidate if it differs, else a direct grid answer from the model, else the best
  unverified program by cell accuracy (POETIQ's failure ranking). When NVARC's top candidate equals
  a verified output, that output goes first.
- **No verified program:** NVARC's two attempts (or the model's direct grid answers when NVARC does
  not run), as BARC's ensemble does ("attempting induction
  first, then transduction if none of the candidate hypotheses explained the examples", BARC
  Section 2). NVARC's own confidence is informative for triage (promotion AUROC 0.82 on P2.6-01,
  `docs/STATE.md` P2.8 entry).
- **Library admission** (Section 5.3) uses the same gate plus provenance and rollback, so a
  contaminated block can be removed with every program that used it.

## 5. A concrete design for 12 h on 4x L4

### 5.1 Process layout

```text
Kaggle 4x L4 session: 48 CPUs, 173 GiB cgroup memory (MT-01 run record), offline
  [hybrid only] NVARC stage first: writes candidates and scores, frees the GPUs
  engine: SGLang or vLLM, OpenAI-compatible server on localhost, prefix caching on;
          TP and replica layout, quantization and MTP or EAGLE settings from the models note
  orchestrator: one Python asyncio process
    scheduler      task queue, phases, token and wall-clock governor,
                   writes submission.json after every finished task (crash-safe)
    agents         16 to 64 concurrent coroutines, one conversation each
    library writer the only code that mutates the library; publishes immutable lib/v{n}/
    selector       groups verified programs, picks 2 attempts, NVARC fallback
  REPL sandboxes:  one subprocess per active agent; task grids and lib/v{n} preloaded;
                   RLIMIT_AS 2 GiB, 10 s per cell, killed and restarted on breach
  verifier pool:   about 16 stateless subprocesses; fresh namespace; AST policy; 2 s per call
```

NVARC's 4B workers and the 27B engine cannot share the GPUs, so a hybrid runs them in sequence. The
writer lives in the orchestrator's event loop, so there is exactly one writer without locks.
Sandboxes import a fixed snapshot version and see new blocks only when they restart or call
`lib.refresh()`.

### 5.2 Agent loop

- **Prompt:** a compact shared prefix (rules, REPL API, helper index of 1k to 2k tokens; cached
  across requests), then the task grids as text, the top helpers by execution probe, and in later
  phases at most 2 earlier failed programs with their NV-style reports.
- **Turn:** the model emits one code cell; the REPL runs it (printing and analysis allowed). When
  the cell defines `transform`, the verifier runs it and replies with the NV-style report. Stop on a
  verified program, on the attempt's token cap, or after 3 to 5 rounds. `answer(grid)` records a
  direct grid as an unverified candidate, which hedges against tasks that Python expresses badly
  (BERMAN).
- **Sub-agents:** the scheduler, not the model, launches independent attempts (diversity, LAND
  Section 3). The REPL exposes `spawn()` only to the librarian in v1, so token costs do not
  compound.
- **"Many smaller agents":** with one 27B engine this means many short concurrent calls of the
  strongest model, not smaller models. SOAR shows open coders without long reasoning reach 1 to
  2.5% one-shot and need thousands of samples per task (SOAR Table 1), and CH shows a capability
  floor; a second, smaller model is worth it only if the models note finds one with a strong
  ARC-AGI-2 score.

### 5.3 Library

- **Contents:** code blocks only, each a pure function with docstring, typed signature, unit tests,
  provenance (task ids, agent, phase) and version. No prompt notes or memories in v1: CH found most
  authored entries unused, and LAND found prescriptive prompting narrowed hypotheses.
- **Admission (the /refine analogue):** a librarian call (at most 4k output tokens) reads one
  verified program and proposes a block plus the program rewritten to call it (ReGAL-style
  refactoring). The writer accepts only if the block passes the AST policy, its unit tests pass,
  the rewritten program still verifies on its own task, the block is not a behavioral duplicate on a
  fixed probe set of grids, and it holds no demonstration literal. Each accepted edit is logged as a
  refinement event with its evidence (PA's schema), so it can be rolled back.
- **What a block buys:** a block such as the human's "rotate, then swap colors" is two numpy lines.
  Its value is that it is tested and that execution probes surface it on a later task where it
  reproduces part of a demonstration, not that it saves tokens (code is about 9% of output, Section
  2.1).
- **Demotion:** a block used by verified programs whose test outputs disagree, or one never used
  after N tasks, leaves the default index but stays importable.
- **Cost cap:** librarian calls at most 5% of tokens.
- **Composition at zero tokens (optional):** idle CPUs can enumerate depth-2 compositions of
  grid-to-grid blocks against the demonstrations (Icecuber-style). Expected yield is small: Icecuber
  scores 2.9% on the public and 0.4% on the private ARC-AGI-2 evaluation (APD).

### 5.4 Scheduling and cascade

- **Phase 0 (CPU, during engine load):** parse tasks, compute signatures, run execution probes of
  the library, read NVARC candidates and confidence if present.
- **Phase 1 (cheap, every task, easy first):** order by predicted ease (small grids, same in and out
  shape, few colors) so early wins seed the library; one attempt with capped reasoning and up to 2
  feedback rounds.
- **Phase 2 (deep, unsolved tasks):** a long attempt with Phase 1's failures as context; in a hybrid,
  tasks where NVARC is least confident go first.
- **Phase 3 (frontier and second pass):** extra attempts on tasks with partial progress (some
  demonstrations reproduced, high cell accuracy) or with disagreeing verified programs, and a second
  pass over unsolved tasks once the library has grown. Stop generation at 11 h.
- **Evidence for cascading:** Qwen3.8-27B scores 22.8% at Low effort for $0.238 per task against
  42.4% at XHigh for $0.447 (Medium is anomalous because of the chat template, AP-QWEN; APD);
  Land's early stop on agreeing candidates was right on 36 of 37 instances (LAND Section 6.6);
  BARC's induction-then-transduction policy.

### 5.5 Token budget per task

Assumptions (all **unverified** until the spike): plan to finish generation at 11 h of the 12 h
limit; engine start 0.25 h; a hybrid gives NVARC either 6 h (reduced views and steps) or its current
projection of about 8.9 h (0.74 of 12 h on 240 tasks, P2.6-01 in `RESULTS.md` L26); aggregate
decode throughput 400 tokens/s (participants' reports for Qwen3.8-27B FP8 on 4x L4, leaders note
Section 1.3) or 2000 tokens/s (the optimistic case to be measured). Prefill is not counted; with
prefix caching expect roughly 10 to 20% more engine time (estimate).

| Scenario | Generation time | Total at 400 tok/s | Per task | Total at 2000 tok/s | Per task |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 agent only | 10.75 h | 15.5M | 64k | 77.4M | 322k |
| S2 NVARC 6 h, then agent | 4.75 h | 6.8M | 28.5k | 34.2M | 143k |
| S3 NVARC as now (8.9 h), then agent | 1.85 h | 2.7M | 11k | 13.3M | 56k |

Reference point (**derived**): Qwen3.8-27B at XHigh spent about 172k output tokens per task on the
semi-private set ($0.4469 per task at $2.55 per million output tokens, AP-QWEN, input neglected),
for direct grid answers at 42.4%.

Proposed allocation (design targets for the spike to tune):

| Phase | S1, 400 tok/s | S1, 2000 tok/s | S2, 400 tok/s | S2, 2000 tok/s |
| --- | --- | --- | --- | --- |
| 1, every task (S2: NVARC-uncertain tasks) | 12k each, 2.9M | 24k each, 5.8M | 12k on 140 tasks, 1.7M | 16k each, 3.8M |
| 2, unsolved | 40k on about 170, 6.8M | 2 x 60k on about 170, 20.4M | 40k on about 100, 4.0M | 2 x 60k on about 170, 20.4M |
| 3, frontier and second pass | 5.0M (about 50 tasks at 100k) | 47.3M | 0.8M | 8.3M |
| librarian (at most 5%) | 0.8M | 3.9M | 0.3M | 1.7M |

At 400 tokens/s an agent-only run gets about one XHigh-length call per task, a fraction of what
produced 42.4%; at 2000 tokens/s it gets about twice that. In S3 the agent can afford about 45 long
calls at 400 tokens/s, all on NVARC's least confident tasks, and library growth is pointless. The
throughput measurement decides the plan more than any harness choice.

### 5.6 Minimum viable version (about a week, CPU, mocked model)

Module names are placeholders for the orchestrator's task brief.

1. `sandbox`: persistent REPL subprocess with limits, kill and restart; stateless verifier pool.
2. `guards`: AST policy, literal and shape-dispatch detection, test-input sanity checks.
3. `engine`: async OpenAI-compatible client with token accounting, plus a mock engine that replays
   (a) Hodel solvers for ARC-AGI-2 training tasks (389 of 391 verify, PROBE), (b) hostile programs
   (literal lookups of demonstration outputs, file reads, infinite loops, huge allocations, wrong
   shapes, exceptions), (c) plausible wrong programs, to test feedback rounds and budget stops.
4. `agent`: the single-task loop with NV-style reports and `answer(grid)`.
5. `library`: store, writer, snapshots, signatures, execution probes, `lib.search`; librarian
   mocked.
6. `scheduler` and `select`: phases, governor, crash-safe submission writer, attempt selection,
   NVARC fallback from its candidates file.
7. Tests: unit tests per module; an end-to-end CPU run on DEV tasks with the mock engine; fault
   injection (sandbox crash, engine stall, time-limit hit); the verification throughput on 240 tasks
   times 50 programs.

GPU runs then follow BRIEF A8 (script, YAML config, CPU `--smoke`, runbook) and A9 (approved
pre-run brief).

### 5.7 The ablation that shows the library helps

- **Set:** 40 ARC-AGI-2 public evaluation tasks not used for tuning. Qwen3.8's exposure to them is
  unknown and affects all arms alike.
- **Arms at equal tokens per task, same seeds:** A, no library (numpy only); B, frozen pre-seeded
  helpers; C, B plus in-run growth and a second pass, run in 2 task orders.
- **Measures:** tasks correct at pass@2; verified solves per million tokens; verified-but-wrong
  count; share of verified programs that call a block; share of grown blocks used by a different
  task.
- **Decision:** ship C only if it beats A by at least 2 tasks of 40 (the project's DEV40 noise
  floor) and grown blocks are reused by other tasks; otherwise ship B if it is not worse than A,
  else A. Use paired per-task comparisons. 40 tasks is underpowered for small effects; a larger set
  would cost more 4x L4 quota than the schedule allows.

## 6. Risks and the 3-day spike

### 6.1 Risks

| Risk | Evidence | Mitigation |
| --- | --- | --- |
| Throughput near 400 tokens/s leaves about 1 long call per task | Section 5.5; leaders note 1.3 | measure on day 2; hybrid with NVARC triage; stop if below 300 tokens/s |
| Engine support for the 27B architecture on L4, offline | leaders note 5.1 | models note |
| The model uses the REPL and library poorly | PA-paper 5; CH 4.4, 6; AP-M1 | plain mode: one code cell per round with execution feedback; library off by default |
| Python is brittle for ARC-AGI-2 rules | BERMAN; LAND Section 6.6 (code is one of three channels) | `answer(grid)` direct candidates; NVARC fallback |
| Library pollution or reward hacking | PA-paper 3.5 | Section 4 guards, admission, provenance, rollback |
| Verified-but-wrong programs take attempt slots | Section 4 | votes, size, sanity checks; measure precision |
| A0 says nothing in the pipeline is textual | BRIEF A0; `docs/STATE.md` 9 Oct checkpoint | the human's waiver is still pending |
| Schedule: 24 days, 1 submission per day, long queue delays, run-to-run spread of 28 to 34 for identical NVARC code | leaders note 1.1, 1.3 | spike gate on day 4; keep a pure NVARC selection |
| Licenses for the open-source requirement | Section 1.1, 3.4 | vendor only MIT or Apache code; reimplement BARC and Pang ideas; attribute NV if mined |

### 6.2 The spike

- **Day 1 (CPU):** sandbox, verifier and guards; the single-task loop with NV-style reports; mock
  engine; attempt selection; a frozen helper module of about 40 functions modeled on the operations
  Hodel's solvers use most. Prepare the GPU script per A8 and its pre-run brief per A9. The library
  writer waits.
- **Day 2 (4x L4, about 3 h, about 6 quota hours):** engine from the models note. Measure aggregate
  tokens/s at concurrency 16, 32, 64 with the real prompts, plus prefix-cache hit rate and time to
  first token. Then Phases 1 and 2 on 40 public evaluation tasks (including 20 that NVARC misses):
  verified rate, precision of verified programs on test, tokens per verified solve, rounds used,
  guard rejections, sandbox failures, wall time per phase.
- **Day 3 (4x L4, about 3 h):** if day 2 found verified solves, arms A, B, C of Section 5.7 at equal
  tokens on the same 40 tasks; record block usage and admissions.
- **Day 4 gate:** go if at least about 5 of the 20 NVARC misses are solved by verified programs and
  throughput is at least 300 tokens/s (leaders note 5.1); ship library growth only under the rule of
  Section 5.7.

## 7. Open and unverified items

- Throughput, engine and quantization for the 27B on 4x L4, and the 2000 tokens/s case (models note).
- Whether Qwen3.8-27B was trained on NV traces or on ARC-AGI-2 public evaluation tasks.
- The derived token counts from arcprize costs; the hidden test's task sizes.
- Whether Prime Agent's ARC-AGI-3 runs built skills during the scored runs (not stated by PA-paper
  or PA-blog); no ablation isolates the continual harness.
- The NV "natural-language grid manipulator" schema (not in my 35 rows).
- Node availability in Kaggle's image (not needed by the recommendation).
- How often LLM-written ARC-AGI-2 programs hardcode demonstration details (only the NV tool text
  suggests it is common).
