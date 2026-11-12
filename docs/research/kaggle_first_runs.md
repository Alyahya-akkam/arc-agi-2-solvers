# Kaggle before the first runs: reference notebook, competition, API mechanics

Author: `researcher` subagent. Date: 2026-09-29. Feeds: P0.1-01 (reference capture), the CPU
environment probe, P0.5, the A5 checklist, P0.2 (metric), ADR-008 (`scripts/kaggle_ops.py`).

## Question

1. Which Kaggle notebook is the 2025 first-place TTT notebook on `sorokin/qwen3_4b_grids15_sft139`,
   under what license, with which attachments, packages, dev IDs, seeds, adapter handling, TTT and
   decode settings, worker layout, and output files?
2. What are the facts of the 2026 ARC-AGI-2 Kaggle competition (slug, data, submission, runtime,
   accelerators, internet, limits, deadlines, open-source rule, quota, rerun flag), including the
   exact scoring formula?
3. How does the Kaggle CLI behave for what `kaggle_ops.py` does (metadata fields, accelerators,
   quota, `--file-pattern`, cancel, notebook chaining, output limits)?

## Sources and conventions

| Source | Revision | Kind |
| --- | --- | --- |
| `github.com/1ytic/NVARC` (README, `ARC-AGI1/`, `nvarc_2025.pdf`) | `846d0198efa752534594e321fc3289fc0a06c657` | primary (authors); no LICENSE file |
| `github.com/Kaggle/kaggle-cli` tag `v2.2.4` (installed locally, with `kagglesdk` 0.1.37) | `f0afa32699d28c97f82691728ada3ed8c16c5abf` | primary |
| `github.com/Kaggle/kaggle-cli` `main` (unreleased "Next") | `8ba133d8b54b9d21e52638bf4e7da07500d21b63` | primary |
| `kagglesdk` 0.1.37 sdist, <https://pypi.org/project/kagglesdk/0.1.37/> | 2026-08-06 | primary |
| `github.com/Kaggle/docker-python` | `e096cbc5b6ff50ab49130187f8677dedb9b5e9bd` | primary |
| `github.com/arcprize/arc-agi-benchmarking` (`scoring/scoring.py`) | `28e67d54b05df5be10281892243c509a42a874f1` | primary (organizers) |
| `github.com/arcprize/ARC-AGI-3-Kaggle-Starter` | `eeb1535404f321d280a8f9194bbc1d7aca5f05fc` | primary (organizers) |
| `github.com/arcprize/ARC-AGI-2` | `f3283f727488ad98fe575ea6a5ac981e4a188e49` | primary |
| arcprize.org: `/competitions/2026`, `/competitions/2026/arc-agi-2`, `/competitions/2025`, `/blog/announcing-arc-agi-2-and-arc-prize-2025`, `/blog/arc-prize-2025-results-analysis` | read 2026-09-29 | primary |
| `unsloth` 2025.9.7 sdist, <https://pypi.org/project/unsloth/2025.9.7/> | 2025-09-19 | primary |
| `github.com/johnsonhk88/kaggle-ARC-Prize-2026-ARC-AGI-2` (copy of the Kaggle overview; four Kaggle-run NVARC forks with logs) | `9d86c9592ee58597ed960db0e1baadc26dec81a6` | third party |
| `github.com/lhzwb2008/arc-prize-2026` (2026 NVARC port, `kernel-metadata.json`) | `4ef2d594d2a93bd6f8d73452bbd7f0980798f734` | third party |
| `github.com/vankhaivn/compute-relay/pull/36`, `github.com/tushar-mahalya/kdev/issues/17`, <https://trelis.substack.com/p/nvarc-2025-arc-prize-winners> | 2026 | third party |

Every kaggle.com page (notebooks, model, competition overview, data, rules, evaluation, docs,
discussions) returns only its `<title>` to a non-JS fetch: bodies are JS-rendered and were not
readable. Page titles confirmed that `sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue`,
`gregkamradt/arc2-qwen3-unsloth-flash-lora-batch8-queue-trm2`, `sorokin/pip-install-unsloth-flash-patch`,
`cpmpml/arc2-trm-v31`, and `models/sorokin/qwen3_4b_grids15_sft139` ("ivan | qwen3_4b_grids15_sft139")
exist publicly. Anything resting on a kaggle.com body, a search snippet, or a third party is marked
**unverified**. Line numbers in notebooks are 1-based within the cell source (the `%%writefile`
line is line 1).

## Short answer

- **Reference:** `sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue`, named "the submission
  notebook" by NVARC's README; `ARC-AGI1/002_ivan_arc1.ipynb` in NVARC's repo is a readable copy
  (same code, other paths). ARC Prize's 2025 page instead links a fork,
  `gregkamradt/arc2-qwen3-unsloth-flash-lora-batch8-queue-trm2`, as NVARC's 24.0% code.
- **Attachments:** competition data, model `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`,
  and notebook `sorokin/pip-install-unsloth-flash-patch`, whose pip-installed packages supply the
  stack (third-party Kaggle logs: Python 3.11.13, torch 2.8.0+cu128, transformers 4.55.4, unsloth
  2025.9.7, xformers 0.0.32.post2, triton 3.4.0, FA2). No datasets, no wheel datasets, no vLLM, no
  merge.
- **Method:** rsLoRA $r=256$, $\alpha=32$ on all 7 projections plus fully trained `embed_tokens`
  and `lm_head`, 1 epoch over 128 augmented samples; turbo DFS over 16 views per test input;
  8-view rescoring; 4 workers, one per L4, on a shared queue. DEV4 = `0934a4d8`, `36a08778`,
  `981571dc`, `aa4ec2a5`.
- **Not deterministic by construction:** salted `hash(bk)` seeds, batched DFS (authors say so),
  wall-clock caps. Third-party Kaggle runs of the same code scored DEV4 3.0/4 twice.
- **The 2025 environment no longer runs by default:** Kaggle's image source (docker-python) moved
  to Python 3.12 on 2025-12-12 (`4ae73d1`) and to Python 3.13 with `transformers>=5` on 2026-09-28
  (`e096cbc`); deployment dates are unknown. Mount
  paths changed in 2026: data at `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/`, model at
  `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1`. The reference
  needs the old image pinned and a path patch.
- **Competition:** slug `arc-prize-2026-arc-agi-2`, `submission.json`, 12 h, L4x4 only for
  notebooks attached to the competition, at twice the quota rate, internet off, 1 submission per
  day and 2 final (**unverified**), entry 26 Oct, final 2 Nov 2026. Prize code: CC0 or MIT-0,
  published before private scores are received.
- **Scoring:** both texts say "averaged over task test outputs", but the leaderboard, the
  organizers' scorer, and NVARC's validator all average per task (fractional credit within a
  task). BRIEF B4 is right; P0.2 should also log the per-output variant.
- **CLI:** accelerator IDs are enum names (`NvidiaL4` = 4xL4 observed, `NvidiaL4X1`,
  `NvidiaTeslaT4` = 2xT4; `NvidiaTeslaP100` is retired and silently runs on T4x2), not `4xL4`.
  `enable_internet` defaults to true in code. `--file-pattern` is an unanchored `re.search` on the
  relative path, and the session log always downloads. No CLI cancel exists; `--timeout` is the
  only programmatic brake.
- **Gateway gaps:** `uses_gpu` ignores `machine_shape` and `enable_tpu`, so a GPU push can skip the
  brief check; no `enable_internet` check; no `--timeout`; a broad pull pattern; the installed CLI
  2.2.4 has an output path-escape bug that is fixed only on `main`.

## 1. Reference notebook

### 1.1 Candidates

| Handle | Evidence | Verdict |
| --- | --- | --- |
| `sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue` | NVARC `README.md` ("The submission notebook is available on Kaggle"); `ARC-AGI1/README.md` says the ARC-AGI-1 code "is the same as in the notebook we used for our winning solution", the only change being TTT on the ARC 2024 evaluation data | **Reference** |
| `gregkamradt/arc2-qwen3-unsloth-flash-lora-batch8-queue-trm2` | arcprize.org/competitions/2025 links it (`/edit?fromFork=1`) as NVARC's "Code", 24.0%; the original `sorokin/...-trm2` returns HTTP 404 (private or deleted) | Fork by ARC Prize staff; the name suggests Qwen3 with batch-8 DFS plus a TRM ensemble; contents unread |
| `cpmpml/arc2-trm-v31` | NVARC README: the TRM submission notebook | TRM only, no TTT |
| NVARC `ARC-AGI1/002_ivan_arc1.ipynb` + `ARC-AGI1/pip-install-unsloth-flash-patch.ipynb` | Readable copy. Its log shows an H100 run on 2025-12-11 (`/home/local/cpmp/...`), `rerun_mode` forced to `True`, paths `../input/arc-prize-2024/...`, output `submission1.json` | Proxy, not byte-identical |
| johnsonhk88 `src/sample-code/qwen3_4b_grids15_sft139/reproduce-nvarc-2025-results.ipynb` (Kaggle run 2026-04-08) and `arc-agi2-original-kg.ipynb` (2026-06-18) | Diff of the April file against the NVARC copy: 37 changed lines, all paths, the rerun toggle, and the output name. The June file adds a path resolver and a stable seed | **Unverified** provenance, but these carry real Kaggle logs |

Which submission won the private leaderboard: the NVARC paper (p. 10) names two final selections,
"Qwen3 2B" plus TRM (public 20.28) and "the last submission with Qwen3 4B alone". Trelis says "The
winning submission used Qwen-4B alone, scoring 24% on the private leaderboard" (**unverified**,
secondary). The batch4 notebook (4B alone) is therefore the likely private winner, but the
trm2 fork link on arcprize.org leaves this **unverified**.

### 1.2 License

- NVARC GitHub repo: no LICENSE file; the GitHub API reports `license: null`. The GitHub copy has
  no explicit license.
- Public Kaggle notebooks show a license footer, commonly Apache 2.0 by default. **Unverified**:
  the page is unreadable and this rests on search results only. The human reads the footer on
  the batch4 page.
- 2025 open-source rule (arcprize.org announcement): "in order to be eligible for prizes,
  contestants must open source and share their solution and work into the public domain at the end
  of the competition." Results post: "all ARC Prize 2025 winning solutions and papers are
  open-source." No license is named on the pages read.
- 2026 consequence: third-party code in our submission must be under an open-source license that
  allows public sharing (Section 2), so the batch4 page's license decides whether extracted NVARC
  code may ship. The model's license on its Kaggle page is unread.

### 1.3 Attachments

From `metadata.kaggle` of `002_ivan_arc1.ipynb` (NVARC copy) and of the 2026 forks:

| Source type | ID in metadata | Handle and mount |
| --- | --- | --- |
| `competition` | 91496 (bundle 11802066) in the 2025 copy; 133469 in the 2026 forks | Forks read `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/` |
| `modelInstanceVersion` | 627758 (modelId 488596, instance 472720), identical in all copies | `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1` (lhzwb2008 `notebooks/nvarc_2026/kernel-metadata.json`). Mount: `/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1` in the April 2026 log; `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1` in the June 2026 log ("Using model dir: ...") |
| `kernelVersion` | 272622465, identical in all copies | `sorokin/pip-install-unsloth-flash-patch` (named in lhzwb2008 `kernel_sources`; NVARC `ARC-AGI1/README.md`) |
| datasets, wheel datasets | none | |

Other settings: accelerator `nvidiaL4`, `isInternetEnabled: false`, `dockerImageVersionId: 31090`,
Python 3.11.13. The ID-to-handle mapping is **unverified** (third-party metadata).

### 1.4 Packages

- The main notebook installs nothing. Cell 1 runs `!pip uninstall -y tensorflow`.
- The install notebook (NVARC copy, `pip-install-unsloth-flash-patch.ipynb`; its metadata: T4,
  internet on, image 31090):
  - cell 0: `pip install unsloth==2025.9.7 unsloth_zoo==2025.9.9 numpy==2.2.6 matplotlib==3.10.6
    scikit-learn==1.7.2`;
  - cell 1: `pip install --upgrade --no-deps --no-build-isolation flash-attn`, **unpinned**;
  - cells 5 to 6: patch `unsloth/models/qwen3.py` so batched decode attention calls
    `flash_attn_func`, plus a `temp_O` shape fix.
- Versions logged on Kaggle with kernel version 272622465 attached (forks, 2026-04-08 and
  2026-06-18): Unsloth 2025.9.7, Transformers 4.55.4, Torch 2.8.0+cu128, CUDA toolkit 12.8,
  Triton 3.4.0, Xformers 0.0.32.post2, FA2 True, bf16 True, "NVIDIA L4 ... Max memory: 22.034 GB".
  **Unverified** (third-party logs).
- The NVARC copy's own install log shows drift: torch 2.9.1, triton 3.5.1, xformers 0.0.33.post2,
  flash-attn 2.8.3. The unpinned install resolves differently over time, so the attached
  notebook *version* is what fixes the stack.
- `peft` is not printed anywhere. The base image of the NVARC install log had 0.16.0; lhzwb2008
  `scripts/nvarc/requirements.txt` claims 0.17.1 and a `flash_attn-2.8.2+cu128torch2.8` wheel.
  **Unverified**.
- vLLM is not used.

### 1.5 Code facts (NVARC copy; the forks match apart from paths)

- **Dev IDs:** `starter.py` (cell 5) line 53: `["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]`.
  All four are in ARC-AGI-2 public evaluation (`arcprize/ARC-AGI-2` `data/evaluation`, 120
  tasks). DEV4 has 5 test outputs (fork log: "Generating submission for 5 outputs").
- **Mode:** `rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")` (cell 4 line 223; cell 5
  line 39; cell 9). In rerun mode the notebook takes `arc-agi_test_challenges.json` and all keys.
  Otherwise it takes `arc-agi_evaluation_challenges.json` and DEV4 only, then scores with
  `arc-agi_evaluation_solutions.json` in the last cell.
- **Seeds:**
  - `augment()` calls `np.random.seed(seed)` (cell 2 lines 248 to 255). Training uses `seed=1`
    (cell 4 line 328), evaluation `seed=2` (line 363), and rescoring `seed=hash(bk) % 1024**2`
    (line 444). The last is Python's per-process salted `str` hash, and no `PYTHONHASHSEED` is set.
  - LoRA init uses `random_state=42` and the Trainer `seed=42` (lines 225 to 261).
  - No torch determinism flags are set.
- **Determinism:** the paper, section 3.3, says batched DFS "became nondeterministic", and the
  batch-invariant fix was dropped as 17% slower. Section 3.5: "1-2 points variability in the
  scores across different re-runs of the same notebook". Results also depend on wall-clock caps:
  540 s per DFS call (cell 4 line 121), 1200 s per puzzle (line 406), and 12 h minus 600 s overall
  (cell 0).
- **Adapters:** no `merge_and_unload`. The initial PEFT state is cloned (lines 280 to 281) and
  restored per puzzle with `set_peft_model_state_dict` (lines 318 to 322). Unsloth kernels are
  used throughout: `FastLanguageModel.from_pretrained`, `get_peft_model`, `for_training` and
  `for_inference`. All fp32 parameters are cast to bf16 after PEFT wrapping (lines 276 to 278).
- **Worker layout:** `mp.spawn(local_worker, nprocs=4)` (cell 5 line 59) with
  `CUDA_VISIBLE_DEVICES=rank` (line 12). Workers share a `Manager().Queue()` of sorted keys plus
  four `None` sentinels. Unsloth import is staggered through marker files `../worker{rank}`
  (lines 17 to 25). Launch line (cell 6): `UNSLOTH_DISABLE_STATISTICS=1
  TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 python starter.py --end-time ...`.
  So yes: 4 workers on 4xL4.
- **Files written:**
  - in `/kaggle/working`: `arc_loader.py`, `arc_decoder.py`, `arc_solver.py`, `starter.py` (from
    `%%writefile`) and `submission.json` (the NVARC copy writes `submission1.json`);
  - outside `/kaggle/working`, so not saved as outputs: `/kaggle/worker0..3` markers and
    `/kaggle/inference_outputs/<subkey>` (bz2 pickles of `{beam_score, score_aug, solution}`).

TTT and decode at a glance (cell 4 unless noted):

| Setting | Value |
| --- | --- |
| LoRA | $r=256$, $\alpha=32$, `use_rslora=True` so scale $\alpha/\sqrt{r}=2$, dropout 0, bias none (lines 225 to 235) |
| Targets | `q,k,v,o,gate,up,down_proj` as LoRA. `embed_tokens` and `lm_head` become `modules_to_save`, i.e. full trainable copies (unsloth 2025.9.7 `unsloth/models/llama.py` lines 2484 to 2505; the log prints "Training embed_tokens in mixed precision") |
| Optimizer | 1 epoch, batch 1, accumulation 1, lr $5\times10^{-5}$, cosine, warmup ratio 0.1, `adamw_torch`, weight decay 0, clip 1.0, bf16, no gradient checkpointing, `max_seq_length` 8192, no 4-bit (lines 237 to 272) |
| Loss | completion-only, on the output grid of every demo pair (collator, lines 78 to 95) |
| Train data | 8 geometric views (identity or transpose, times $\mathrm{rot90}^k$) $\times$ 16 color permutations = 128 samples, demo order shuffled, demos dropped until $\le 8192$ tokens (lines 328 to 329). The fork logs show `global_step=128` |
| Decode | per test input, 8 geometric $\times$ 2 permutations = 16 views in 4 batches of 4 (lines 363 to 397). Turbo DFS keeps prefixes with cumulative NLL $< -\ln 0.2 \approx 1.609$ (line 292); `max_new_tokens` = tokens of a $30\times30$ reply + 1 |
| Rescore and select | each candidate is scored under 8 fixed views in two batches of 4 (lines 444 to 452). `score_kgmon` = DFS hit count minus mean augmented NLL (cell 3 lines 30 to 36, 61 to 62). Top 2 become the attempts; missing tasks get `[[0]]` (cell 2 lines 257 to 261) |

### 1.6 Observed cost on Kaggle (third-party logs, **unverified**)

- DEV4 notebook wall time on `nvidiaL4`: 1606 s (2026-04-08) and 1694 s (2026-06-18).
- Per puzzle (April), total with training in parentheses: `0934a4d8` 541 s (393 s), `aa4ec2a5`
  645 s (298 s), `36a08778` 1203 s (208 s), `981571dc` 1280 s (880 s).
- Peak memory: 13.0 GB in training, 17.1 GB in inference, out of 22 GB.
- DEV4 score: 3.0/4 on 2026-04-08 and 2026-04-16 (original seeds); 2.5/4 on 2026-06-18 (seeds
  modified).

## 2. Competition (ARC Prize 2026, ARC-AGI-2 track)

| Item | Answer | Source | Status |
| --- | --- | --- | --- |
| Slug | `arc-prize-2026-arc-agi-2` | arcprize.org/competitions/2026 (Kaggle link) | primary |
| Data mount | `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/` | June 2026 fork log loads evaluation challenges and solutions from there; lhzwb2008 README line 167; the organizers' ARC-AGI-3 starter installs from `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/` (`scripts/build_notebook.py` line 68) | **unverified** for this slug (strong) |
| Files | `arc-agi_evaluation_challenges.json`, `arc-agi_evaluation_solutions.json`, `arc-agi_test_challenges.json` (seen in Kaggle-run code); `arc-agi_training_{challenges,solutions}.json` and `sample_submission.json` expected from 2025 | forks; 2025 layout | **unverified** |
| Test file | Commit runs see a public placeholder; the rerun swaps in the hidden tasks, 240 in 2025 ("Kaggle then runs the notebook against a hidden set of 240 tasks", NVARC paper p. 1) | third parties disagree on the placeholder's content | **unverified** for 2026 |
| Submission | `submission.json` holding `{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`, one dict per test input in order, every task id present, both attempts present | Kaggle overview copied in johnsonhk88 README ("Submission File"); matches NVARC `get_submission` | **unverified** (copy) |
| Runtime | CPU or GPU notebook $\le$ 12 h | overview copy; 2025: "12 hours with 4 L4 GPUs for solving 240 tasks" (NVARC paper p. 2) | 2026 **unverified** |
| Accelerator | "Kaggle's pool of L4x4 machines (96GB of GPU memory)"; "L4x4 notebooks consume GPU quota at twice the rate of T4x2 and P100 machines"; "L4s are only available for notebooks attached to this competition; attempts to circumvent this may lead to team or account bans"; "all L4 sessions must have internet disabled" | overview copy ("Upgraded Accelerators"); 2025 announcement: "2x compute vs. 2024 (L4x4s)" | **unverified** for 2026 |
| Rerun accelerator | The submitted version's shape, which was 4xL4 in 2025 | NVARC paper p. 2 | mechanism **unverified** |
| Internet | "Internet access is not available during Kaggle evaluation" | arcprize.org/competitions/2026 | primary |
| Daily limit | 1 per day | 2025: "Kaggle limits submissions to one per day per team" (NVARC paper p. 1); 2026: search snippet of the rules plus two third-party READMEs | 2026 **unverified** |
| Final submissions | up to 2 | search snippet of the rules ("You may select up to two (2) Final Submissions") | **unverified** |
| Deadlines | Entry and team merger 26 Oct 2026; final 2 Nov 2026; results 4 Dec 2026; 11:59 PM UTC | arcprize.org (2 Nov, 4 Dec); overview copy (26 Oct, time of day) | mixed |
| Open source | "all code and methods authored by the submitter must be made open source under a permissive public domain license (eg. CC0 or MIT-0)"; third-party code "under, at least, an open source license which allows public sharing (eg. Apache-2.0, GPLv3)"; "Participants must open source their solutions before receiving official private evaluation scores" | arcprize.org/competitions/2026 | primary |
| Removal rule and writeup | "participants eligible for a prize will be removed from the competition if they do not open source their solutions"; Grand Prize artifacts attached to a Solution Writeup "within seven days of the competition's submission deadline" | arcprize.org/competitions/2026/arc-agi-2 | primary |
| Kaggle winner-license clause | A snippet of the ARC-AGI-3 Kaggle rules mentions CC-BY 4.0 and OSI "open source weights" | search snippet | **unverified**; read the rules tab |
| Rerun vs quota | The rerun after Submit runs on Kaggle's side and does not count; the Save & Run All version does count | Kaggle Q&A snippets (451524, 453105) | **unverified** |
| `KAGGLE_IS_COMPETITION_RERUN` | Truthy only in the rerun ("Phase B", triggered by "Submit to Competition"), absent in Save & Run All; test with `os.getenv(...)` truthiness | organizers' starter `scripts/build_notebook.py` lines 84, 140 and README "Phase A/Phase B"; NVARC code | usage primary; exact value **unverified** |
| Runtime reporting | "Submission runtimes are obfuscated; repeating the exact same submission may show up to 10 minutes of variance" | overview copy | **unverified** |

### 2.1 Scoring formula (coordinator's addendum)

Wording:

- arcprize.org/competitions/2026/arc-agi-2 (primary, readable): "For each task, you should predict
  exactly 2 outputs for every test input grid. If any of the 2 predicted outputs matches the
  ground truth exactly, you score 1 for that task, otherwise 0. The final score is the average
  across all task test outputs."
- The Kaggle evaluation page is JS-rendered and unreadable. The third-party copy (johnsonhk88
  README at `9d86c95`, section "Evaluation") reads: "Each task test output has one ground truth.
  If any of the 2 predicted outputs matches the ground truth exactly, you score 1 for that task
  test output, otherwise 0." and "The final score is the sum of the highest score per task output,
  averaged over the total number of task test outputs."
- `arcprize/ARC-AGI-2` README defines human "solve" as all test inputs correct (all or nothing).
  That is a testing definition, not the Kaggle metric.

Read literally, both competition texts describe a per-output (micro) average. Let
$c_{t,i}=\mathbb{1}[a^{(1)}_{t,i}=y_{t,i} \lor a^{(2)}_{t,i}=y_{t,i}]$, where task $t$ has $n_t$
test outputs. The literal reading is

$$S_\text{out}=\frac{\sum_t\sum_{i=1}^{n_t} c_{t,i}}{\sum_t n_t}.$$

Everything measurable points instead to per-task averaging:

$$S_\text{task}=\frac{1}{|T|}\sum_{t\in T}\frac{1}{n_t}\sum_{i=1}^{n_t} c_{t,i}.$$

1. The organizers' scorer computes `task_score / num_pairs` per task (`scoring.py` lines 45 to
   46 and 123 to 127) and prints `total_score / total_tasks * 100` (lines 160 to 172 and 234).
2. NVARC's `validate_submission` adds `1 / len(v)` per matched output (cell 2 lines 272 to 281)
   and prints DEV4 as "acc: 2.5/4" over 5 outputs.
3. Derived check (not stated by Kaggle): all eight NVARC 2025 scores in the paper equal $k/720$,
   i.e. multiples of $\tfrac16$ task over 120 tasks. These are 27.64, 29.72, 22.22, 27.22, 21.53,
   20.28 and 22.50 public, and 24.03 private. Halves and thirds come from 2- and 3-output tasks.
   No test-output denominator $N \in [100, 999]$ other than 720 reproduces all seven public
   scores, and 720 is not a plausible output count for 120 tasks.

Recommendation: BRIEF B4 (task score = mean over its test outputs, reported as percent of tasks)
matches the leaderboard. P0.2 implements $S_\text{task}$ and also logs $S_\text{out}$, so any
divergence is visible. `docs/STATE.md` records the wording conflict.

## 3. Kaggle API mechanics (kaggle-cli 2.2.4, as installed)

### 3.1 `kernel-metadata.json` fields

Code: `src/kaggle/api/kaggle_api_extended.py` `kernels_push`, lines 6319 to 6453. Docs:
`docs/kernels_metadata.md` lines 26 to 42.

| Field | CLI behavior | Note for us |
| --- | --- | --- |
| `id` | `owner/slug`, no version; warns if the title does not slugify to it | |
| `id_no` | numeric id, **preferred over `id`** | `kernels pull -m` writes it (lines 6604 to 6626). Strip it before pushing a pulled notebook, or the push targets the original owner's kernel |
| `title` | at least 5 characters; slug must match | |
| `code_file` | required, relative to the folder; code-cell outputs are stripped on push | |
| `language`, `kernel_type` | `python`, `r`, `rmarkdown`; `script` or `notebook` (lines 940 to 941); type is fixed after creation (SDK doc) | |
| `is_private` | default true; bool or `"true"`/`"false"` (`get_bool`, lines 8386 to 8400) | |
| `enable_gpu`, `enable_tpu` | default false; SDK: "DEPRECATED: use `machine_shape` instead" (kagglesdk types lines 1999 to 2014) | generic GPU = default GPU (Section 3.2) |
| `enable_internet` | **defaults to true in code** (line 6439), although the docs say false; the `kernels init` template writes `"true"` (line 6294) | always set `false` explicitly |
| `machine_shape` | accelerator ID, used when `--accelerator` is absent (line 6450) | |
| `dataset_sources` | `owner/slug[/version]` (lines 8763 to 8781) | whether the server honors a version is **unverified** |
| `competition_sources` | `competition-slug` | L4 requires this competition attached (overview copy) |
| `kernel_sources` | `owner/slug[/version]`, slug of at least 5 characters (lines 8955 onward) | pins an upstream output version |
| `model_sources` | the CLI requires `owner/model-slug/framework/variation-slug/version-number` with an integer version (lines 8927 to 8953); the server also accepts a versionless form (SDK lines 2006 to 2011) | `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1` (**unverified**; take the exact string from `kernels pull -m`) |
| `keywords` | tags (`category_ids`) | |
| `docker_image_pinning_type` | `original` or `latest` (line 942) | |
| `docker_image` | "must be one of the Kaggle-provided, known images" (SDK lines 2022 to 2025) | `kernels pull -m` reports an existing kernel's image |
| flag `-t/--timeout` | `session_timeout_seconds`: "terminate the kernel session after this many seconds of runtime" (SDK lines 2015 to 2017) | the only programmatic brake |

### 3.2 Accelerators

- `--accelerator` overrides the metadata and is forwarded unvalidated as `machine_shape`: "The
  allowed names are in an enum that is not currently included in kagglesdk" (line 6449).
- Accepted IDs (`docs/kernels.md` at `8ba133d`, "as of Sep 2026"): `NvidiaTeslaT4` (GPU T4 ×2,
  the default GPU), `NvidiaTeslaA100`, `NvidiaL4`, `TpuV5E8`, `NvidiaL4X1`, `TpuV6E8`,
  `NvidiaH100`, `NvidiaRtxPro6000`. "Some of these are only available to participants of specific
  competitions".
- Retired (commit `bc6f535`): `NvidiaTeslaP100`, `TpuV38`, `Tpu1VmV38`, `TpuV232`, `TpuV2256`.
  "The server silently runs the session on a replacement instead": P100 becomes T4x2. `Gpu`
  aliases the P100 value, so a generic GPU push runs on T4x2. The default image's PyTorch cu128
  build lacks Pascal `sm_60` kernels (kaggle-cli CHANGELOG, "Next"), and docker-python CI dropped
  its P100 test agent (`1ed8d43`).
- The caller's names map as follows:
  - 4xL4 = `NvidiaL4` (observed: forks with accelerator `nvidiaL4` ran four ranks, one L4 each;
    lhzwb2008 README lines 190 to 193 push `--accelerator NvidiaL4` for "GPU L4 ×4");
  - 1xL4 = `NvidiaL4X1` (by name; **unverified**);
  - 2xT4 = `NvidiaTeslaT4` (primary, commit `47354f3`);
  - P100 is retired.
- Strings such as `4xL4`, `2xT4`, `P100`, or `gpu` (the example in `skills/references/kernels.md`)
  are not valid IDs.
- The organizers' ARC-AGI-3 starter picks the accelerator through the notebook JSON
  (`metadata.kaggle.accelerator`, e.g. `nvidiaTeslaT4`) plus `enable_gpu`. Whether push reads
  that field is **unverified**. Set `machine_shape` and pass `--accelerator` together.

### 3.3 Quota and session limits

- `kaggle quota` (2.2.4) prints GPU and TPU hours used, remaining, total, and the refresh time
  (`skills/references/quota.md`). The SDK's `ApiAcceleratorQuota` also has `time_reserved` for
  running sessions.
- L4x4 costs quota at 2x (overview copy, **unverified**). The other figures are **unverified**
  (search summaries only): about 30 GPU hours per week, floating; a 12 h cap per GPU session; a
  weekly reset around Saturday 00:00 UTC.
- `kaggle competitions submission-limits <slug>` (2.2.4) prints "Submissions today", "Lifetime
  submissions", "Remaining today" (lines 2117 to 2155).

### 3.4 `kernels output --file-pattern`

- The pattern is compiled with `re.compile`, and each file is kept if
  `pattern.search(item.file_name)` matches (lines 6679 to 6685 and 6708 to 6711).
- `file_name` is the path relative to `/kaggle/working`, subdirectories included. The match is
  unanchored unless the pattern anchors itself.
- The session log is always written as `<kernel-slug>.log`, whatever the pattern (lines 6734 to
  6737).
- The listing has no sizes (`ApiKernelSessionOutputFile`: `url`, `file_name`), and each file is
  read fully into memory. `kaggle kernels files` does return sizes (`ApiListKernelFilesItem`:
  `name`, `size`, `creation_date`).
- 2.2.4 ignores the version in `owner/slug/version` and acts on the latest run (fixed on `main`
  in `30540b4`).
- In 2.2.4, a server-supplied name such as `../x` can escape `--path`. The fix is `68dcbdc`, on
  `main` only and unreleased.

### 3.5 Cancelling a running kernel

- The CLI has no cancel command, neither in 2.2.4 nor on `main` `8ba133d`.
- `kagglesdk` 0.1.37 has `KernelsApiClient.cancel_kernel_session(ApiCancelKernelSessionRequest)`,
  which calls `/api/v1/kernels/cancel-session/{kernel_session_id}` (services lines 124 to 134;
  types lines 115 to 145). However, no public response the CLI uses returns a session id:
  - `ApiSaveKernelResponse` has `ref`, `url`, `version_number`, `error`, `invalid_*`,
    `kernel_id`;
  - the status call returns only `status` and `failure_message`.
- Status values: `QUEUED`, `RUNNING`, `COMPLETE`, `ERROR`, `CANCEL_REQUESTED`,
  `CANCEL_ACKNOWLEDGED`, `NEW_SCRIPT` (`kernels_enums.py` lines 31 to 38).
- In practice, runs are stopped on the website and bounded with `--timeout`.

### 3.6 Chaining notebooks, and output limits

- List the upstream notebook in `kernel_sources`, optionally with a version. Its saved outputs
  (`/kaggle/working` at session end) mount read-only under `/kaggle/input`. The old layout was
  `/kaggle/input/<slug>/`; the 2026 layout is probably `/kaggle/input/notebooks/<owner>/<slug>/`
  (**unverified**; the probe prints it). Datasets now mount at
  `/kaggle/input/datasets/<owner>/<slug>/` (compute-relay PR 36, merged 2026-09-17, third party).
- "A notebook cannot mount its own output: Kaggle rejects it through `invalidKernelSources`"
  (kdev issue 17, third party).
- Packages pip-installed by an attached notebook were importable in NVARC's notebook (observed).
  The mechanism is **unverified**: the lhzwb2008 port removes `/kaggle/usr/lib/notebooks` entries
  from `sys.path`, which suggests that mount point.
- Output limit: about 20 GB of `/kaggle/working` (**unverified**, search summary).
- `datasets create` and `datasets version` default to private (`cli.py` line 2676) and to
  `--dir-mode skip`, which silently drops subdirectories (lines 1076 and 2678 to 2680).

### 3.7 `scripts/kaggle_ops.py` review (not changed here)

1. `uses_gpu` (lines 134 to 144) returns false when the metadata sets `machine_shape` or
   `enable_tpu`, yet the CLI still sends `machine_shape` (line 6450), so an accelerated run would
   skip the brief check. Treat any non-empty `machine_shape`, `enable_gpu`, `enable_tpu`, or
   `--accelerator` as accelerated.
2. Require `"enable_internet": false` explicitly, because the CLI defaults to true.
3. Require `--timeout <s>` on GPU pushes, taken from the brief's wall-time kill criterion, and
   forward it as `-t`.
4. Anchor `PULL_PATTERN`, e.g.
   `^(submission\.json|run_logs/[^/]+\.(jsonl|json|md|csv|txt|log|yaml|png))$`. The current
   extension pattern would pull a `config.json` or `tokenizer.json` written anywhere under
   `/kaggle/working`.
5. Refuse metadata that carries `id_no`, or an `id` whose owner is not ours.
6. Validate `--accelerator` against the ID list above and refuse retired IDs.
7. Consider read-only `quota` and `submission-limits` actions (needs an ADR-008 amendment).
8. Pin the CLI, or upgrade once the path-escape fix is released.

## What this means for

### P0.1-01 reference capture run

- **Source:** the human downloads the batch4 notebook from its page, or runs
  `kaggle kernels pull sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue -p <dir> -m`. This
  fetches source plus metadata only: no outputs, no weights. The pulled metadata settles the exact
  `kernel_sources` (with version), `model_sources`, `competition_sources`, and `docker_image`.
  Save the notebook as `reference/reference_notebook.ipynb` (gitignored).
- **It cannot run untouched:** 2026 mounts differ from 2025.
  - Patch only the path literals: data directory to
    `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/`, model to
    `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1` (or the
    resolved probe path).
  - Prepend one cell setting `os.environ["PYTHONHASHSEED"] = "0"`. The `!python` child and the
    spawned workers inherit it, so no original cell changes.
  - Append one post hoc cell that dumps `/kaggle/inference_outputs/*` (grid, `beam_score`,
    `score_aug`) and `pip freeze` into `/kaggle/working/run_logs/` as JSON.
  - Commit the diff (original vs pushed) with the fixture as its provenance.
- **Metadata:** `id` `<user>/arc26-p0-1-reference` (a title that slugifies to it), `is_private`
  true, `enable_internet` false, `enable_gpu` true, `machine_shape` `NvidiaL4`,
  `competition_sources` `["arc-prize-2026-arc-agi-2"]`, `model_sources`
  `["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]`, `kernel_sources`
  `["sorokin/pip-install-unsloth-flash-patch/<version>"]`, `docker_image_pinning_type`
  `"original"`, and `docker_image` from the pulled metadata (the Python 3.11 image). Without the
  pin, the latest image (Python 3.12, soon 3.13) cannot import the cp311 packages. lhzwb2008
  README lines 174 to 177 and 190 record exactly this failure, and warn that the web UI "Save
  Version" drops the pinned image.
- **Cost:** about 27 to 28 min wall per DEV4 run (Section 1.6), about 1 quota hour at the 2x L4x4
  rate. Push with `--accelerator NvidiaL4 --timeout 3600`.
- **Parity rule for the brief:** GPU nondeterminism remains even with the hash seed fixed. Run the
  reference twice (A, B; about 2 quota hours total), use A as the fixture, and accept P0.1 when
  extracted-vs-A agreement is at least B-vs-A agreement, with identity required if A equals B.
- **Pull:** `submission.json` becomes `runs/P0.1/reference_output.json`. Also pull
  `run_logs/*.json` and the session log.

### CPU-only environment probe notebook

- **Purpose:** answer mount paths, image and Python version, library versions, `KAGGLE_*`
  variables, the `sys.path` effect of the attached install notebook, and disk and RAM, without
  GPU quota. CPU sessions need no brief (A9 covers GPU runs), but only after gateway fix 1, since
  today a stray `machine_shape` would slip past.
- **Two pushes of the same code:** (a) `docker_image_pinning_type` `latest`, which shows what P0.5
  must target if unpinned; (b) pinned to the NVARC image from the pulled metadata.
- **Metadata:** no `machine_shape`; `enable_gpu`, `enable_tpu`, and `enable_internet` all false.
  Attach the competition, the model, and `sorokin/pip-install-unsloth-flash-patch`; attach the
  code dataset once it exists.
- **Contents:** write `/kaggle/working/run_logs/env_probe.json` and `.md` with:
  - `sys.version`, `/etc/os-release`, CPU count, RAM, and `shutil.disk_usage` for
    `/kaggle/working` and `/tmp`;
  - all `KAGGLE_*` variables (`KAGGLE_IS_COMPETITION_RERUN` expected absent) and `sys.path`;
  - a depth-limited listing of `/kaggle/input` and `/kaggle/usr/lib`, with sizes;
  - `importlib.metadata.version` for torch, transformers, peft, accelerate, trl, datasets,
    unsloth, unsloth_zoo, xformers, flash_attn, triton, bitsandbytes, vllm, numpy, plus
    `pip list --format=json`. Do not `import unsloth` on CPU; it expects a GPU (**unverified**).
  - When the code dataset exists: one `pip install --no-index --find-links <resolved path> arc26`.

### P0.5 packaging

- **Resolve paths by search.** A5 step 3's `/kaggle/input/arc26-code` is the legacy layout; the
  2026 layout is `/kaggle/input/datasets/<owner>/arc26-code/`. The environment-check cell prints
  resolved paths, fails loudly in dev mode, and falls back in competition mode.
- **Dependencies.** The `arc26` wheel is pure Python. For dependencies, choose between two
  options, and never target `latest` (3.12 today, 3.13 with `transformers>=5` next):
  - (a) Reuse NVARC's stack: pin the old image and attach the install notebook at a fixed version.
    This preserves parity but depends on a third-party notebook and an old image.
  - (b) Our own internet-on CPU notebook builds cp3xx wheels for the pinned image. flash-attn
    needs a prebuilt wheel matching torch, CUDA, and Python.
- **Builder output:** the builder writes the metadata of the P0.1 section with `enable_internet`
  false explicit. Keep dataset folders flat, or pass `--dir-mode zip`.
- **Competition mode:** write only `submission.json` to `/kaggle/working`. The organizers warn
  that extra outputs can be picked in the Submit dialog (`build_notebook.py` lines 72 to 75).
  `run_logs/` stays dev-only, as A5 already says.
- **Rerun flag:** detect it with `bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))`. Commit runs
  should process only `DEV_SUBSET`, as NVARC does with DEV4; the Submit rerun does the full set on
  Kaggle's side.
- **Validator and metric:** follow Section 2 (format) and Section 2.1 (per-task score).
- **Watchdog:** NVARC stops at 12 h minus 10 min, and Kaggle reports up to 10 min of runtime
  variance. A5's 12 h minus 30 min watchdog is suitably conservative.

### A5 checklist: draft answers for the human to confirm

| Question | Draft answer | Confidence | How to confirm |
| --- | --- | --- | --- |
| Competition data mount path | `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/` | high (Kaggle-run logs) | probe listing |
| Base model mount path | `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1`; handle `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1` | high (June 2026 log) | probe listing; `kernels pull -m` of the batch4 notebook |
| Submission format | `/kaggle/working/submission.json`: task id mapped to a list (one per test input, in order) of `{"attempt_1", "attempt_2"}`; all ids; both attempts always | high | Data and Evaluation tabs |
| Daily submission limit | 1 | medium (2025 primary, 2026 snippets) | `kaggle competitions submission-limits arc-prize-2026-arc-agi-2` |
| Final submissions | 2 | medium | rules tab |
| Accelerators, commit and interactive vs scored rerun | commit: `NvidiaL4` (4xL4, 96 GB) for notebooks attached to this competition, internet off, 2x quota; also T4x2; `NvidiaL4X1` unknown; P100 retired. Rerun: 4xL4, 12 h | medium | Settings, Accelerator on an attached notebook; first GPU run's environment cell |
| Reruns count against quota | No for the rerun; yes for the Save & Run All version | low | `kaggle quota` before and after the first Submit |
| Weekly GPU quota | about 30 h, floating | low | `kaggle quota` |
| Open-source requirement (G5) | own code CC0 or MIT-0; third-party code under OSS that allows sharing; published before private scores; Grand Prize writeup within 7 days of 2 Nov | high (arcprize.org) | also read the Kaggle rules' winner-license clause |
| Reference notebook license | likely Apache 2.0 (Kaggle default); NVARC GitHub has none | low | footer of the batch4 notebook page |
| Kaggle username | needed for every kernel `id` | | human |

## Open risks

1. **Environment drift.** Kaggle's image source went to Python 3.13 with `transformers>=5`
   yesterday (docker-python `e096cbc`); deployment timing is unknown. NVARC's stack needs the old
   Python 3.11 image to stay pinnable and the third-party install notebook to stay unchanged, so
   pin its version.
2. **Parity may be unreachable as written.** The P0.1 acceptance "identical attempt grids" can
   fail between two runs of the untouched notebook itself (batched DFS, GPU kernels, time caps).
3. **Quota arithmetic.** At 2x, DEV4 costs about 1 quota hour. DEV120 needs about
   $120 \times 720\,\text{s} / 4 \approx 6$ h wall, about 12 quota hours (estimate). With about
   30 h per week (**unverified**), that is at most two DEV120 runs a week with nothing else.
   Re-plan P1.1, P2.6, P4.4, and meta-training against `kaggle quota` before G1.
4. **L4 policy.** L4 use is limited to notebooks attached to this competition, and circumventing
   that risks a ban. Whether offline meta-training on L4 is acceptable should be settled at P0.4.
5. **Scoring wording conflicts** with the observed per-task metric (Section 2.1).
6. **Mount layout moved within 2026** (the model path changed between April and June), so nothing
   may hardcode `/kaggle/input` paths.
7. **Reference identity and license:** the batch4 notebook versus the `trm2` fork, and the
   notebook and model licenses, are unread (JS pages).
8. **Gateway gaps** (Section 3.7), especially the `machine_shape` bypass of the brief check and the
   missing `--timeout`, given there is no CLI cancel.
9. **Rerun-only code paths** (test file, all 240 tasks) run for the first time after a Submit that
   spends the daily submission. Exercise the rerun branch in commit runs on the placeholder test
   file with a small subset.
10. **kaggle.com pages were unreadable**, so every rules-level fact here comes from arcprize.org,
    code, or third-party copies. The human's reading of the rules tab is the final word.
