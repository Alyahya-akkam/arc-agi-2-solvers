# `arc26/runner.py`: launch the solver, write the submission

Task P0.1 (BRIEF B5 P0.1), ADR-009; timing records, task P0.3b; LoRA scope, task P1.3; the
solver's code, task P2.0 (ADR-011); the hold-out settings, task P2.1 (ADR-013); the rounds inside
them, task P2.3 (ADR-015). Our code, not
moved: it replaces
the reference notebook's launch line (cell 6) and calls its selection cell (cell 7,
`arc26.nvarc.select`). The P0.5 notebook uses this interface; keep names, parameters, and
defaults stable.

## Purpose

The reference hard-codes its paths, the DEV4 filter, four workers, and its LoRA rank and target
modules, and launches `starter.py` from a shell line. `RunConfig` makes those values run settings,
and the functions launch the solver the way the notebook did, so dev runs can use one GPU, a DEV
subset, or another LoRA scope while the submission run keeps the reference's values. `solver`
picks the solver's code: the reference's statements split into `arc26.ttt` and `arc26.decode`
(`arc26.solver`, ADR-011, the default since run `P2.0-01` passed, 2 Oct) or the moved reference
(`arc26.nvarc.starter`, the parity anchor). `holdout` holds
one demonstration pair per puzzle out of training and prompts and runs the held-out gate
(`arc26.loop`, `docs/modules/loop.md`), on the split solver only. `adapters` (task P3.7, ADR-019)
runs the split solver's arms entry, `arc26.arms` (`docs/modules/arms.md`), instead of
`arc26.solver`: every puzzle under several adapter sources, paired.

## Interface

```python
@dataclass(frozen=True)
class RunConfig:
    data_dir: str                                    # competition data directory
    model_dir: str                                   # base model directory (holds config.json)
    outputs_dir: str = "/kaggle/inference_outputs"   # candidate pickles, reference default
    marker_dir: str = "/kaggle"                      # worker{rank} marker files, reference default
    subset: tuple[str, ...] | None = None            # None: every task
    workers: int = 4                                 # one per GPU
    hash_seed: int = 0                               # PYTHONHASHSEED of the solver process
    timing_dir: str | None = None                    # JSONL timing records; None: none
    lora_r: int = LORA_R                             # 256, the reference's LoRA rank
    lora_targets: tuple[str, ...] = LORA_TARGETS     # the reference's nine target modules
    solver: str = "split"                            # a key of SOLVER_MODULES
    holdout: HoldOut | None = None                   # arc26.loop.HoldOut, its optional rounds
                                                     # included; None: no hold-out
    adapters: Arms | None = None                     # arc26.arms.Arms (task P3.7); None: one
                                                     # adapter source, the reference's fit

SOLVER_MODULES = {"nvarc": "arc26.nvarc.starter", "split": "arc26.solver"}
ARMS_MODULE = "arc26.arms"                           # started instead when adapters is set

def solver_env(cfg: RunConfig, base: Mapping[str, str]) -> dict[str, str]
def solver_command(cfg: RunConfig, end_time: float, python: str,
                   module: str | None = None) -> list[str]
def run_solver(cfg: RunConfig, end_time: float, python: str | None = None,
               timeout: float | None = None, module: str | None = None) -> int
def make_submission(cfg: RunConfig, submission_path: str) -> None
```

- `module` (task P4.7b): the entry module, None for `SOLVER_MODULES[cfg.solver]`; the notebook's
  rank-256 tail passes `"arc26.tail"` (`docs/modules/tail.md`), which takes the same arguments.
- `RunConfig` validates in `__post_init__`: directories are non-empty strings (`timing_dir` may
  also be None); `subset` is None or a non-empty list or tuple of eight-lowercase-hex task ids (a
  list is stored as a tuple); `workers` is an integer $\ge 1$; `hash_seed` is an integer in
  $[0, 2^{32} - 1]$, the range Python accepts. `TypeError` or `ValueError` otherwise. `subset`
  applies only outside a competition rerun; a rerun always solves every task, as in the
  reference. `timing_dir` (P0.3b) is where the solver's workers write
  `timing-rank<rank>.jsonl` and `make_submission` writes `timing-main.jsonl`
  (`docs/modules/timing.md`); None records nothing, as the reference. `lora_r` (P1.3) is an
  integer $\ge 1$; `lora_targets` is a list or tuple of distinct names from `LORA_TARGETS`
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`,
  `embed_tokens`, `lm_head`, the reference's order), stored as a tuple, with at least one of
  the seven projections (`LORA_PROJECTIONS`): Unsloth trains `embed_tokens` and `lm_head` as
  full copies, and PEFT refuses a LoRA config that matches no module. The solver keeps the
  reference's $\alpha = 32$ with rsLoRA, so the adapter's scale is $32 / \sqrt{r}$. `solver`
  (P2.0) is a key of `SOLVER_MODULES`: `"split"`, the default since run `P2.0-01` passed and the
  human approved the flip (2 Oct, `docs/experiments/P2.0-01.md`), runs the reference's statements
  as `arc26.ttt.fit` and `arc26.decode.decode_puzzle` (`docs/modules/solver.md`); `"nvarc"` runs
  the moved reference. A value that is
  not a string raises `TypeError`, any other key `ValueError`. `holdout` (P2.1) is None, the
  default, or `arc26.loop.HoldOut` settings (a mapping of its fields is converted): the gate's
  candidate folder `gate_dir`, the verdict folder `log_dir` (or None), the rule (`"last"` or
  `"first"`), `k2`, `in_test_prompts`, and (task P2.3) `rounds`, `arc26.loop.Rounds` or a mapping
  of its fields: the round folder `round_dir`, the parsed governor config, `max_rounds` (1 or 2),
  `tau`, `continue_steps` ($s'$), and `final_in_prompts` (default true: the final decode with the
  held-out pair, ADR-013 as amended); rounds need `in_test_prompts` False. Set,
  it needs `solver` `"split"` (the moved reference trains and decodes in one loop body, ADR-011),
  and none of its folders (the gate's, the verdicts', the rounds') may lie inside `outputs_dir`,
  whose every file the selection reads; `TypeError` or `ValueError` otherwise.
- `solver_env` returns `base` plus the reference's launch variables
  (`UNSLOTH_DISABLE_STATISTICS=1`, `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`,
  `OMP_NUM_THREADS=12`) and `PYTHONHASHSEED=str(hash_seed)`, which override `base`.
- `solver_command` returns
  `[python, "-m", SOLVER_MODULES[cfg.solver], "--end-time", repr(end_time), "--config", json]`:
  `arc26.nvarc.starter` or `arc26.solver`, with the same arguments. The JSON holds exactly
  `data_dir`, `model_dir`, `outputs_dir`, `marker_dir`, `subset` (a list, or null), `workers`,
  `timing_dir` (a folder, or null), `lora_r`, and `lora_targets` (a list): the keys the moved
  code reads, and the split solver too; plus `holdout` (`HoldOut.to_json()`) and `adapters`
  (`Arms.to_json()`, task P3.7) only when set, so a run without them passes exactly the JSON of
  before; with `adapters` the module is `ARMS_MODULE`; its `rounds` appear inside it only
  when set, so a hold-out run without rounds passes task P2.1's JSON. `solver` is not in the JSON: it
  picks the module, as `hash_seed` reaches the solver through its environment.
  `end_time` is an absolute
  `time.time()` deadline; a non-finite value raises `ValueError`.
- `run_solver` runs that command in a new session with `solver_env(cfg, base)`, where `base` is
  `os.environ` plus `PYTHONUNBUFFERED=1`. It copies the solver's stdout and stderr to `sys.stdout` as they arrive and returns
  the exit code. After `timeout` seconds, or on any exception such as `KeyboardInterrupt`, it
  kills the whole process group (the solver and every worker it spawned) with `SIGKILL`; a
  timeout returns $-9$. `python` defaults to `shutil.which("python")`, the interpreter the
  reference's `!python starter.py` used; `FileNotFoundError` when there is none.
- `make_submission` creates `outputs_dir` if missing, then calls
  `arc26.nvarc.select.write_submission(data_dir, outputs_dir, submission_path)`. With
  `timing_dir` set, that call (not the import before it) is recorded as op `main_selection` with
  a null `puzzle_id` and null tokens in `timing_dir/timing-main.jsonl`, through a
  `TimingLogger` of its own, so the calling process keeps no process-wide logger; a call that
  raises writes no record. With `timing_dir` None it is the P0.1 call, unchanged.

## Data flow

```text
RunConfig -> run_solver -> python -m arc26.nvarc.starter --end-time T --config JSON   (nvarc)
                           or python -m arc26.solver --end-time T --config JSON         (split)
                           -> workers x local_worker -> outputs_dir/<view> (bz2 pickles)
                                                     -> timing_dir/timing-rank<rank>.jsonl
RunConfig -> make_submission -> select.write_submission -> submission_path
                             -> timing_dir/timing-main.jsonl
```

A P0.5 notebook cell does, for example:

```python
import subprocess
import time
from arc26.runner import RunConfig, make_submission, run_solver

end_time = time.time() + 12 * 3600 - 600          # the reference's budget (cell 0)
# Cell 1 of the reference, which run A ran (tensorflow 2.18.0 removed); skipping it changes the
# environment that P0.1's parity rule compares.
subprocess.run(["python", "-m", "pip", "uninstall", "-y", "tensorflow"], check=False)
cfg = RunConfig(
    data_dir="/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
    model_dir="/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
    subset=("0934a4d8", "36a08778", "981571dc", "aa4ec2a5"),   # DEV4; None for every task
)
code = run_solver(cfg, end_time, timeout=end_time + 300 - time.time())   # always pass a timeout:
# run_solver waits for the output pipe to close
make_submission(cfg, "/kaggle/working/submission.json")
```

## Origin

- Cell 6: `!UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
  OMP_NUM_THREADS=12 python starter.py --end-time {global_end_time}`, run by IPython, which gives
  the process a terminal (line-buffered output) and waits for it.
- Cell 7: the selection, moved as `arc26.nvarc.select.write_submission`.
- `hash_seed`: run A of `P0.1-01` fixed `PYTHONHASHSEED=0` (flagged conflict 5,
  `docs/experiments/P0.1.md`).

## Tests

`tests/test_runner.py`: the dataclass fields, defaults, and function signatures of the interface;
the command, its JSON (round trip back to an equal `RunConfig`, and the same keys as the moved
code reads), the environment and its launch variables against cell 6; validation errors;
`run_solver` with a fake interpreter (exit code, streamed and merged output, `python` from
`PATH`, timeout kill of the process group returning $-9$, kill on an exception);
`make_submission` creating a missing `outputs_dir`; `timing_dir` in the JSON (null by default)
and refused when empty or not a string; `make_submission` writing one `main_selection` record
in both modes without touching the process-wide logger, none without `timing_dir`, and none
when the selection raises; importing the runner imports no GPU stack; the LoRA defaults are the
reference's, a scope reaches the solver's JSON, and the LoRA validation errors; `solver`
defaults to `"split"`, which starts `arc26.solver` with the moved starter's arguments (also through
`run_solver`), the split modules read exactly the JSON's keys, both modules are process entries,
and the `solver` validation errors; `holdout` absent from the JSON by default, its round trip
through the JSON, `run_solver` passing it, the split solver as its only reader
(`HoldOut.from_cfg`), and its validation errors (the `nvarc` solver, bad settings, a gate folder
inside `outputs_dir`).
`tests/test_lora_scope.py` evaluates the solver's `peft_params` with the JSON's cfg.
`tests/test_replay.py` runs `make_submission` on run A's candidates.

## Known limits

- POSIX only: `run_solver` uses a new session, `os.killpg`, and a selector on the pipe (Kaggle
  and CI are Linux).
- `run_solver` adds `PYTHONUNBUFFERED=1` to the base environment, so output arrives when printed
  and survives a kill; `solver_env` itself returns exactly the reference's variables.
- `make_submission` needs `transformers` (the moved `arc_loader` imports it) and, in dev mode,
  raises `ValueError` from the reference's `benchmark_selection_algos` when no candidate is
  correct, after writing the submission.
- `outputs_dir`, `marker_dir`, and `timing_dir` are not cleaned between launches; use fresh
  folders for a second launch in one session (`docs/modules/nvarc.md`).
