# Module: `arc26/agent/sandbox.py`, `guards.py`, `verify.py`, `lib/`

Task P6.1 (ADR-020; design note `docs/research/agent_library_design.md` sections 3.4, 3.5, 4,
5.1, 5.3). Tests: `tests/agent/test_sandbox.py`, `test_guards.py`, `test_verify.py`,
`test_lib.py`. CPU only, Linux only, no network. Contract: `arc26/agent/types.py` (`Repl`,
`Verifier`, `CellResult`, `GuardFinding`, `PairResult`, `Verification`).

## Purpose

The agent solver runs Python that a language model writes. This module set runs that code without
letting it touch the solver, judges programs only by what they do on the task's grids, and gives
the model a tested numpy helper library:

- `SubprocessRepl` (`Repl`): a persistent per-attempt REPL for exploration, with the task's grids
  loaded.
- `PoolVerifier` (`Verifier`): re-runs a self-contained program in a fresh, least-privilege
  process on every demonstration input and test input, and reports exactness per pair.
- `guards`: the AST safety policy and the hardcoding guards, as `GuardFinding`s with stable rule
  names.
- `lib`: the frozen helper library v0 (45 functions and `Object`), preloaded as `lib`.

Model code never runs in the solver's process. Only verified grids leave the agent (ADR-020's
waiver of BRIEF A0 for Phase 6).

## Process layout

```text
solver process (trusted; asyncio runner, P6.2)
  guards.check_policy / check_program        AST only, before anything runs
  SubprocessRepl  ->  python -B -S -c <bootstrap> arc26.agent.sandbox <roots>   one per attempt
  PoolVerifier    ->  python -B -S -c <bootstrap> arc26.agent.verify <roots>  x 8  workers:
                        numpy, lib, builtins loaded once
                        fork per program  ->  program child: fresh namespace {np, lib}, own
                                              process group, dies with its worker (PDEATHSIG)
```

Every sandbox process starts from `spawn_command(module)`: the parent's interpreter without the
`site` module (`-S`: no site folders, `.pth` files, or `sitecustomize`) and without bytecode
writes (`-B`); a bootstrap sets its import path to the standard library followed by
`import_roots()`, the folders the parent imports `arc26`, `numpy`, and `scipy` from (found with
`importlib.util.find_spec`, in the parent's `sys.path` order), then runs the module as
`__main__`. The standard library always comes first: run `P6.3-02` (12 Oct) passed the parent's
site folder as `PYTHONPATH`, which goes before the standard library, and Kaggle's image keeps the
obsolete `pathlib` 1.0.1 backport there, which fails to import on Python 3.10 and later, so every
REPL and verifier worker died on import.

Messages are JSON lines on private pipes (the child moves the protocol off fds 0 and 1, which then
point at `/dev/null`), size-capped and validated by the receiver; nothing is pickled, so a
compromised child cannot run code in the parent through the protocol.

## Layers between model code and the machine

| Layer | Where | What it stops |
|---|---|---|
| AST policy (`check_policy`) | parent, before sending | imports outside the allowlist, `open`, `eval`, `exec`, `compile`, `__import__`, `globals`, `vars`, `locals`, `input`, `breakpoint`, dunder access (except `__name__`, `__doc__`, `__init__`), attributes that reach files, processes, frames, native code, or eval (`np.load`, `tofile`, `np.ctypeslib`, `typing.sys`, `get_type_hints`, ...), the same inside strings that parse as code |
| Restricted builtins | child | the forbidden builtins are absent; `__import__` applies the allowlist (covers imports inside library eval paths); `getattr`, `hasattr`, `setattr`, `delattr` refuse names built at run time |
| Audit hook (PEP 578) | child, after startup, irreversible | `os.fork`, `os.system`, `os.exec*`, `posix_spawn`, `subprocess.Popen`, `os.kill`, sockets, `ctypes`, `mmap`, file writes and filesystem changes, `resource.setrlimit`, `gc.get_objects`, code and function construction, unpickling; reads (`open`, `os.listdir`, `os.scandir`) only below the Python installation and `sys.path` |
| Resource limits | child | `RLIMIT_AS` 2 GiB (`MemoryError`, reported as `"memory"`), `RLIMIT_FSIZE` 0, `RLIMIT_CORE` 0, `RLIMIT_NPROC` 0 (binds non-root users; the hook covers root), `RLIMIT_CPU` per program child as a backstop |
| Wall time | parent (REPL) and worker (verifier) | REPL: kill the process group, start a fresh child with the grids reloaded. Verifier: kill the program child, mark the call `"timeout"`, continue with the next input in a new child |
| Environment | spawn | minimal: no inherited variables except `LD_LIBRARY_PATH`, no `PYTHONPATH`; empty temporary working and home directory; `PYTHONHASHSEED=0`, one BLAS thread; the import path is the standard library and `import_roots()` only (no `site`) |
| Data | verifier | the worker never receives demonstration outputs (the parent compares), so even escaped code cannot read the answers from memory |

The Python-level layers (policy, builtins) are best effort: Python cannot be sandboxed from
within. The process-level layers (hook, limits, process isolation, data separation) bound what an
escape could do. `policy=False` turns the Python-level layers off; only the tests use it, to show
that the process-level layers hold on their own.

## Interfaces

```python
SubprocessRepl(task, *, memory_bytes=2 * GIB, max_output_chars=8_000, policy=True,
               preload=("scipy.ndimage",), startup_timeout_s=60.0)
    .run_cell(code, timeout_s) -> CellResult
    .close(); context manager; .pid, .restarts

PoolVerifier(workers=8, *, call_timeout_s=2.0, memory_bytes=2 * GIB, policy=True,
             preload=("scipy.ndimage",), startup_timeout_s=60.0, hang_grace_s=10.0)
    .verify(task, source) -> Verification       thread-safe
    .close(); context manager; .worker_pids(), .replaced

sandbox.spawn_command(module) -> list[str]     the child's command line (see Process layout)
sandbox.import_roots(packages=("arc26", "numpy", "scipy")) -> list[str]

guards.check_policy(source) / check_hardcoding(source, task) / check_program(source, task)
guards.check_outputs(task, test_outputs) / defines_transform(source)
guards.RULES, SAFETY_RULES, FATAL_RULES, ALLOWED_MODULES, FORBIDDEN_BUILTINS, FORBIDDEN_ATTRIBUTES

lib.<helper>(...); lib.index_text() -> str; lib.doc(name) -> str; lib.__all__
```

### REPL

The namespace starts with `train` (a list of `(input, output)` tuples of `numpy.int64` arrays),
`test_inputs` (a list of arrays), `np`, and `lib`; `__name__` is `"__main__"`. A cell whose last
statement is an expression prints its `repr`. `CellResult.error` is the exception's last traceback
line, `"timeout"` (child killed and restarted, state lost, grids reloaded), `"memory"` (state
kept), `"PolicyError: <rule>: <detail>"` or `"SyntaxError: ..."` (the cell did not run), or
`"crashed: <signal or exit code>"` (restarted). Output keeps the first 6,000 and last 2,000
characters of a cell's output, with the cut marked, so tracebacks survive. `defines_transform`
comes from the AST (a top-level `def`, assignment, or import alias of `transform`).

A REPL serves one attempt and is never shared between threads; an internal lock only serializes
accidental concurrent calls.

### Verifier

`transform` receives each input as a fresh 2-D `numpy.int64` array (as in the REPL) and returns an
array or nested lists; the program's namespace starts with `np` and `lib`, `__name__` is
`"program"`, and its top-level code runs once per program child under the call time limit. Valid
outputs are rectangular, 1 to 30 cells per side, integer colors 0 to 9 (integral floats accepted,
booleans not); anything else is `"invalid output: <reason>"`. Per-pair `error` values: an
exception's last line, `"timeout"`, `"timeout (top-level code)"`, `"memory"`, `"crashed: ..."`,
`"invalid output: ..."`, `"invalid result"` (a malformed reply), `"not run: <rule>"` (a safety
finding), `"worker failure: ..."` (the worker died or hung and was replaced). `cell_accuracy` is
the fraction of equal cells when shapes match, else 0.

Thread-safety: `verify` may run from many threads at once (P6.2 calls it through
`asyncio.to_thread`, up to 64 attempts). Each call checks a worker out of a queue for its
exclusive use and keeps all per-call state local; the live-worker set and the replacement count
change only under a lock. A crashed, hung, or misbehaving worker is killed and replaced, so the
pool keeps its size; `close()` releases waiting callers and creates no worker afterwards.

## Guard rules

| Rule | Class | Catches | Notes |
|---|---|---|---|
| `syntax_error` | safety, not run | source does not parse | |
| `source_too_large` | safety, not run | more than 100,000 characters | |
| `forbidden_import` | safety, not run | import outside numpy, scipy, math, itertools, collections, functools, copy, typing, `arc26.agent.lib`; relative and star imports; `import arc26.agent.lib` without an alias (it binds `arc26`) | `numpy.ctypeslib`, `numpy.testing`, `scipy.io` are refused by component name |
| `forbidden_builtin` | safety, not run | the forbidden builtins | a name the program binds itself (a parameter called `input`) is its own |
| `dunder_access` | safety, not run | dunder attributes and names, `getattr` with a dunder string, `case C(__class__=x)` | |
| `forbidden_attribute` | safety, not run | the attributes listed in `FORBIDDEN_ATTRIBUTES` | |
| `literal_output` | hardcoding, fatal, runs | a grid literal equal to a whole demonstration output; a grid literal containing a whole output as a block (padded, cropped in code); a flat literal containing an output's row-major or column-major flattening as a run (extra values, reshaped in code) | containment counts for non-uniform outputs of at least 9 cells; a smaller output only when a literal equals it, since it can sit inside a kernel or a color table by chance |
| `constant_output_literal` | demotion | any of these, when at least two demonstrations share that one output (the literal may be the rule) | |
| `partial_output_literal` | demotion | a literal holding a 3 by 3 window of a demonstration output, for example a stencil the program stamps | windows that are uniform or 0/1-valued are ignored (masks, kernels) |
| `input_literal_compare` | hardcoding, fatal, runs | a grid literal equal to any task input; `==`, `array_equal`, `array_equiv`, `allclose` between the whole input and a grid literal | the input is tracked through copies and conversions into helpers it is passed to |
| `input_hash_key` | hardcoding, fatal, runs | `hash()` of the input, `.tobytes()` of the whole input, or the input's bytes, string, or tuple form used as a key, a membership test, or compared with a constant | `tuple(row)` keys, sub-grid bytes, prints are not flagged |
| `shape_dispatch` | demotion | comparing the input's height, width, shape, or size with a value that is a demonstration or test input size; `match` on the shape; dict lookups keyed by such shapes | |
| `output_equals_input` | demotion | a test output equal to its input while no demonstration output equals its input | |
| `output_equals_demo` | demotion | a test output equal to a demonstration output, unless all demonstration outputs are equal | |
| `novel_colors` | demotion | a test output color absent from the demonstration outputs and the test input | |

Programs with a hardcoding finding still run, so the agent's report can show what they did; they
are never `verified`. A demoted program can be `verified`; selection ranks it below undemoted ones.

## Library

Conventions: grids are 2-D integer arrays (nested lists accepted), coordinates `(row, col)`,
boxes `(top, left, bottom, right)` with bottom and right exclusive, masks boolean arrays; every
helper returns new arrays. Helpers: `palette`, `color_counts`, `most_common_color`,
`least_common_color`, `background_color`, `recolor`, `mask_of`, `fill`, `coords`, `dilate`,
`outline`, `flood_fill`, `fill_enclosed`, `objects` (returning `Object`: color, colors, size, bbox,
mask, patch, top, left, height, width, cells, shape_mask), `bbox`, `crop`, `crop_to_content`,
`adjacent`, `paint`, `move`, `rot90`, `rot180`, `rot270`, `flip_lr`, `flip_ud`, `transpose`,
`anti_transpose`, `dihedral`, `upscale`, `downscale`, `tile`, `hconcat`, `vconcat`, `shift`,
`gravity`, `is_symmetric`, `complete_symmetry`, `find_period`, `complete_tiling`, `split_grid`,
`hsplit`, `vsplit`, `draw_line`, `draw_rect`, `ray`. Written from scratch after the operations
Hodel's DSL solvers use most (MIT; ideas, not code). `index_text()` is the prompt index: 47 lines,
about 4,000 characters (about 1,000 tokens); `doc(name)` returns a helper's full docstring, since
the policy refuses most dunder access. The helpers use numpy and the standard library only (a
test fixes the imports), so they behave the same here and on Kaggle; the module binds no module
but `np`, so it opens no path to `os` or `sys`.

## Optional scipy

Programs may import scipy (the allowlist has it); the library does not use it. Both sandboxes
import `scipy.ndimage` at startup when it is installed (`DEFAULT_PRELOAD`), before the audit hook
and the process limit, so its import cannot trip either and a program behaves the same in the REPL
and the verifier. scipy is absent on the dev machine and in CI, so the two tests that use it skip
there; the preload mechanism itself is tested with a standard-library module.

## Measurements (this machine, 16 CPUs, 9 Oct 2026)

Trivial program (`np.rot90`), 3 demonstrations and 1 test input of 10 by 10, as many threads as
workers, median of 3 runs of 400 verifications (`runs/P6.1/throughput.txt`):

| Workers | 1 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| Verifications per second | 237 | 813 | 1,134 | 1,156 |

Target: at least 50 per second with 8 workers. One verification costs a fork, the program, and the
JSON round trips (about 4 ms). Closing inherited descriptors by listing `/proc/self/fd` instead
of `os.closerange` up to `RLIMIT_NOFILE` (1,048,576 here; the venv's interpreter lacks
`close_range(2)`) took a verification from 250 ms to 4 ms. A REPL starts in about 80 ms.

## Tests

`tests/agent/`: 325 tests (2 skip without scipy), about 15 s. Every guard rule has hostile programs
it must catch and ordinary programs it must pass; the REPL's limits (timeout with restart and grids
reloaded, memory, endless output, huge output, crash) and its containment with the policy off (file
read and write, directory listing, fork, fork bomb, `os.system`, subprocess, socket, ctypes, `kill`
of the parent, `setrlimit`, the `typing.get_type_hints` eval path); the verifier's results on
correct, wrong, and partially correct programs, every invalid output kind, per-call and top-level
timeouts, memory, policy rejections that never run, hardcoded programs that pass the demonstrations
but are refused (plain and padded), a stencil program and a constant-output program that verify with
a demotion, garbage on the result pipe, a crashing program, killed and stopped workers, 48 mixed
correct, wrong, and hanging programs from 32 threads (results checked, no leaked worker or program
process), close while callers wait, statelessness between programs, that demonstration outputs never
reach a worker, and throughput; every library helper (one case each, plus its docstring example,
purity, signature, and the index budget).

## Known limits

- The Python-level layers can be escaped by a determined program; the containment then rests on
  the audit hook, the limits, and process separation. Native code reached through a numpy memory
  bug could bypass the hook; `as_strided` is refused for that reason. A namespace or Landlock
  sandbox would close this; Kaggle's support for either is unverified.
- `RLIMIT_NPROC` does not bind root (Kaggle notebooks may run as root); there the audit hook alone
  stops forks.
- The hook allows reading anything below the Python installation and `sys.path`; in an editable
  install that includes the repository.
- `literal_output` misses hardcoding encoded as strings or arithmetic, and a padded copy of an
  output under 9 cells. A partial block only demotes, so a program that embeds output fragments of
  at least 3 by 3 can still verify.
- Shape tracking follows the input through simple assignments and calls to top-level functions,
  not through containers or attributes.
- The REPL loses a timed-out cell's partial output.
