"""The timing wiring of the moved solver (task P0.3b): where each op is recorded, with which token
count, and that a null ``timing_dir`` records nothing.

``arc_solver.py`` and ``starter.py`` import torch and unsloth, which CPU tests do not have, so the
wiring is checked twice. First on the AST of the generated modules: each op's span starts right
before the statements it times and is recorded right after them, the train, decode, and selection
spans lie inside the puzzle span, nothing in a puzzle can skip its record, and every record call
is guarded by the worker's logger, which only ``starter.local_worker`` creates, and only for a
non-null ``timing_dir``. Then by running ``starter.local_worker`` (and with it
``arc_solver.worker``) in a subprocess against stand-in modules for torch, unsloth, datasets,
transformers, peft, and tqdm, with a one-token-per-character tokenizer and deterministic fakes for
the reference's DFS (``inference_turbo_dfs``) and rescoring (``calc_scores``). The two synthetic
tasks are never competition data. With ``timing_dir`` null nothing but the candidates and the
marker is written; with it set every op is recorded with the token counts the brief defines; a
puzzle cut by the 1200 s cap keeps its record; and the candidates and numpy's random state are the
same with and without timing.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parent.parent
NVARC = REPO / "arc26" / "nvarc"
GUARD = "timing_logger is not None"


def module_tree(name):
    return ast.parse((NVARC / name).read_text(encoding="utf-8"))


def function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def same(node, source):
    """Whether a statement or expression node is the given source, layout aside."""
    expected = (
        ast.parse(source, mode="eval").body
        if isinstance(node, ast.expr)
        else ast.parse(source).body[0]
    )
    return ast.dump(node) == ast.dump(expected)


def blocks(node):
    """Every statement list below node: bodies, else branches, and finally blocks."""
    for child in ast.walk(node):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(child, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def is_guard(node):
    return isinstance(node, ast.If) and same(node.test, GUARD) and not node.orelse


class Span(NamedTuple):
    """One op's span: in ``block``, ``<op>_start = time.perf_counter()`` at index ``start`` and
    the ``if timing_logger is not None:`` that records it at index ``guard``."""

    block: list
    start: int
    guard: int

    @property
    def timed(self):
        return self.block[self.start + 1 : self.guard]

    @property
    def record(self):
        return self.block[self.guard]


def span(worker, start):
    found = []
    for block in blocks(worker):
        for i, statement in enumerate(block):
            if same(statement, f"{start} = time.perf_counter()"):
                guard = next(j for j in range(i + 1, len(block)) if is_guard(block[j]))
                found.append(Span(block, i, guard))
    assert len(found) == 1, f"{start}: {len(found)} spans"
    return found[0]


def escapes(statements):
    """Statements that would leave the enclosing loop body: a return, or a break or continue
    outside every loop nested in the statements."""
    found = []

    def visit(node, in_loop):
        if isinstance(node, ast.Return) or (
            isinstance(node, ast.Break | ast.Continue) and not in_loop
        ):
            found.append(ast.unparse(node))
        if isinstance(node, ast.FunctionDef | ast.Lambda):
            return
        for child in ast.iter_child_nodes(node):
            visit(child, in_loop or isinstance(node, ast.For | ast.While))

    for statement in statements:
        visit(statement, False)
    return found


@pytest.fixture(scope="module")
def solver():
    return module_tree("arc_solver.py")


@pytest.fixture(scope="module")
def worker(solver):
    return function(solver, "worker")


@pytest.fixture(scope="module")
def puzzle_loop(worker):
    return next(statement for statement in worker.body if isinstance(statement, ast.While))


# Where each op is recorded (AST of the generated arc_solver.py).


def test_the_worker_fetches_its_logger_once_before_the_load(worker):
    assigned = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "timing_logger" for t in node.targets)
    ]
    assert len(assigned) == 1 and same(assigned[0], "timing_logger = timing.get_logger()")
    load = span(worker, "load_start")
    assert load.block is worker.body and load.block[load.start - 1] is assigned[0]


def test_load_times_the_model_load_before_any_puzzle(worker, puzzle_loop):
    load = span(worker, "load_start")
    timed = load.timed
    assert len(timed) == 5
    assert ast.unparse(timed[0]).startswith("model, tokenizer = FastLanguageModel.from_pretrained(")
    assert same(timed[1], "model = FastLanguageModel.get_peft_model(model, **peft_params)")
    assert isinstance(timed[2], ast.For) and "torch.bfloat16" in ast.unparse(timed[2])
    snapshot = 'default_weights = get_peft_model_state_dict(model, adapter_name="default")'
    assert same(timed[3], snapshot)
    copy = "default_weights = {k: v.clone().detach() for k, v in default_weights.items()}"
    assert same(timed[4], copy)
    (record,) = load.record.body
    assert same(record, 'timing_logger.record(None, "load", time.perf_counter() - load_start)')
    assert load.guard < worker.body.index(puzzle_loop)


def test_puzzle_spans_the_whole_puzzle_and_nothing_skips_its_record(worker, puzzle_loop):
    puzzle = span(worker, "puzzle_start")
    assert puzzle.block is puzzle_loop.body
    assert same(puzzle.block[puzzle.start - 1], "start_time = time.time()")
    assert same(puzzle.timed[0], "torch.cuda.reset_peak_memory_stats()")
    assert same(puzzle.timed[-1], 'print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")')
    assert puzzle.guard == len(puzzle_loop.body) - 1
    (record,) = puzzle.record.body
    expected = "timing_logger.record(key, timing.PUZZLE, time.perf_counter() - puzzle_start)"
    assert same(record, expected)
    # The 1200 s cap and the end time break out of the decode loop only, so a cut puzzle still
    # reaches its record; only an exception (a crashed worker) skips it.
    assert escapes(puzzle.timed) == []
    cap = [
        node
        for node in ast.walk(puzzle_loop)
        if isinstance(node, ast.If) and "spend_time > 1200" in ast.unparse(node.test)
    ]
    assert len(cap) == 1 and isinstance(cap[0].body[-1], ast.Break)


def test_train_times_trainer_train_and_counts_the_collator_input_ids(worker):
    train = span(worker, "train_start")
    assert [ast.unparse(s) for s in train.timed] == ["stats = trainer.train()"]
    # Inside the reference's output redirection, so nothing the count does reaches the log.
    owner = next(n for n in ast.walk(worker) if isinstance(n, ast.With) and n.body is train.block)
    assert "redirect_stdout(buf)" in ast.unparse(owner.items[1])
    seconds, attempt, record = train.record.body
    assert same(seconds, "train_seconds = time.perf_counter() - train_start")
    assert isinstance(attempt, ast.Try) and not attempt.orelse and not attempt.finalbody
    (count,) = attempt.body
    assert same(count, 'train_tokens = sum(map(len, trainer.train_dataset["input_ids"]))')
    (handler,) = attempt.handlers
    assert same(handler.type, "Exception")
    assert [ast.unparse(s) for s in handler.body] == ["train_tokens = None"]
    assert same(record, "timing_logger.record(key, timing.TRAIN, train_seconds, train_tokens)")


def test_decode_times_each_dfs_call_with_its_prompt_tokens(worker):
    decode = span(worker, "decode_start")
    assert [ast.unparse(s) for s in decode.timed] == [
        "dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)"
    ]
    (record,) = decode.record.body
    assert same(
        record,
        "timing_logger.record(key, timing.DECODE, time.perf_counter() - decode_start, "
        "sum(map(len, tokens)))",
    )
    # tokens holds the batch's prompts here: built just before, rebound only later by the beams.
    built = decode.block[decode.start - 1]
    assert isinstance(built, ast.For)
    assert "tokens.append(tokenizer.encode(data['input']))" in ast.unparse(built)
    beams = decode.block[decode.guard + 1]
    assert isinstance(beams, ast.For) and same(beams.iter, "dfs_result")


def test_selection_times_each_pair_of_calc_scores_calls(worker):
    selection = span(worker, "selection_start")
    assert [ast.unparse(s) for s in selection.timed] == [
        "augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)",
        "augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)",
    ]
    seconds, count, record = selection.record.body
    assert same(seconds, "selection_seconds = time.perf_counter() - selection_start")
    assert same(
        count,
        "selection_tokens = sum(len(tokenizer.encode(text)) for text in aug_queries + aug_answers)",
    )
    assert same(
        record, "timing_logger.record(key, timing.SELECTION, selection_seconds, selection_tokens)"
    )
    after = selection.block[selection.guard + 1]
    assert same(after, "augmented_scores = augmented_scores1 + augmented_scores2")


def test_train_decode_and_selection_lie_inside_the_puzzle_span(worker):
    puzzle = span(worker, "puzzle_start")
    first, last = puzzle.block[puzzle.start].lineno, puzzle.record.lineno
    for start in ("train_start", "decode_start", "selection_start"):
        inner = span(worker, start)
        assert first < inner.block[inner.start].lineno < inner.record.end_lineno < last


def test_every_record_is_guarded_by_the_worker_logger(solver):
    parents = {child: node for node in ast.walk(solver) for child in ast.iter_child_nodes(node)}
    records = [
        node
        for node in ast.walk(solver)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record"
    ]
    assert len(records) == 5
    for call in records:
        assert same(call.func.value, "timing_logger")
        node = call
        while not is_guard(node):
            node = parents[node]  # a KeyError at the module root fails the test
    # The solver uses the timing module only to fetch its logger and for the op names: it never
    # configures a logger or writes a file itself.
    used = {
        node.attr
        for node in ast.walk(solver)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "timing"
    }
    assert used == {"get_logger", "TRAIN", "DECODE", "SELECTION", "PUZZLE"}
    imports = [n for n in solver.body if isinstance(n, ast.ImportFrom) and n.module == "arc26"]
    assert [[alias.name for alias in n.names] for n in imports] == [["timing"]]


# The starter: one logger per worker process, only for a non-null timing_dir.


def test_each_worker_process_configures_its_logger_only_for_a_timing_dir():
    tree = module_tree("starter.py")
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "arc26"]
    assert [[alias.name for alias in n.names] for n in imports] == [["timing"]]
    configures = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and same(node.func, "timing.configure")
    ]
    assert len(configures) == 1
    body = function(tree, "local_worker").body
    (guard,) = [
        s for s in body if isinstance(s, ast.If) and same(s.test, 'cfg["timing_dir"] is not None')
    ]
    assert not guard.orelse
    (call,) = guard.body
    assert same(call, 'timing.configure(cfg["timing_dir"], f"rank{rank}")')
    # After the import of the solver and the marker that lets the next rank start, before the
    # worker runs.
    marker = next(i for i, s in enumerate(body) if isinstance(s, ast.With))
    run = next(i for i, s in enumerate(body) if same(s, "worker(rank, queue, end_time, cfg)"))
    assert marker < body.index(guard) < run


# The wired worker, run against stand-ins in a subprocess.

DRIVER = r'''
import contextlib, hashlib, json, os, pickle, queue, sys, time, types

work, timing_dir, mode = sys.argv[1], sys.argv[2] or None, sys.argv[3]
EOS = 15
calls = {"train_tokens": [], "decode_tokens": [], "score_tokens": []}


def stand_in(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


class Tokenizer:
    """One token per character: enough for the formatter, cut_to_len, and grid parsing."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(map(chr, tokens))


class Model:
    def named_parameters(self):
        return []


class FastLanguageModel:
    @staticmethod
    def from_pretrained(**kwargs):
        return Model(), Tokenizer()

    @staticmethod
    def get_peft_model(model, **kwargs):
        return model

    @staticmethod
    def for_training(model):
        return model

    @staticmethod
    def for_inference(model):
        return model


class PreparedDataset:
    """The trainer's prepared training set: input_ids beside the text, as trl prepares it."""

    def __init__(self, rows, tokenizer, field):
        self.columns = {field: [row[field] for row in rows]}
        if mode != "no-input-ids":
            self.columns["input_ids"] = [tokenizer.encode(row[field]) for row in rows]
        self.column_names = list(self.columns)

    def __getitem__(self, name):
        return self.columns[name]


class UnslothTrainer:
    def __init__(
        self, model, tokenizer, data_collator, train_dataset, dataset_text_field, max_seq_length,
        args,
    ):
        self.train_dataset = PreparedDataset(train_dataset.rows, tokenizer, dataset_text_field)
        unwrap = lambda model, keep_fp32_wrapper: model
        self.accelerator = types.SimpleNamespace(unwrap_model=unwrap)

    def train(self):
        ids = self.train_dataset.columns.get("input_ids")
        calls["train_tokens"].append(None if ids is None else sum(map(len, ids)))
        return "TrainOutput(stand-in)"


class Dataset:
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def from_list(cls, rows):
        return cls(rows)


class DataCollatorForLanguageModeling:
    def __init__(self, tokenizer, mlm):
        self.tokenizer = tokenizer


cuda = types.SimpleNamespace(
    reset_peak_memory_stats=lambda: None, empty_cache=lambda: None, max_memory_allocated=lambda: 0
)
torch = stand_in(
    "torch", float32="float32", bfloat16="bfloat16", cuda=cuda,
    no_grad=lambda: (lambda function: function), inference_mode=contextlib.nullcontext,
    set_default_device=lambda device: None,
)
torch.multiprocessing = stand_in("torch.multiprocessing")
stand_in(
    "unsloth", FastLanguageModel=FastLanguageModel, UnslothTrainingArguments=dict,
    UnslothTrainer=UnslothTrainer,
)
stand_in("datasets", Dataset=Dataset)
stand_in(
    "transformers", DataCollatorForLanguageModeling=DataCollatorForLanguageModeling,
    AutoTokenizer=Tokenizer,
)
stand_in(
    "peft", get_peft_model_state_dict=lambda model, adapter_name: {},
    set_peft_model_state_dict=lambda model, weights, adapter_name: None,
)
stand_in("tqdm", tqdm=lambda iterable, **kwargs: iterable)

import numpy as np
from arc26 import timing
from arc26.nvarc import arc_solver, starter

real_time = time.time
offset = [0.0]
time.time = lambda: real_time() + offset[0]


def fake_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    """Per view: the 1x1 grid [[7]], whose inverted grids repeat across views, so rescoring is
    cached for some, and a beam too short to parse, which the worker skips."""
    calls["decode_tokens"].append(sum(map(len, prefix_tokens)))
    if mode == "cap":
        offset[0] += 1300.0  # the next batch finds the puzzle past the reference's 1200 s cap
    grid = Tokenizer().encode("7\n") + [EOS]
    return [(i, [(0.25 + i, grid), (0.5 + i, [EOS])]) for i in range(len(prefix_tokens))]


def fake_scores(queries, answers, tokenizer, model):
    pairs = list(zip(queries, answers))
    calls["score_tokens"].append(
        sum(len(tokenizer.encode(q)) + len(tokenizer.encode(a)) for q, a in pairs)
    )
    return [float(len(q) % 97 + len(a)) for q, a in pairs]


arc_solver.inference_turbo_dfs = fake_dfs
arc_solver.calc_scores = fake_scores

cfg = {
    "data_dir": os.path.join(work, "data"),
    "model_dir": os.path.join(work, "model"),
    "outputs_dir": os.path.join(work, "outputs"),
    "marker_dir": os.path.join(work, "markers"),
    "subset": None,
    "workers": 1,
    "timing_dir": timing_dir,
    "lora_r": 256,
    "lora_targets": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",
    ],
}
os.makedirs(cfg["marker_dir"])
with open(os.path.join(cfg["data_dir"], "arc-agi_evaluation_challenges.json")) as handle:
    keys = sorted(json.load(handle))
tasks = queue.Queue()
for key in keys:
    tasks.put(key)
tasks.put(None)
starter.local_worker(0, tasks, real_time() + 1e6, cfg)

outputs = cfg["outputs_dir"]
candidates = {}
for name in sorted(os.listdir(outputs)):
    with open(os.path.join(outputs, name), "rb") as handle:
        candidates[name] = hashlib.sha256(handle.read()).hexdigest()
logger = timing.get_logger()
summary = {
    "calls": calls,
    "candidates": candidates,
    "rng": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
    "logger": None if logger is None else str(logger.path),
}
with open(os.path.join(work, "summary.json"), "w") as handle:
    json.dump(summary, handle)
'''

# Two synthetic tasks, never competition data: one test input, then two.
TASKS = {
    "00000001": {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
            {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
        ],
        "test": [{"input": [[1, 1], [2, 2]]}],
    },
    "00000002": {
        "train": [
            {"input": [[1]], "output": [[1, 1]]},
            {"input": [[2]], "output": [[2, 2]]},
            {"input": [[3]], "output": [[3, 3]]},
        ],
        "test": [{"input": [[4]]}, {"input": [[5]]}],
    },
}
# Decode batches per task: 16 augmented views per test input, 4 views per batch.
BATCHES = {"00000001": 4, "00000002": 8}


class Run(NamedTuple):
    work: Path
    summary: dict
    output: str
    records: list | None


def run_worker(work, mode, timing):
    """Run starter.local_worker on the synthetic tasks in a fresh interpreter."""
    pytest.importorskip("numpy")
    data = work / "data"
    data.mkdir(parents=True)
    challenges = json.dumps(TASKS)
    (data / "arc-agi_evaluation_challenges.json").write_text(challenges, encoding="utf-8")
    driver = work.parent / f"{work.name}-driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    timing_dir = work / "run_logs" / "timing" if timing else None
    env = {key: value for key, value in os.environ.items() if key != "KAGGLE_IS_COMPETITION_RERUN"}
    # The rescoring augmentations are seeded with hash(), so both runs need one hash seed.
    env.update(PYTHONHASHSEED="0", PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, str(driver), str(work), str(timing_dir or ""), mode],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    records = None
    if timing_dir is not None:
        path = timing_dir / "timing-rank0.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
    return Run(work, summary, result.stdout, records)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("solver")
    return {
        name: run_worker(root / name, mode, timing)
        for name, mode, timing in (
            ("off", "plain", False),
            ("on", "plain", True),
            ("no-input-ids", "no-input-ids", True),
            ("cap", "cap", True),
        )
    }


def files_under(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def by_puzzle(records):
    """The records after the load, grouped by puzzle in the order they were written."""
    groups = {}
    for record in records[1:]:
        groups.setdefault(record["puzzle_id"], []).append(record)
    return groups


def test_a_null_timing_dir_writes_nothing(runs):
    off = runs["off"]
    assert off.summary["logger"] is None
    written = files_under(off.work)
    assert not [name for name in written if "timing" in name]
    expected = {"data/arc-agi_evaluation_challenges.json", "markers/worker0", "summary.json"}
    assert set(written) - {f"outputs/{name}" for name in off.summary["candidates"]} == expected


def test_timing_changes_neither_the_candidates_nor_the_random_state(runs):
    off, on = runs["off"], runs["on"]
    assert off.summary["candidates"] and on.summary["candidates"] == off.summary["candidates"]
    assert on.summary["rng"] == off.summary["rng"]
    assert on.summary["calls"] == off.summary["calls"]
    assert runs["no-input-ids"].summary["candidates"] == off.summary["candidates"]


def test_every_op_is_recorded_in_order_with_its_tokens(runs):
    on = runs["on"]
    assert on.summary["logger"].endswith("run_logs/timing/timing-rank0.jsonl")
    records = on.records
    assert records[0] == {**records[0], "puzzle_id": None, "op": "load", "tokens": None}
    assert [r for r in records if r["puzzle_id"] is None] == records[:1]
    groups = by_puzzle(records)
    assert list(groups) == sorted(TASKS)
    calls = on.summary["calls"]
    for index, (key, group) in enumerate(groups.items()):
        ops = [r["op"] for r in group]
        assert ops[0] == "train" and ops[-1] == "puzzle" and group[-1]["tokens"] is None
        assert set(ops[1:-1]) == {"decode", "selection"} and ops.count("decode") == BATCHES[key]
        assert group[0]["tokens"] == calls["train_tokens"][index] > 0
        # Overhead (puzzle minus the three phases) is never negative; seconds are rounded to
        # microseconds per record.
        phases = sum(r["seconds"] for r in group[:-1])
        assert group[-1]["seconds"] >= phases - 1e-6 * len(group)
    decode = [r["tokens"] for r in records if r["op"] == "decode"]
    assert decode == calls["decode_tokens"] and all(decode)
    selection = [r["tokens"] for r in records if r["op"] == "selection"]
    scores = calls["score_tokens"]
    assert selection == [scores[i] + scores[i + 1] for i in range(0, len(scores), 2)]
    # Some inverted grids repeat across views and are not rescored again.
    views = 16 * sum(len(task["test"]) for task in TASKS.values())
    assert 0 < len(selection) < views


def test_a_trainer_without_input_ids_records_null_train_tokens(runs):
    records = runs["no-input-ids"].records
    train = [r for r in records if r["op"] == "train"]
    assert len(train) == len(TASKS) and all(r["tokens"] is None for r in train)
    assert [r["op"] for r in records] == [r["op"] for r in runs["on"].records]


def test_a_puzzle_cut_by_the_1200_s_cap_keeps_its_records(runs):
    cap = runs["cap"]
    assert cap.output.count("timeout after") == len(TASKS)
    groups = by_puzzle(cap.records)
    assert list(groups) == sorted(TASKS)
    for group in groups.values():
        ops = [r["op"] for r in group]
        assert ops[0] == "train" and ops.count("decode") == 1 and ops[-1] == "puzzle"
