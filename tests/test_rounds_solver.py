"""The refinement rounds on the split solver's worker, end to end on GPU stand-ins (task P2.3).

B5 P2.3's acceptance asks for an integration test on DEV4 that runs one round end to end, and for
a test that injects a wrong pseudo-pair and checks that the rollback restores the previous adapter
and version. Competition data never enters a test, so the worker runs on four synthetic tasks
under DEV4's ids with DEV4's layout (1, 2, 1, and 1 test inputs), against the stand-ins for the GPU
stack of ``tests/test_solver_timing.py``, extended so that the adapter has weights: the stand-in
model holds numpy arrays behind the torch and PEFT calls that ``arc26.solver.load`` and
``arc26.ttt.SGDProvider`` make, and the stand-in trainer changes them in place from what it
trains on. Every decode answers each view's query with its own input grid (an identity DFS), but
a query of exactly 5 cells gets its first cell changed, which its views then disagree on.

- ``0934a4d8``: the identity, one test input: the round is kept, and with every input promoted no
  decode follows.
- ``36a08778``: the identity, two test inputs, the second of 5 cells: input 0 is promoted, the
  kept adapter decodes input 1, and its candidates replace round 0's.
- ``981571dc``: colors 1 and 2 swapped: the identity answer is a wrong pseudo-pair, confident and
  re-decoded alike; the candidate adapter trained with it fails the gate, and the rollback must
  restore the weights bit for bit.
- ``aa4ec2a5``: the identity, one test input of 5 cells: nothing to promote.

The runs: the hold-out without rounds (task P2.1's path, the control); one round, its decode
without the held-out pair (``final_in_prompts`` false); a governor that declines every round;
raising ops; a re-decode cut by the 1200 s cap; the default, the final decode with the held-out
pair back in its prompts; and two rounds. The provider itself is checked in a run of its own.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

from arc26.loop import (
    ALL_PROMOTED,
    BELOW_TAU,
    CANDIDATE_TRAIN,
    DECLINED,
    FINAL_DECODE,
    FINAL_SELECTION,
    FINGERPRINT,
    GATE,
    GATE_DECODE,
    KEPT,
    NO_TIME,
    NOT_PROMOTED,
    NOTHING_TO_PROMOTE,
    REDECODE,
    REDECODE_CUT,
    REDECODE_DECODE,
    REDECODE_RAISED,
    RESTORE,
    ROLLED_BACK,
    ROUND,
    ROUND_DECODE,
    ROUND_SELECTION,
    SNAPSHOT,
    TIED,
)

REPO = Path(__file__).resolve().parent.parent
GOVERNOR = yaml.safe_load((REPO / "configs" / "governor.yaml").read_text(encoding="utf-8"))


def load_timing_tests():
    path = REPO / "tests" / "test_solver_timing.py"
    spec = importlib.util.spec_from_file_location("_rounds_stand_ins", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAND_INS = load_timing_tests()
# The stand-in modules of test_solver_timing's driver, up to its imports of the solver.
STAND_IN_HEAD, _IMPORT, _ = STAND_INS.DRIVER.partition("import numpy as np\n")
assert _IMPORT and "def stand_in(" in STAND_IN_HEAD

# Weights for the stand-in model, torch's tensor calls on numpy arrays, a trainer that changes the
# weights from what it trains on, and the phase of every decode (test, gate, a round's).
WEIGHTS = r'''
import numpy as np


class Tensor:
    """A numpy array behind the torch calls of arc26.solver.load and arc26.ttt.SGDProvider."""

    def __init__(self, array):
        self.array = array

    dtype = property(lambda self: self.array.dtype)
    shape = property(lambda self: self.array.shape)

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.array.copy())

    def to(self, device, copy=False):
        return Tensor(self.array.copy()) if copy else self

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.array))

    def view(self, dtype):
        return Tensor(self.array.view(dtype))

    def numpy(self):
        return self.array


sys.modules["torch"].uint8 = np.uint8


class WeightModel(Model):
    """The adapter's weights: two arrays, trained in place."""

    def __init__(self):
        self.weights = {
            "lora_A": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            "lora_B": np.zeros(4, dtype=np.float32),
        }


def weights_digest(model):
    data = b"".join(model.weights[name].tobytes() for name in sorted(model.weights))
    return hashlib.sha256(data).hexdigest()[:16]


FastLanguageModel.from_pretrained = staticmethod(lambda **kwargs: (WeightModel(), Tokenizer()))
peft_stand_in = sys.modules["peft"]
peft_stand_in.get_peft_model_state_dict = lambda model, adapter_name: {
    name: Tensor(array) for name, array in model.weights.items()
}


def set_state(model, weights, adapter_name):
    for name, tensor in weights.items():
        np.copyto(model.weights[name], tensor.array)


peft_stand_in.set_peft_model_state_dict = set_state
current = [None]
trainings = []


class WeightTrainer(UnslothTrainer):
    """Trains the model's weights in place: a step that depends on the samples it trains on,
    the first max_steps of them when the arguments set it."""

    def __init__(
        self, model, tokenizer, data_collator, train_dataset, dataset_text_field, max_seq_length,
        args,
    ):
        super().__init__(
            model, tokenizer, data_collator, train_dataset, dataset_text_field, max_seq_length,
            args,
        )
        self.model, self.args = model, args
        self.texts = [row[dataset_text_field] for row in train_dataset.rows]

    def train(self):
        steps = self.args.get("max_steps", -1)
        seen = self.texts if steps < 0 else self.texts[:steps]
        before = weights_digest(self.model)
        step = int(hashlib.sha256("".join(seen).encode()).hexdigest()[:6], 16) % 997 / 1000
        for array in self.model.weights.values():
            array += np.float32(step + 0.25)
        trainings.append({
            "key": current[0], "steps": steps, "samples": len(seen), "views": len(self.texts),
            "tokens": sum(map(len, self.texts)), "before": before,
            "after": weights_digest(self.model),
        })
        if mode == "raises" and steps > 0 and current[0] == "0934a4d8":
            raise MemoryError("CUDA out of memory (stand-in fit)")
        return super().train()


sys.modules["unsloth"].UnslothTrainer = WeightTrainer
'''

DRIVER = STAND_IN_HEAD + WEIGHTS + r'''
from arc26 import decode, solver, timing, ttt

holdout = json.loads(sys.argv[4])
end_in = float(sys.argv[5])
phase = ["test"]
dfs_calls = []
events = []
real_time = time.time
offset = [0.0]
time.time = lambda: real_time() + offset[0]


def identity_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    """Each view's answer is its query's input grid; a query of 5 cells gets its first cell set
    to 0 in the view's frame, so its views disagree once mapped back."""
    found = []
    for row, tokens in enumerate(prefix_tokens):
        text = "".join(map(chr, tokens))
        query = text.rsplit("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        dfs_calls.append([current[0], phase[0], weights_digest(model), text])
        if mode == "raises" and current[0] == "981571dc" and "redecode" in phase[0]:
            raise MemoryError("CUDA out of memory (stand-in re-decode)")
        if mode == "raises" and current[0] == "36a08778" and phase[0] == "round1":
            raise MemoryError("CUDA out of memory (stand-in decode)")
        answer = query
        if sum(c.isdigit() for c in query) == 5:
            answer = "0" + query[1:]
        found.append((row, [(0.25, [ord(c) for c in answer] + [EOS])]))
    if mode == "cut" and "redecode" in phase[0]:
        offset[0] += 1300.0  # the next batch finds the re-decode past the 1200 s cap
    return found


def fake_scores(queries, answers, tokenizer, model):
    return [float(len(q) % 97 + len(a)) for q, a in zip(queries, answers)]


moved_decode = decode.decode_puzzle
moved_fit = ttt.fit


def tagged_decode(**kwargs):
    folder = kwargs["dir_outputs"]
    if folder == cfg["outputs_dir"]:
        phase[0] = "test"
    elif folder.startswith(holdout["gate_dir"]):
        phase[0] = "gate"
    else:
        phase[0] = os.path.basename(folder)
    current[0] = kwargs["key"]
    return moved_decode(**kwargs)


def tracked_fit(**kwargs):
    current[0] = kwargs["key"]
    return moved_fit(**kwargs)


decode.decode_puzzle = tagged_decode
decode.inference_turbo_dfs = identity_dfs
decode.calc_scores = fake_scores
ttt.fit = tracked_fit
moved_adapt, moved_rollback = ttt.SGDProvider.adapt, ttt.SGDProvider.rollback


def adapt(self, demos, init, budget_s):
    events.append({
        "key": self.key, "event": "adapt", "steps": self.steps, "budget": budget_s,
        "pairs": len(demos.pairs),
        "weights": {name: array.tolist() for name, array in init.weights.items()},
    })
    return moved_adapt(self, demos, init, budget_s)


def rollback(self, previous):
    model = moved_rollback(self, previous)
    events.append({
        "key": self.key, "event": "rollback",
        "weights": {name: array.tolist() for name, array in model.weights.items()},
    })
    return model


ttt.SGDProvider.adapt = adapt
ttt.SGDProvider.rollback = rollback

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
    "holdout": holdout,
}
os.makedirs(cfg["marker_dir"])
with open(os.path.join(cfg["data_dir"], "arc-agi_evaluation_challenges.json")) as handle:
    keys = sorted(json.load(handle))
tasks = queue.Queue()
for key in keys:
    tasks.put(key)
tasks.put(None)
solver.local_worker(0, tasks, real_time() + end_in, cfg)


def digests(folder):
    found = {}
    for directory, _, files in sorted(os.walk(folder)):
        for name in sorted(files):
            path = os.path.join(directory, name)
            with open(path, "rb") as handle:
                found[os.path.relpath(path, folder)] = hashlib.sha256(handle.read()).hexdigest()
    return found


logs = {}
if os.path.isdir(holdout["log_dir"]):
    for name in sorted(os.listdir(holdout["log_dir"])):
        with open(os.path.join(holdout["log_dir"], name)) as handle:
            logs[name] = [json.loads(line) for line in handle]
rounds = holdout.get("rounds")
summary = {
    "candidates": digests(cfg["outputs_dir"]),
    "rounds": digests(rounds["round_dir"]) if rounds else {},
    "logs": logs,
    "dfs": dfs_calls,
    "trainings": trainings,
    "events": events,
}
with open(os.path.join(work, "summary.json"), "w") as handle:
    json.dump(summary, handle)
'''


def identity(rows):
    return {"input": rows, "output": rows}


def swap(rows):
    return {"input": rows, "output": [[{1: 2, 2: 1}.get(c, c) for c in row] for row in rows]}


# Synthetic tasks under DEV4's ids, never competition data. Only the noisy test inputs have
# exactly 5 cells; the held-out pair is the last of each task.
TASKS = {
    "0934a4d8": {
        "train": [
            identity([[1, 2], [3, 4]]), identity([[5, 6, 7]]), identity([[1, 1, 2], [3, 3, 4]]),
        ],
        "test": [{"input": [[8, 9], [9, 8]]}],
    },
    "36a08778": {
        "train": [
            identity([[2, 3], [4, 2]]), identity([[6, 7, 8]]), identity([[3, 3, 4], [5, 5, 6]]),
        ],
        "test": [{"input": [[7, 8], [8, 7]]}, {"input": [[1, 2, 3, 4, 6]]}],
    },
    "981571dc": {
        "train": [swap([[1, 2, 3]]), swap([[4, 1, 2, 6]]), swap([[1, 2, 7], [8, 2, 1]])],
        "test": [{"input": [[1, 2], [3, 4]]}],
    },
    "aa4ec2a5": {
        "train": [
            identity([[3, 4], [4, 3]]), identity([[5, 6, 5]]), identity([[1, 3, 1], [3, 1, 3]]),
        ],
        "test": [{"input": [[2, 4, 6, 8, 9]]}],
    },
}
HELD = {key: task["train"][-1] for key, task in TASKS.items()}


class Run(NamedTuple):
    work: Path
    summary: dict
    output: str
    records: list


def run_rounds_worker(work, rounds, mode="plain", end_in=1e6, tasks=None, k2=True):
    """Run the split solver's local_worker on ``tasks`` (TASKS by default) with a held-out pair
    and the given round settings (None: task P2.1's path)."""
    pytest.importorskip("numpy")
    data = work / "data"
    data.mkdir(parents=True)
    challenges = json.dumps(TASKS if tasks is None else tasks)
    (data / "arc-agi_evaluation_challenges.json").write_text(challenges, encoding="utf-8")
    driver = work.parent / f"{work.name}-driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    timing_dir = work / "run_logs" / "timing"
    holdout = {
        "rule": "last", "k2": k2, "in_test_prompts": False,
        "gate_dir": str(work / "gate_outputs"), "log_dir": str(work / "run_logs" / "gate"),
    }  # fmt: skip
    if rounds is not None:
        holdout["rounds"] = {
            "round_dir": str(work / "round_outputs"), "governor": GOVERNOR, **rounds
        }  # fmt: skip
    env = {key: value for key, value in os.environ.items() if key != "KAGGLE_IS_COMPETITION_RERUN"}
    env.update(PYTHONHASHSEED="0", PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [
            sys.executable, str(driver), str(work), str(timing_dir), mode, json.dumps(holdout),
            str(end_in),
        ],  # fmt: skip
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    lines = (timing_dir / "timing-rank0.jsonl").read_text(encoding="utf-8").splitlines()
    return Run(work, summary, result.stdout, [json.loads(line) for line in lines])


# The runs whose candidates are compared with round 0's decode without the held-out pair: the
# decode after a kept round, and none after a round that keeps nothing (final_in_prompts false).
PAIRS = {"final_in_prompts": False}
RUNS = {
    "plain": (None, "plain", 1e6),
    "rounds": (PAIRS, "plain", 1e6),
    # 300 s before the end time: every round's estimate, and the final decode's, is above what
    # is left.
    "declined": ({}, "plain", 300.0),
    "raises": (PAIRS, "raises", 1e6),
    "cut": (PAIRS, "cut", 1e6),
    "final": ({}, "plain", 1e6),
    "two": ({"max_rounds": 2}, "plain", 1e6),
}


# A two-pair identity task with the k = 2 flag off: no pair is held out, so no gate and no round.
K2_TASKS = {
    "0934a4d8": {
        "train": [identity([[1, 2], [3, 4]]), identity([[5, 6, 7]])],
        "test": [{"input": [[8, 9], [9, 8]]}],
    },
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("rounds")
    found = {
        name: run_rounds_worker(root / name, rounds, mode, end_in)
        for name, (rounds, mode, end_in) in RUNS.items()
    }
    found["k2-off"] = run_rounds_worker(root / "k2-off", {}, tasks=K2_TASKS, k2=False)
    return found


def log(run, prefix):
    return run.summary["logs"].get(f"{prefix}-rank0.jsonl", [])


def by_key(lines):
    found = {}
    for line in lines:
        found.setdefault(line["puzzle_id"], []).append(line)
    return found


def files_of(candidates, key, index=None):
    """The candidate files of a puzzle, or of one of its test inputs: their names start with the
    view's key, ``<task>_<test index>.``."""
    found = {}
    for name, digest in candidates.items():
        base = name.split(os.sep)[-1].split(".")[0]
        task, _, number = base.rpartition("_")
        if task == key and (index is None or number == str(index)):
            found[name.split(os.sep)[-1]] = digest
    return found


def round_files(run, key, folder):
    """The files of one folder of a puzzle's round folder, by name."""
    prefix = os.path.join(key, folder) + os.sep
    return {
        name[len(prefix) :]: digest
        for name, digest in run.summary["rounds"].items()
        if name.startswith(prefix)
    }


def query_cells(text):
    """The cell count of a prompt's query, its last user turn."""
    turn = text.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0]
    return sum(c.isdigit() for c in turn)


def demo_cells(text):
    """The cell counts of a prompt's demonstration inputs, sorted (the views shuffle them)."""
    turns = text.split("<|im_start|>user\n")[1:-1]
    return sorted(sum(c.isdigit() for c in turn.split("<|im_end|>")[0]) for turn in turns)


def fits(run, key):
    """The stand-in trainer's fits of one puzzle: round 0's (full epoch), then candidates'."""
    return [t for t in run.summary["trainings"] if t["key"] == key]


# B5 acceptance: one round end to end, and the rollback after a wrong pseudo-pair.


def test_one_round_runs_end_to_end_on_the_split_solver(runs):
    run = runs["rounds"]
    rounds = {key: lines for key, lines in by_key(log(run, "round")).items()}
    outcomes = {key: [line["outcome"] for line in lines] for key, lines in rounds.items()}
    assert outcomes == {
        "0934a4d8": [KEPT], "36a08778": [KEPT], "981571dc": [ROLLED_BACK],
        "aa4ec2a5": [NOTHING_TO_PROMOTE],
    }  # fmt: skip
    kept = rounds["36a08778"][0]
    assert (kept["version"], kept["next_version"], kept["promoted"]) == (0, 1, [0])
    assert kept["screened"] == [0] and kept["decision"] == "gradient"
    # A plan for every round with something to promote, none for the one without.
    assert sorted(by_key(log(run, "plan"))) == ["0934a4d8", "36a08778", "981571dc"]
    assert all(plan["decision"] == "gradient" for plan in log(run, "plan"))
    # The candidate fit continued from round 0's adapter for s' = 32 of the 128 views.
    round0, candidate = fits(run, "36a08778")
    assert (round0["steps"], round0["samples"], round0["views"]) == (-1, 128, 128)
    assert (candidate["steps"], candidate["samples"], candidate["views"]) == (32, 32, 128)
    assert candidate["before"] == round0["after"] != candidate["after"]
    (adapted,) = [e for e in run.summary["events"] if e["key"] == "36a08778"]
    assert adapted["event"] == "adapt" and adapted["steps"] == 32 and adapted["pairs"] == 3


def test_the_kept_adapter_decodes_the_test_inputs_and_the_folder_holds_its_candidates(runs):
    run, plain = runs["rounds"], runs["plain"]
    _, candidate = fits(run, "36a08778")
    decodes = [c for c in run.summary["dfs"] if c[0] == "36a08778" and c[1] == "round1"]
    # Only input 1, still open, is decoded: input 0's candidates are the ones that promoted it.
    assert len(decodes) == 16 and {c[2] for c in decodes} == {candidate["after"]}
    assert {query_cells(c[3]) for c in decodes} == {5}
    tests = [c for c in run.summary["dfs"] if c[0] == "36a08778" and c[1] == "test"]
    assert {c[2] for c in tests} == {candidate["before"]}
    # Input 1's candidates are the round's, input 0 keeps the ones that promoted it.
    outputs, before = run.summary["candidates"], plain.summary["candidates"]
    round1 = round_files(run, "36a08778", "round1")
    assert files_of(outputs, "36a08778", 1) == files_of(round1, "36a08778", 1) != {}
    assert files_of(outputs, "36a08778", 1) != files_of(before, "36a08778", 1)
    assert files_of(outputs, "36a08778", 0) == files_of(before, "36a08778", 0) != {}
    # The round's decode prompts hold the enlarged set: round 0's two pairs and the pseudo-pair.
    assert {tuple(demo_cells(c[3])) for c in decodes} == {(3, 4, 4)}
    assert {tuple(demo_cells(c[3])) for c in tests} == {(3, 4)}
    # Round 0's candidates stay in the round folder (P2.8).
    assert round_files(run, "36a08778", "round0") == files_of(before, "36a08778")
    ((decoded),) = by_key(log(run, "decode"))["36a08778"]
    assert (decoded["name"], decoded["inputs"], decoded["installed"]) == ("round1", [1], [1])
    assert (decoded["batches"], decoded["expected"], decoded["cut"]) == (4, 4, False)
    # Its files carry the puzzle's index of the input, 1, though it was decoded alone.
    assert set(round1) == set(files_of(round1, "36a08778", 1))
    # A puzzle whose every input was promoted decodes nothing after the round.
    ((skipped),) = by_key(log(run, "decode"))["0934a4d8"]
    assert skipped["skipped"] == ALL_PROMOTED
    assert files_of(outputs, "0934a4d8") == files_of(plain.summary["candidates"], "0934a4d8")


def test_a_wrong_pseudo_pair_rolls_back_the_weights_bit_for_bit_and_the_version(runs):
    run, plain = runs["rounds"], runs["plain"]
    (record,) = by_key(log(run, "round"))["981571dc"]
    assert record["outcome"] == ROLLED_BACK and record["restored"] is True
    assert (record["version"], record["next_version"], record["promoted"]) == (0, 0, [])
    assert record["weights_before"] == record["weights_after"] is not None
    adapted, rolled = [e for e in run.summary["events"] if e["key"] == "981571dc"]
    assert (adapted["event"], rolled["event"]) == ("adapt", "rollback")
    assert rolled["weights"] == adapted["weights"]  # every float, exactly
    _, candidate = fits(run, "981571dc")
    assert candidate["after"] != candidate["before"]  # the fit did change them first
    (promotion,) = by_key(log(run, "promote"))["981571dc"]
    assert promotion["reason"] == "gate_failed" and promotion["fitted"] == [0]
    assert promotion["gate"]["expected_rank"] is None
    # Round 0's candidates are submitted, as without rounds.
    assert files_of(run.summary["candidates"], "981571dc") == files_of(
        plain.summary["candidates"], "981571dc"
    )


def test_nothing_to_promote_writes_its_reason_and_no_plan(runs):
    (promotion,) = by_key(log(runs["rounds"], "promote"))["aa4ec2a5"]
    assert promotion["reason"] in (TIED, BELOW_TAU) and promotion["redecode"] is None
    assert "aa4ec2a5" not in {e["key"] for e in runs["rounds"].summary["events"]}


# The governor, raising and cut ops, and the timing records.


def test_the_governor_declines_every_round_and_each_has_a_plan_record(runs):
    run, plain = runs["declined"], runs["plain"]
    outcomes = {k: [line["outcome"] for line in v] for k, v in by_key(log(run, "round")).items()}
    assert outcomes == {
        "0934a4d8": [DECLINED], "36a08778": [DECLINED], "981571dc": [DECLINED],
        "aa4ec2a5": [NOTHING_TO_PROMOTE],
    }  # fmt: skip
    plans = by_key(log(run, "plan"))
    assert sorted(plans) == ["0934a4d8", "36a08778", "981571dc"]
    for lines in plans.values():
        (plan,) = lines
        assert plan["decision"] == "none" and plan["estimates"]["gradient"] > plan["remaining"]
    assert run.summary["events"] == []
    # Every voted answer's agreement is recorded: the inputs that passed the free checks as
    # declined, the others with their reason; no re-decode, fit, or gate ran.
    promotions = by_key(log(run, "promote"))
    assert sorted(promotions) == sorted(TASKS)
    for key in ("0934a4d8", "36a08778", "981571dc"):
        first = promotions[key][0]
        assert first["reason"] == "declined" and first["agreement"] == 1.0
        assert first["redecode"] is None and first["gate"] is None
    # The final decode (the default) cannot fit either, so none runs.
    decodes = log(run, "decode")
    assert sorted(d["puzzle_id"] for d in decodes) == sorted(TASKS)
    assert {(d["name"], d["skipped"]) for d in decodes} == {("final", NO_TIME)}
    # Round 0 is untouched: the same candidates, verdicts, and timing ops as without rounds.
    assert run.summary["candidates"] == plain.summary["candidates"]
    strip = [{k: v for k, v in line.items() if k != "seconds"} for line in log(plain, "gate")]
    assert [{k: v for k, v in line.items() if k != "seconds"} for line in log(run, "gate")] == strip
    assert [r["op"] for r in run.records] == [r["op"] for r in plain.records]


def test_a_raising_fit_redecode_or_decode_leaves_round_0s_candidates_and_the_worker_going(runs):
    run, plain = runs["raises"], runs["plain"]
    rounds = {key: lines[0] for key, lines in by_key(log(run, "round")).items()}
    # The fit raised mid-training: rolled back, the weights restored.
    assert rounds["0934a4d8"]["outcome"] == ROLLED_BACK and rounds["0934a4d8"]["restored"]
    (fit_failed,) = by_key(log(run, "promote"))["0934a4d8"]
    assert fit_failed["reason"] == "fit_raised" and "stand-in fit" in fit_failed["error"]
    adapted, rolled = [e for e in run.summary["events"] if e["key"] == "0934a4d8"]
    assert rolled["weights"] == adapted["weights"]
    # The re-decode raised: no fit, nothing promoted.
    assert rounds["981571dc"]["outcome"] == NOT_PROMOTED
    (redecode_failed,) = by_key(log(run, "promote"))["981571dc"]
    assert redecode_failed["reason"] == REDECODE_RAISED
    # The decode after the kept round raised: nothing replaced, the rounds end.
    assert rounds["36a08778"]["outcome"] == KEPT
    ((decoded),) = by_key(log(run, "decode"))["36a08778"]
    assert "MemoryError" in decoded["error"] and decoded["installed"] == []
    # Every puzzle ran to its end, and the submission's candidates are round 0's.
    assert [r["puzzle_id"] for r in run.records if r["op"] == "puzzle"] == sorted(TASKS)
    assert run.summary["candidates"] == plain.summary["candidates"]
    assert "[Rank 0] round 1 for 0934a4d8: rolled_back, weights restored" in run.output


def test_a_redecode_cut_by_the_time_cap_fails_its_check(runs):
    run, plain = runs["cut"], runs["plain"]
    promotions = by_key(log(run, "promote"))
    for key in ("0934a4d8", "36a08778", "981571dc"):
        first = promotions[key][0]
        assert first["reason"] == REDECODE_CUT and first["redecode"]["cut"] is True
        assert first["redecode"]["candidates"] == 4 and first["redecode"]["agreed"] is False
    assert {line["outcome"] for line in log(run, "round")} == {NOT_PROMOTED, NOTHING_TO_PROMOTE}
    assert run.summary["events"] == [] and run.summary["candidates"] == plain.summary["candidates"]
    assert run.output.count("timeout after") >= 3


def test_the_rounds_ops_are_recorded_inside_the_puzzle_span(runs):
    groups = STAND_INS.by_puzzle(runs["rounds"].records)
    assert list(groups) == sorted(TASKS)
    # Spans that hold no other op of the puzzle: round 0's, and the round with what follows it.
    outer = {"train", "decode", "selection", ROUND, ROUND_DECODE, "round_selection"}
    for group in groups.values():
        ops = [r["op"] for r in group]
        assert ops[0] == "train" and ops[-1] == "puzzle" and ops.count("puzzle") == 1
        span = group[-1]["seconds"]
        assert span >= max(r["seconds"] for r in group[:-1])
        assert span >= sum(r["seconds"] for r in group if r["op"] in outer) - 1e-3
    kept = [r["op"] for r in groups["36a08778"]]
    assert kept.count(REDECODE_DECODE) == 4 and kept.count(REDECODE) == 1
    assert kept.count(GATE_DECODE) == 8 and kept.count(GATE) == 2  # round 0's and the round's
    assert kept.count(CANDIDATE_TRAIN) == 1 and kept.count(ROUND) == 1
    assert kept.count(ROUND_DECODE) == 4 and kept.count(SNAPSHOT) == 1
    assert kept.count(FINGERPRINT) == 2 and RESTORE not in kept
    rolled = [r["op"] for r in groups["981571dc"]]
    assert rolled.count(RESTORE) == 1 and rolled.count(FINGERPRINT) == 2
    assert ROUND_DECODE not in rolled and rolled.index(ROUND) > rolled.index(RESTORE)
    # The final decode's ops lie inside the span as well, after the round's: 4 DFS batches of the
    # one open input of each puzzle but 0934a4d8, whose every input was promoted.
    final_outer = {"train", "decode", "selection", ROUND, FINAL_DECODE, FINAL_SELECTION}
    batches = {"0934a4d8": 0, "36a08778": 4, "981571dc": 4, "aa4ec2a5": 4}
    for key, group in STAND_INS.by_puzzle(runs["final"].records).items():
        ops = [r["op"] for r in group]
        assert ROUND_DECODE not in ops and ROUND_SELECTION not in ops
        assert ops.count(FINAL_DECODE) == batches[key]
        span = group[-1]["seconds"]
        assert span >= sum(r["seconds"] for r in group if r["op"] in final_outer) - 1e-3
    # P0.3b's count: the tokens of the dataset the trainer prepared, its 128 views, though s' = 32
    # steps train on 32 of them (the governor prices those, SGDProvider.fit_shape).
    candidate = next(r for r in groups["36a08778"] if r["op"] == CANDIDATE_TRAIN)
    assert candidate["tokens"] == fits(runs["rounds"], "36a08778")[1]["tokens"] > 0


# The final decode with the held-out pair (the default), and two rounds.


def test_final_in_prompts_decodes_with_the_held_out_pair_back_in_the_prompts(runs):
    run, plain = runs["final"], runs["plain"]
    # The default: the solver JSON carried no final_in_prompts.
    assert "final_in_prompts" not in RUNS["final"][0]
    decodes = by_key(log(run, "decode"))
    held = sum(len(row) for row in HELD["36a08778"]["input"])
    finals = [c for c in run.summary["dfs"] if c[0] == "36a08778" and c[1] == "final"]
    assert len(finals) == 16 and not [c for c in run.summary["dfs"] if c[1] == "round1"]
    # Every prompt holds the two training pairs, the held-out pair (6 cells), and the pseudo-pair.
    assert held == 6 and {tuple(demo_cells(c[3])) for c in finals} == {(3, 4, 4, 6)}
    ((final),) = decodes["36a08778"]
    assert (final["name"], final["prompts"], final["installed"]) == ("final", "held_out", [1])
    assert final["estimate"] > 0 and final["remaining"] > final["estimate"]
    # After a rolled-back round, and with nothing to promote, the final decode still runs with
    # round 0's (restored) adapter and the pair back in the prompts.
    assert decodes["981571dc"][0]["installed"] == [0] and decodes["aa4ec2a5"][0]["installed"] == [0]
    assert [(d["name"], d["skipped"]) for d in decodes["0934a4d8"]] == [("final", ALL_PROMOTED)]
    final_ops = [r["op"] for r in run.records if r["puzzle_id"] == "981571dc"]
    assert final_ops.count(FINAL_DECODE) == 4
    assert files_of(run.summary["candidates"], "981571dc") != files_of(
        plain.summary["candidates"], "981571dc"
    )


def test_two_rounds_run_only_after_a_kept_round_one(runs):
    run = runs["two"]
    rounds = {k: [line["outcome"] for line in v] for k, v in by_key(log(run, "round")).items()}
    # 36a08778: round 1 kept; round 2 on the round-1 decode's candidates finds input 1 still
    # disagreeing, so nothing to promote. Every other puzzle stops after round 1.
    assert rounds == {
        "0934a4d8": [KEPT], "36a08778": [KEPT, NOTHING_TO_PROMOTE], "981571dc": [ROLLED_BACK],
        "aa4ec2a5": [NOTHING_TO_PROMOTE],
    }  # fmt: skip
    second = [p for p in by_key(log(run, "promote"))["36a08778"] if p["round"] == 2]
    assert [p["reason"] for p in second][0] == "already_promoted"
    assert second[1]["reason"] in (TIED, BELOW_TAU)
    # Round 1's decode, without the pair, fed round 2; the final decode, with it, follows.
    decodes = by_key(log(run, "decode"))["36a08778"]
    assert [(d["name"], d["prompts"], d["installed"]) for d in decodes] == [
        ("round1", "pairs", [1]), ("final", "held_out", [1])
    ]  # fmt: skip


def test_a_puzzle_without_a_held_out_pair_records_its_free_checks_and_runs_no_round(runs):
    run = runs["k2-off"]
    ((record),) = log(run, "round")
    assert (record["puzzle_id"], record["outcome"]) == ("0934a4d8", "no_held_out")
    ((promotion),) = log(run, "promote")
    assert promotion["reason"] == "no_held_out" and promotion["agreement"] == 1.0
    assert log(run, "plan") == [] and run.summary["events"] == []
    # Round 0's decode already held every pair: no final decode.
    assert log(run, "decode") == []
    ((verdict),) = log(run, "gate")
    assert verdict["applied"] is False
    ops = [r["op"] for r in run.records if r["puzzle_id"] == "0934a4d8"]
    assert ops[0] == "train" and ops[-1] == "puzzle" and ROUND not in ops


# The provider on its own: the snapshot, the continuation's steps, the restore, the price.

PROVIDER_DRIVER = STAND_IN_HEAD + WEIGHTS + r'''
import math
from arc26 import ttt, timing
from arc26.loop import DemoSet
from arc26.nvarc.arc_loader import QwenFormatter

task = json.loads(sys.argv[4])
demos = DemoSet.from_task("00000001", task)
logger = timing.TimingLogger(os.path.join(work, "timing"), "rank0")
tokenizer = Tokenizer()
formatter = QwenFormatter(tokenizer=tokenizer)
collator = ttt.QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
provider = ttt.SGDProvider(
    key="00000001", formatter=formatter, max_seq_length=8192, tokenizer=tokenizer,
    collator=collator, train_args={"num_train_epochs": 1, "seed": 42}, timing_logger=logger,
    rank=0, steps=32,
)
model = WeightModel()
result = {}
result["unchanged_rollback"] = provider.rollback(model) is model
before = {name: array.copy() for name, array in model.weights.items()}
first = provider.fingerprint(model)
state = np.random.get_state()
result["shape"] = list(provider.fit_shape(demos.without_held_out()))
result["rng_kept"] = all(
    (a == b).all() if hasattr(a, "all") else a == b
    for a, b in zip(state, np.random.get_state())
)
trained = provider.adapt(demos.without_held_out(), model, 123.0)
result["same_model"] = trained is model
result["trained"] = trainings[-1]
result["changed"] = provider.fingerprint(model) != first
result["snapshot_independent"] = all(
    (provider._snapshot[name].array == before[name]).all() for name in before
)
restored = provider.rollback(model)
result["restored_bits"] = all(
    restored.weights[name].tobytes() == before[name].tobytes() for name in before
)
result["fingerprint_equal"] = provider.fingerprint(restored) == first
errors = []
for steps in (0, 1.5):
    try:
        ttt.SGDProvider(
            key="k", formatter=formatter, max_seq_length=8192, tokenizer=tokenizer,
            collator=collator, train_args={}, timing_logger=None, rank=0, steps=steps,
        )
    except (TypeError, ValueError) as err:
        errors.append(type(err).__name__)
result["errors"] = errors
records, _ = timing.read_records([logger.path])
result["ops"] = [r.op for r in records]
# Round 1 kept (no rollback), then round 2's copy raises: its rollback must leave round 1's
# weights, never load round 1's snapshot (round 0's weights) onto them.
provider.adapt(demos.without_held_out(), model, 123.0)
kept = {name: array.copy() for name, array in model.weights.items()}
real_to = Tensor.to


def failing_to(self, device, copy=False):
    if copy:
        raise MemoryError("host out of memory (stand-in)")
    return real_to(self, device, copy)


Tensor.to = failing_to
try:
    provider.adapt(demos.without_held_out(), model, 123.0)
except MemoryError:
    result["copy_raised"] = True
Tensor.to = real_to
after = provider.rollback(model)
result["round_two_kept"] = all(
    after.weights[name].tobytes() == kept[name].tobytes() for name in kept
)
result["snapshot_cleared"] = provider._snapshot is None
with open(os.path.join(work, "summary.json"), "w") as handle:
    json.dump(result, handle)
'''


@pytest.fixture(scope="module")
def provider_run(tmp_path_factory):
    pytest.importorskip("numpy")
    work = tmp_path_factory.mktemp("provider")
    driver = work.parent / f"{work.name}-driver.py"
    driver.write_text(PROVIDER_DRIVER, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key != "KAGGLE_IS_COMPETITION_RERUN"}
    env.update(PYTHONHASHSEED="0", PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    task = TASKS["36a08778"]
    result = subprocess.run(
        [sys.executable, str(driver), str(work), "", "plain", json.dumps(task)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((work / "summary.json").read_text(encoding="utf-8"))


def test_the_provider_continues_from_a_snapshot_and_restores_it_bit_for_bit(provider_run):
    result = provider_run
    assert result["unchanged_rollback"] is True  # no snapshot yet: nothing to restore
    assert result["same_model"] and result["changed"] and result["snapshot_independent"]
    assert result["restored_bits"] and result["fingerprint_equal"]
    trained = result["trained"]
    assert (trained["steps"], trained["samples"], trained["views"]) == (32, 32, 128)
    assert result["ops"] == [FINGERPRINT, SNAPSHOT, CANDIDATE_TRAIN, FINGERPRINT, RESTORE,
                             FINGERPRINT]  # fmt: skip
    assert result["errors"] == ["ValueError", "TypeError"]


def test_a_copy_that_raises_in_round_two_restores_nothing_onto_round_ones_adapter(provider_run):
    """The snapshot is dropped before each copy: when round 2's copy raises after a kept round 1,
    the rollback has nothing to load, and the weights stay round 1's."""
    result = provider_run
    assert result["copy_raised"] and result["snapshot_cleared"] and result["round_two_kept"]


def test_the_provider_prices_s_prime_of_the_128_views_and_keeps_numpys_state(provider_run):
    tokens, steps = provider_run["shape"]
    assert steps == 32 and provider_run["rng_kept"]
    # One stand-in token per character: the tokens of the 128 views the fit builds, scaled to the
    # 32 that s' steps train on.
    trained = provider_run["trained"]
    assert trained["views"] == 128 and tokens == -(-trained["tokens"] * 32 // 128) > 0


# On the GPU stack (run by the human per docs/runbooks/, never by the agent): the provider's
# snapshot, restore, and fingerprint on CUDA bf16 PEFT weights, the training replaced by an
# in-place change of every adapter weight.


@pytest.mark.gpu
def test_the_provider_restores_cuda_bf16_lora_weights_bit_for_bit(monkeypatch):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from arc26 import ttt
    from arc26.loop import DemoSet

    config = transformers.LlamaConfig(
        vocab_size=16, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4,
    )  # fmt: skip
    base = transformers.LlamaForCausalLM(config).to(torch.bfloat16).cuda()
    # As the notebook configuration: LoRA factors, and embed_tokens and lm_head as full copies.
    lora = peft.LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )  # fmt: skip
    model = peft.get_peft_model(base, lora)

    def fake_fit(model, default_weights, **kwargs):
        peft.set_peft_model_state_dict(model, default_weights.copy(), adapter_name="default")
        with torch.no_grad():
            for _, param in model.named_parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * 0.01)
        return model, None

    monkeypatch.setattr(ttt, "fit", fake_fit)
    monkeypatch.setattr(ttt.FastLanguageModel, "for_inference", staticmethod(lambda m: m))
    provider = ttt.SGDProvider(
        key="00000001", formatter=None, max_seq_length=8192, tokenizer=None, collator=None,
        train_args={}, timing_logger=None, rank=0, steps=32,
    )  # fmt: skip
    task = TASKS["0934a4d8"]
    demos = DemoSet.from_task("00000001", task).without_held_out()
    before = {
        name: tensor.detach().clone()
        for name, tensor in peft.get_peft_model_state_dict(model).items()
    }
    first = provider.fingerprint(model)
    trained = provider.adapt(demos, model, 100.0)
    assert provider.fingerprint(trained) != first
    restored = provider.rollback(trained)
    after = peft.get_peft_model_state_dict(restored)
    assert all(torch.equal(after[name], before[name]) for name in before)
    assert provider.fingerprint(restored) == first
    assert all(t.device.type == "cpu" for t in provider._snapshot.values())
    # The copy holds the full copies of embed_tokens and lm_head beside the LoRA factors.
    assert any("embed_tokens" in name for name in provider._snapshot)
    assert any("lm_head" in name for name in provider._snapshot)
    assert any("lora_A" in name for name in provider._snapshot)
