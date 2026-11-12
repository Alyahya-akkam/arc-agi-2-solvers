"""P4.1: H's adapter providers on the CPU, and refine() with all three providers (B5 P4.1).

Acceptance (docs/tasks/P4.1.md, Work item 5, and B5 P4.1): each provider's protocol methods; only
cache misses are encoded (a counting encoder); fingerprints deterministic and order-independent;
rollback restores the weights bit for bit; refine() end to end with ``SGDProvider``,
``HyperProvider``, and ``HyperPolishProvider`` on DEV4-shaped stand-ins; a zero-initialized H
gives the base's logits through the decode path. Skipped without torch, transformers, peft, or
safetensors (CI's hyper-cpu job installs them).

Real: a tiny random Qwen3 (``arc26.hyper.train.tiny_base``) wrapped by PEFT as
``arc26.solver.load`` wraps the base (rsLoRA, dropout 0, no bias, q, k, v, o), P3.2's encoder and
cache, H, the export, ``arc26.ttt.fit`` and ``SGDProvider`` (the moved code: the dataset build, the
collator, ``UnslothFixedTrainer.compute_loss``), ``arc26.decode.calc_scores``, ``refine``,
``promote``, and the governor. Stubbed, because they cannot run on the CPU: unsloth
(``FastLanguageModel.for_training`` and ``for_inference`` as ``train()`` and ``eval()``; its
trainer as a stand-in that runs AdamW over the first ``max_steps`` views with the moved collator
and loss; ``UnslothTrainingArguments`` as ``dict``), datasets (``Dataset.from_list``), torch.cuda's
peak-memory statistics, and refine's decodes (round 0's candidates, the re-decode, the gate's
decode): a random tiny model decodes no grid, so they return fixed votes, and the gate passes
unless the candidate adapter's training set holds a wrong pseudo-pair. The synthetic tasks under
DEV4's ids are never competition data. Nothing touches the Hugging Face Hub.
"""

import copy
import dataclasses
import hashlib
import importlib
import importlib.machinery
import json
import logging
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

import yaml  # noqa: E402
from peft import PeftConfig, PeftModel, get_peft_model, get_peft_model_state_dict  # noqa: E402
from peft.tuners.lora import Linear as LoraLinear  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

import arc26  # noqa: E402
from arc26 import timing  # noqa: E402
from arc26.governor import GRADIENT, CostModel, Governor  # noqa: E402
from arc26.hyper import providers  # noqa: E402
from arc26.hyper import train as tr  # noqa: E402
from arc26.hyper.encoder import (  # noqa: E402
    EmbeddingCache,
    PairEncoder,
    pair_hash,
    view_descriptors,
)
from arc26.hyper.hypernet import BaseShapes, Hypernet, HypernetConfig  # noqa: E402
from arc26.hyper.inject import (  # noqa: E402
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    peft_config,
    to_peft_state_dict,
)
from arc26.hyper.providers import (  # noqa: E402
    ENCODE,
    H_FORWARD,
    POLISH_STEPS,
    SWAP,
    EncodeShape,
    HyperPolishProvider,
    HyperProvider,
    adapter_digest,
)
from arc26.loop import (  # noqa: E402
    CANDIDATE_TRAIN,
    FINGERPRINT,
    KEPT,
    NOTHING_TO_PROMOTE,
    PSEUDO,
    RESTORE,
    ROLLED_BACK,
    SNAPSHOT,
    Candidate,
    DemoSet,
    HeldOutGate,
    Pair,
    RoundLog,
    refine,
)
from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GOVERNOR = yaml.safe_load((REPO / "configs" / "governor.yaml").read_text(encoding="utf-8"))
TOY = REPO / "configs" / "train_hypernet_toy.yaml"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
# A tiny scope with H's structure: rsLoRA on q, k, v, o, s = 16 / sqrt(4) = 8, H's own scale.
RANK, ALPHA = 4, 16
LAYER = 2  # the encoder runs 2 of the tiny base's 4 layers
VIEWS = view_descriptors(1, 2)  # ADR-001 as amended: 8 views
MODEL_ID = "tiny-random-qwen3"
# The solver's learning rate (arc26.solver.load); the stand-in trainer reads it and max_steps.
TRAIN_ARGS = {"learning_rate": 5e-5, "num_train_epochs": 1, "seed": 42}
SGD_STEPS = 4  # s' of the SGD arm here: a short continuation keeps the tests fast
W = ((0,),)  # a wrong grid


# --- the stubs: unsloth, datasets, torch.cuda's statistics ---------------------------------------

TRAININGS = []  # one record per stand-in trainer run


def state(model):
    return get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)


def copy_state(model):
    return {key: tensor.detach().clone() for key, tensor in state(model).items()}


def same(a, b):
    """Bit for bit: the same keys, dtypes, shapes, and values."""
    return a.keys() == b.keys() and all(
        a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]) for key in a
    )


def digest(model):
    return adapter_digest(state(model))


class StubFastLanguageModel:
    """unsloth's mode switches on a PEFT model: training and evaluation mode."""

    @staticmethod
    def for_training(model):
        return model.train()

    @staticmethod
    def for_inference(model):
        return model.eval()


class PreparedDataset:
    """The trainer's prepared training set: the texts and their tokens, as trl prepares it."""

    def __init__(self, rows, tokenizer, field):
        self.columns = {field: [row[field] for row in rows]}
        self.columns["input_ids"] = [tokenizer.encode(row[field]) for row in rows]

    def __getitem__(self, name):
        return self.columns[name]


class StubTrainer:
    """unsloth's trainer on the CPU: AdamW over the adapter's trainable weights, one view per step
    (batch 1), the first ``max_steps`` views when the arguments set it, with the moved collator
    and ``UnslothFixedTrainer.compute_loss``. ``raise_after`` fails the run after that many
    steps, as an out-of-memory error mid-training would."""

    raise_after = None

    def __init__(
        self, model, tokenizer, data_collator, train_dataset, dataset_text_field, max_seq_length,
        args,
    ):  # fmt: skip
        self.model, self.data_collator, self.args = model, data_collator, args
        self.train_dataset = PreparedDataset(train_dataset.rows, tokenizer, dataset_text_field)
        self.label_smoother = None
        self.accelerator = types.SimpleNamespace(
            unwrap_model=lambda model, keep_fp32_wrapper: model, num_processes=1
        )

    def train(self):
        steps = self.args.get("max_steps", -1)
        views = self.train_dataset["input_ids"]
        rows = views if steps < 0 else views[:steps]
        TRAININGS.append(
            {
                "steps": steps,
                "samples": len(rows),
                "views": len(views),
                "before": digest(self.model),
            }
        )
        params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.args["learning_rate"])
        for step, ids in enumerate(rows):
            if self.raise_after is not None and step == self.raise_after:
                raise MemoryError("CUDA out of memory (stand-in fit)")
            loss = self.compute_loss(self.model, self.data_collator([{"input_ids": ids}]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return types.SimpleNamespace(global_step=len(rows))


class StubDataset:
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def from_list(cls, rows):
        return cls(rows)


def stub_module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    # transformers probes importlib.util.find_spec on its optional packages, which needs a spec.
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


@pytest.fixture(scope="module")
def ttt():
    """``arc26.ttt`` (and ``arc26.decode``) imported against the stubs; ``sys.modules``, the
    ``arc26`` package, logging, and torch.cuda are as before once this module's tests are done."""
    names = ("arc26.ttt", "arc26.decode")
    modules = {name: sys.modules.get(name) for name in names}
    attributes = {name: getattr(arc26, name.split(".")[1], None) for name in names}
    with pytest.MonkeyPatch.context() as patch:
        unsloth = stub_module(
            "unsloth", FastLanguageModel=StubFastLanguageModel, UnslothTrainer=StubTrainer,
            UnslothTrainingArguments=dict,
        )  # fmt: skip
        patch.setitem(sys.modules, "unsloth", unsloth)
        patch.setitem(sys.modules, "datasets", stub_module("datasets", Dataset=StubDataset))
        # fit's peak-memory statistics: reset_peak_memory_stats raises without a GPU.
        patch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
        patch.setattr(torch.cuda, "max_memory_allocated", lambda *args, **kwargs: 0)
        patch.setattr(torch.cuda, "empty_cache", lambda: None)
        for name in names:
            sys.modules.pop(name, None)
        level = logging.root.manager.disable
        try:
            module = importlib.import_module("arc26.ttt")
        finally:
            logging.disable(level)  # importing ttt disables logging at WARNING for the process
        try:
            yield module
        finally:
            for name in names:
                attribute = name.split(".")[1]
                if modules[name] is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = modules[name]
                if attributes[name] is None:
                    if hasattr(arc26, attribute):
                        delattr(arc26, attribute)
                else:
                    setattr(arc26, attribute, attributes[name])


# --- the live model, H, and the providers ---------------------------------------------------------


def set_lora_b(model, seed, std=0.3):
    """Non-zero ``lora_B``, as a previous fit leaves the adapter."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * std)


def live_model(lora_seed=1, rank=RANK, alpha=ALPHA, targets=TARGETS, perturb=None, rslora=True):
    """The live model as ``arc26.solver.load`` wraps the base: rsLoRA, dropout 0, no bias; PEFT's
    initial ``lora_B`` is 0 unless ``perturb`` seeds other values."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(lora_seed)
        values = {**peft_config(rank, alpha, list(targets)), "use_rslora": rslora}
        config = PeftConfig.from_peft_type(**values)
        model = get_peft_model(tr.tiny_base(0), config)
    if perturb is not None:
        set_lora_b(model, perturb)
    return model.eval()


def h_config(**changes):
    values = {
        "embed_dim": 64, "latent": 16, "head_in": 8, "dropout": 0.25, "rank": RANK,
        "alpha": ALPHA, "target_modules": TARGETS, "seed": 0,
    }  # fmt: skip
    values.update(changes)
    return HypernetConfig(**values)


def hypernet(model, trained=True, seed=5):
    """H for the live model's base. Trained: non-zero heads, so the adapter depends on z_bar and B
    is not 0, as after P3.5. Untrained: bias-HyperInit, B = 0 for every task (ADR-002)."""
    h = Hypernet(h_config(), BaseShapes.from_config(model.config))
    if trained:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for head in h.heads.values():
                head.weight.copy_(torch.randn(head.weight.shape, generator=generator) * 0.2)
                head.bias.copy_(torch.randn(head.bias.shape, generator=generator) * 0.05)
    return h


class CountingEncoder(PairEncoder):
    """P3.2's encoder, recording the pair hashes of each encode call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def encode_pairs(self, pairs, puzzle_id=None):
        self.calls.append([pair_hash(pair) for pair in pairs])
        return super().encode_pairs(pairs, puzzle_id)


def make_cache(root, encoder):
    return EmbeddingCache(
        root, model_id=MODEL_ID, layer=encoder.layer, pool="output", view_set_id=encoder.view_set_id
    )


def make_hyper(model, root, h=None, logger=None):
    """The H provider as a worker builds it: the encoder on the live PEFT model."""
    encoder = CountingEncoder(model, tr.standin_tokenizer(), layer=LAYER, views=VIEWS)
    h = hypernet(model) if h is None else h
    return HyperProvider(h, encoder, make_cache(root, encoder), timing_logger=logger)


def make_sgd(ttt, key, steps, logger=None):
    """``arc26.ttt.SGDProvider`` as ``arc26.loop._refine_puzzle`` builds it, on the stand-ins."""
    tokenizer = tr.standin_tokenizer()
    return ttt.SGDProvider(
        key=key,
        formatter=QwenFormatter(tokenizer),
        max_seq_length=8192,
        tokenizer=tokenizer,
        collator=ttt.QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False),
        train_args=TRAIN_ARGS,
        timing_logger=logger,
        rank=0,
        steps=steps,
    )


def make_provider(kind, model, root, ttt, key="0934a4d8", logger=None):
    if kind == "sgd":
        return make_sgd(ttt, key, SGD_STEPS, logger)
    hyper = make_hyper(model, root, logger=logger)
    if kind == "hyper":
        return hyper
    return HyperPolishProvider(hyper, make_sgd(ttt, key, POLISH_STEPS, logger))


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


# --- DEV4-shaped puzzles and the decodes' stand-ins -----------------------------------------------


def grid(*rows):
    return [list(row) for row in rows]


def identity(*rows):
    return {"input": grid(*rows), "output": grid(*rows)}


def swap(*rows):
    """Colors 1 and 2 exchanged."""
    table = {1: 2, 2: 1}
    return {"input": grid(*rows), "output": [[table.get(c, c) for c in row] for row in rows]}


# Four synthetic puzzles under DEV4's ids with DEV4's test layout (1, 2, 1, and 1 test inputs):
# three real pairs each, the last held out by the default rule; no pair, and no identity answer
# of a test input, occurs twice, so each puzzle's encodes are its own. Many colors, so fresh views
# are disjoint from the first decode's at once.
TASKS = {
    "0934a4d8": {
        "train": [
            identity([1, 2], [3, 4]), identity([5, 6, 7]), identity([1, 1, 2], [3, 3, 4]),
        ],
        "test": [{"input": grid([8, 9], [7, 6])}],
    },
    "36a08778": {
        "train": [
            identity([2, 3], [4, 5]), identity([6, 7, 8]), identity([3, 4, 5], [6, 7, 8]),
        ],
        "test": [{"input": grid([7, 8], [9, 6])}, {"input": grid([1, 2, 3, 4, 6])}],
    },
    "981571dc": {
        "train": [swap([1, 2, 3]), swap([4, 1, 2, 6]), swap([1, 2, 7], [8, 2, 1])],
        "test": [{"input": grid([2, 1], [5, 6])}],
    },
    "aa4ec2a5": {
        "train": [
            identity([3, 4], [5, 6]), identity([9, 6, 7]), identity([1, 3, 5], [7, 9, 2]),
        ],
        "test": [{"input": grid([2, 4, 6, 8, 9])}],
    },
}  # fmt: skip
# Round 0's votes: each test input's identity answer on 14 of the 16 views, except these, whose
# views split 8 to 8 (a tie: nothing to promote). A swap puzzle's identity answers are wrong.
TIED = {("36a08778", 1), ("aa4ec2a5", 0)}
SWAPPED = {"981571dc"}
OUTCOMES = {
    "0934a4d8": KEPT, "36a08778": KEPT, "981571dc": ROLLED_BACK, "aa4ec2a5": NOTHING_TO_PROMOTE,
}  # fmt: skip


def votes(*counts):
    """Synthetic view candidates: ``count`` candidates of each grid, in order."""
    found = []
    for answer, count in counts:
        for _ in range(count):
            index = len(found)
            found.append(Candidate(answer, 0.01 * index, (0.5,) * 8, f"v{index}"))
    return found


def round0_candidates(key, demos):
    return [
        votes((query, 8), (W, 8)) if (key, index) in TIED else votes((query, 14), (W, 2))
        for index, query in enumerate(demos.test_inputs)
    ]


def redecode(adapter, context, query, seed):
    """The re-decode under fresh views agrees with round 0's vote."""
    return votes((query, 14), (W, 2))


class GateDecode:
    """The gate's decode of the held-out input: its output, or a wrong grid when the candidate
    adapter's set holds a pseudo-pair of a swap puzzle; records the provider's fingerprint of the
    adapter it was handed."""

    def __init__(self, key, held_out, fingerprint):
        self.key, self.right, self.fingerprint = key, held_out.output, fingerprint
        self.seen = []

    def __call__(self, adapter, context, query, views):
        self.seen.append(self.fingerprint(adapter))
        taught = [pair for pair in context.pairs if pair.kind == PSEUDO]
        return votes((W if self.key in SWAPPED and taught else self.right, views))


class Clock:
    """The governor's clock, fixed."""

    def __call__(self):
        return 1000.0


def governor():
    return Governor.from_mapping(GOVERNOR, end_time=1e7, workers=1, clock=Clock())


def demo_set(key="0934a4d8"):
    return DemoSet.from_task(key, TASKS[key])


def pseudo_pair(demos, index=0):
    query = demos.test_inputs[index]
    return Pair(query, query, kind=PSEUDO, round_added=1, agreement=14 / 16, test_index=index)


# --- B5 acceptance: refine() end to end with all three providers on the DEV4 stand-ins ------------

KINDS = ("sgd", "hyper", "polish")


def run_dev4(kind, ttt, cache_root):
    """One worker's DEV4 stand-ins with one provider: round 0's adapter (SGD: ``fit`` from the
    default weights; H: ``adapt``, $H(D_0)$; the polish: $\\mathrm{polish}(H(D_0), s)$), then
    one ``refine`` round per puzzle, on one live model, the puzzles in order; the H providers
    share one H, encoder, and cache, as a worker's would."""
    model = live_model()
    default = copy_state(model)
    hyper = None if kind == "sgd" else make_hyper(model, cache_root)
    puzzles = {}
    for key, task in TASKS.items():
        demos = DemoSet.from_task(key, task)
        training = demos.without_held_out()
        start = 0 if hyper is None else len(hyper.encoder.calls)
        if kind == "sgd":
            provider = make_sgd(ttt, key, SGD_STEPS)
            model, _ = ttt.fit(
                model=model,
                default_weights=default,
                arc_test_set=ArcDataset(queries={key: training.task()}, replies={}),
                key=key,
                formatter=provider.formatter,
                max_seq_length=8192,
                tokenizer=provider.tokenizer,
                collator=provider.collator,
                train_args={**TRAIN_ARGS, "max_steps": 8},
                timing_logger=None,
                rank=0,
            )
        else:
            provider = hyper if kind == "hyper" else HyperPolishProvider(
                hyper, make_sgd(ttt, key, POLISH_STEPS)
            )  # fmt: skip
            model = provider.adapt(training, model, 600.0)
        round0 = copy_state(model)
        round0_fingerprint = provider.fingerprint(model)
        encoded = start if hyper is None else len(hyper.encoder.calls)
        trained = len(TRAININGS)
        gate = GateDecode(key, demos.held_out, provider.fingerprint)
        log = RoundLog(None, "rank0")
        gov = governor()
        budget = gov.puzzle_budget(0)
        result, adapter = refine(
            demos,
            model,
            provider,
            HeldOutGate(lambda s: None, gate),
            gov,
            candidates=round0_candidates(key, demos),
            redecode=redecode,
            deadline=gov.clock() + budget,
            budget=budget,
            log=log,
        )
        puzzles[key] = types.SimpleNamespace(
            demos=demos,
            state=result,
            model=model,
            adapter=adapter,
            round0=round0,
            round0_fingerprint=round0_fingerprint,
            after=copy_state(adapter),
            after_fingerprint=provider.fingerprint(adapter),
            log=log,
            gate=gate,
            provider=provider,
            round0_encodes=[] if hyper is None else hyper.encoder.calls[start:encoded],
            round_encodes=[] if hyper is None else hyper.encoder.calls[encoded:],
            trainings=TRAININGS[trained:],
        )
        model = adapter
    return types.SimpleNamespace(kind=kind, puzzles=puzzles, hyper=hyper)


@pytest.fixture(scope="module")
def dev4_runs(ttt, tmp_path_factory):
    """Each provider's DEV4 run, once for this module's tests."""
    return {kind: run_dev4(kind, ttt, tmp_path_factory.mktemp(f"cache-{kind}")) for kind in KINDS}


@pytest.mark.parametrize("kind", KINDS)
def test_refine_end_to_end_with_all_three_providers_on_dev4_stand_ins(kind, dev4_runs):
    """One round per puzzle: kept twice, rolled back after the wrong pseudo-pair, and nothing to
    promote; the state's version follows, and the gate checked the provider's candidate."""
    dev4_run = dev4_runs[kind]
    for key, run in dev4_run.puzzles.items():
        (record,) = run.log.rounds
        assert record.outcome == OUTCOMES[key], (dev4_run.kind, key, record)
        assert run.adapter is run.model  # every provider's adapter is the live model
        if record.outcome == KEPT:
            assert run.state.version == run.demos.version + 1 == record.next_version
            assert record.promoted == (0,) and run.state.pairs[-1].kind == PSEUDO
            assert record.weights_before == run.round0_fingerprint != record.weights_after
            assert record.weights_after == run.after_fingerprint == run.gate.seen[-1]
        elif record.outcome == ROLLED_BACK:
            assert run.state is run.demos and run.state.version == 0
            assert run.gate.seen[-1] not in (record.weights_before, record.weights_after)
        else:
            assert run.state is run.demos and run.log.plans == [] and run.gate.seen == []


@pytest.mark.parametrize("kind", KINDS)
def test_dev4_rollback_restores_round_0_weights_bit_for_bit(kind, dev4_runs):
    """B5 P2.3's rollback on every provider: the wrong pseudo-pair's candidate fails the gate,
    and the live adapter is round 0's again, bit for bit (the record's digests agree)."""
    dev4_run = dev4_runs[kind]
    run = dev4_run.puzzles["981571dc"]
    (record,) = run.log.rounds
    assert same(run.after, run.round0)
    assert record.restored is True and record.weights_before == record.weights_after
    assert record.weights_after == run.round0_fingerprint == run.after_fingerprint
    untouched = dev4_run.puzzles["aa4ec2a5"]
    assert same(untouched.after, untouched.round0) and untouched.trainings == []


@pytest.mark.parametrize("kind", KINDS)
def test_dev4_kept_rounds_hold_each_providers_candidate(kind, dev4_runs, ttt):
    """SGD continues round 0's adapter for s' steps; H recomputes $H(D_1)$; the polish starts
    from $H(D_1)$ and trains s steps, and gives the same weights from another initial adapter
    (B2 Q4: recomputed from H, never continued from init)."""
    dev4_run = dev4_runs[kind]
    for key in ("0934a4d8", "36a08778"):
        run = dev4_run.puzzles[key]
        enlarged = run.state.without_held_out()
        if dev4_run.kind == "sgd":
            (training,) = run.trainings
            assert training["before"] == adapter_digest(run.round0)
            assert (training["steps"], training["samples"], training["views"]) == (
                SGD_STEPS, SGD_STEPS, 128
            )  # fmt: skip
            continue
        emitted = dev4_run.hyper.emit(enlarged)
        if dev4_run.kind == "hyper":
            assert run.trainings == [] and emitted.digest() == adapter_digest(run.after)
            continue
        (training,) = run.trainings
        assert training["before"] == emitted.digest() != adapter_digest(run.after)
        assert (training["steps"], training["samples"]) == (POLISH_STEPS, POLISH_STEPS)
        other = live_model(lora_seed=7, perturb=11)
        assert not same(copy_state(other), run.round0)
        again = HyperPolishProvider(dev4_run.hyper, make_sgd(ttt, key, POLISH_STEPS))
        again.adapt(enlarged, other, 600.0)
        assert same(copy_state(other), run.after)


@pytest.mark.parametrize("kind", ["hyper", "polish"])
def test_dev4_h_providers_encode_only_the_cache_misses(kind, dev4_runs):
    """Round 0 encodes the two training pairs, each alone; the round encodes only its
    pseudo-pair; the held-out pair is never encoded (ADR-013)."""
    dev4_run = dev4_runs[kind]
    for key, run in dev4_run.puzzles.items():
        training = run.demos.without_held_out()
        expected = sorted(pair_hash(pair.task_pair()) for pair in training.pairs)
        assert run.round0_encodes == [[h] for h in expected]
        if OUTCOMES[key] == NOTHING_TO_PROMOTE:
            assert run.round_encodes == []
        else:
            assert run.round_encodes == [[pair_hash(pseudo_pair(run.demos).task_pair())]]
    held_out = {pair_hash(run.demos.held_out.task_pair()) for run in dev4_run.puzzles.values()}
    assert not held_out & {h for call in dev4_run.hyper.encoder.calls for h in call}


@pytest.mark.parametrize("kind", KINDS)
def test_dev4_plans_price_each_providers_fit(kind, dev4_runs):
    """The governor prices H's round with no train term, the polish's and SGD's with s and s'
    steps of the fit's views (``fit_shape``)."""
    dev4_run = dev4_runs[kind]
    for key in ("0934a4d8", "36a08778", "981571dc"):
        (plan,) = dev4_run.puzzles[key].log.plans
        assert plan["decision"] == GRADIENT
        tokens, steps = plan["shape"]["train_tokens"], plan["shape"]["train_steps"]
        if dev4_run.kind == "hyper":
            assert (tokens, steps) == (0, 0)
        else:
            assert tokens > 0 and steps == (SGD_STEPS if dev4_run.kind == "sgd" else POLISH_STEPS)


# --- acceptance: the protocol, the cache, fingerprints, rollback, the decode path -----------------


@pytest.mark.parametrize("kind", KINDS)
def test_each_provider_has_the_protocol_methods(kind, ttt, tmp_path):
    """``adapt`` returns the live model with the candidate adapter, ``fingerprint`` is stable,
    ``fit_shape`` is the train term, ``rollback`` gives back the previous weights."""
    model = live_model(perturb=3)
    provider = make_provider(kind, model, tmp_path, ttt)
    for name in ("adapt", "rollback", "fingerprint", "fit_shape"):
        assert callable(getattr(provider, name))
    training = demo_set().without_held_out()
    before = copy_state(model)
    first = provider.fingerprint(model)
    assert provider.fingerprint(model) == first
    if kind != "sgd":  # SGDProvider's is a blake2b digest (ADR-015); H's the sha256 of ADR-017
        assert first == adapter_digest(before)
    assert provider.adapt(training, model, 100.0) is model
    assert not same(copy_state(model), before) and provider.fingerprint(model) != first
    tokens, steps = provider.fit_shape(training)
    if kind == "hyper":
        assert (tokens, steps) == (0, 0)
    else:
        sgd = make_sgd(ttt, "0934a4d8", SGD_STEPS if kind == "sgd" else POLISH_STEPS)
        assert (tokens, steps) == sgd.fit_shape(training) and tokens > 0
    assert provider.rollback(model) is model
    assert same(copy_state(model), before) and provider.fingerprint(model) == first


def test_only_cache_misses_are_encoded_counting_encoder(tmp_path):
    """Each missing pair is encoded once, alone; a cached pair is never encoded again, also by
    another provider on the same cache (another round, puzzle, or worker launch)."""
    model = live_model()
    provider = make_hyper(model, tmp_path / "cache")
    demos = demo_set()
    training = demos.without_held_out()
    hashes = sorted(pair_hash(pair.task_pair()) for pair in training.pairs)
    provider.emit(training)
    assert provider.encoder.calls == [[h] for h in hashes]
    provider.emit(training)
    provider.adapt(training, model, 100.0)
    assert len(provider.encoder.calls) == 2
    enlarged = training.enlarged([pseudo_pair(demos)])
    provider.adapt(enlarged, model, 100.0)
    assert provider.encoder.calls[2:] == [[pair_hash(pseudo_pair(demos).task_pair())]]
    again = make_hyper(model, tmp_path / "cache")
    again.emit(enlarged)
    assert again.encoder.calls == []
    # A pair two puzzles share is encoded once.
    other = DemoSet("00000002", (training.pairs[0], pseudo_pair(demos)), None, demos.test_inputs)
    again.emit(other)
    assert again.encoder.calls == []


def test_fingerprints_are_deterministic_and_order_independent(tmp_path):
    """B5 P4.1: emitted adapter hashes are deterministic for a fixed DemoSet, whatever the pairs'
    order, the cache's history, or H's mode; the live model's digest is the emitted one's."""
    model = live_model()
    h = hypernet(model)
    demos = demo_set()
    training = demos.without_held_out()
    enlarged = training.enlarged([pseudo_pair(demos)])
    first = make_hyper(model, tmp_path / "a", h=h).emit(enlarged).digest()
    shuffled = dataclasses.replace(enlarged, pairs=(enlarged.pairs[-1], *training.pairs[::-1]))
    assert make_hyper(model, tmp_path / "b", h=h).emit(shuffled).digest() == first
    history = make_hyper(model, tmp_path / "c", h=h)
    history.emit(dataclasses.replace(training, pairs=training.pairs[1:]))
    assert history.emit(enlarged).digest() == first
    h.train()  # dropout 0.25 in training mode: emit runs H in evaluation mode
    assert history.emit(enlarged).digest() == first and not h.training
    provider = make_hyper(model, tmp_path / "d", h=h)
    provider.adapt(shuffled, model, 100.0)
    assert provider.fingerprint(model) == first == provider.fingerprint(provider.emit(enlarged))
    # Not vacuous: another set gives another adapter.
    assert make_hyper(model, tmp_path / "e", h=h).emit(training).digest() != first


def test_the_fingerprint_is_sha256_over_the_tensors_in_key_order():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.ones(3, 2, dtype=torch.bfloat16)
    expected = hashlib.sha256()
    for key, tensor in (("x.lora_A.weight", a), ("y.lora_B.weight", b)):
        expected.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        expected.update(tensor.view(torch.uint8).numpy().tobytes())
    found = adapter_digest({"y.lora_B.weight": b, "x.lora_A.weight": a})
    assert found == expected.hexdigest()
    assert adapter_digest({"x.lora_A.weight": a.to(torch.float64), "y.lora_B.weight": b}) != found


@pytest.mark.parametrize("kind", KINDS)
def test_rollback_restores_weights_bit_for_bit(kind, ttt, tmp_path):
    """After a candidate the gate rejects, every provider gives back the previous adapter bit for
    bit, also when that adapter was an earlier round's; a rollback without an adapt is a no-op."""
    model = live_model(perturb=3)
    provider = make_provider(kind, model, tmp_path, ttt)
    previous = copy_state(model)
    assert provider.rollback(model) is model and same(copy_state(model), previous)
    demos = demo_set()
    training = demos.without_held_out()
    provider.adapt(training, model, 100.0)
    kept = copy_state(model)  # round 1 kept
    provider.adapt(training.enlarged([pseudo_pair(demos)]), model, 100.0)
    assert not same(copy_state(model), kept)
    provider.rollback(model)
    assert same(copy_state(model), kept) and not model.training


def decode_inputs(training):
    """A puzzle prompt's tokens, and two rescoring queries with their replies for calc_scores."""
    tokenizer = tr.standin_tokenizer()
    formatter = QwenFormatter(tokenizer)
    task = training.task()
    prompt = formatter.fmt_train(task["train"]) + formatter.fmt_query(task["test"])
    answers = [formatter.fmt_reply([grid([8, 9], [7, 6])]), formatter.fmt_reply([grid([9])])]
    return tokenizer, torch.tensor([tokenizer.encode(prompt)]), [prompt] * 2, answers


def test_zero_initialized_h_gives_the_base_logits_through_the_decode_path(ttt, tmp_path):
    """H at initialization (B = 0, ADR-002) swapped into the live model, whatever adapter it held,
    gives the base's logits and rescoring NLLs (``arc26.decode.calc_scores``) bit for bit; so does
    the saved Adapter loaded by ``PeftModel.from_pretrained``."""
    decode = importlib.import_module("arc26.decode")
    base = tr.tiny_base(0)
    model = live_model(perturb=3)  # a previous puzzle's adapter
    training = demo_set().without_held_out()
    tokenizer, ids, queries, answers = decode_inputs(training)
    assert not torch.equal(logits(model, ids), logits(base, ids))  # not vacuous
    provider = make_hyper(model, tmp_path / "cache", h=hypernet(model, trained=False))
    provider.adapt(training, model, 100.0)
    assert torch.equal(logits(model, ids), logits(base, ids))
    found = decode.calc_scores(queries, answers, tokenizer, model)
    assert found == decode.calc_scores(queries, answers, tokenizer, base)
    adapter = provider.emit(training)
    adapter.save(tmp_path / "adapter")
    loaded = PeftModel.from_pretrained(copy.deepcopy(base), str(tmp_path / "adapter"))
    assert torch.equal(logits(loaded, ids), logits(base, ids))


# --- HyperProvider -------------------------------------------------------------------------------


def test_the_encoder_reads_the_frozen_base_whatever_the_live_adapter(tmp_path):
    """The live model's adapter is disabled while pairs are encoded (ADR-001; P3.5 encoded on the
    base): each cached embedding equals the plain base's, bit for bit."""
    model = live_model(perturb=3)
    provider = make_hyper(model, tmp_path / "cache")
    training = demo_set().without_held_out()
    provider.emit(training)
    pairs = [pair.task_pair() for pair in training.pairs]
    cached = provider.cache.get([pair_hash(pair) for pair in pairs])
    plain = PairEncoder(tr.tiny_base(0), tr.standin_tokenizer(), layer=LAYER, views=VIEWS)
    assert torch.equal(cached, torch.cat([plain.encode_pairs([pair]) for pair in pairs]))
    active = PairEncoder(model, tr.standin_tokenizer(), layer=LAYER, views=VIEWS)
    assert not torch.equal(cached, active.encode_pairs(pairs))  # the adapter moves them


def test_adapt_swaps_h_adapter_into_the_live_model_and_ignores_init(tmp_path):
    """The live adapter holds exactly the emitted Adapter, from any initial weights (B2 Q4)."""
    one, two = live_model(perturb=3), live_model(lora_seed=9, perturb=4)
    h = hypernet(one)
    demos = demo_set()
    training = demos.without_held_out()
    first = make_hyper(one, tmp_path / "a", h=h)
    adapter = first.emit(training)
    first.adapt(training, one, 100.0)
    make_hyper(two, tmp_path / "b", h=h).adapt(training, two, 100.0)
    assert same(copy_state(one), copy_state(two))
    assert same(copy_state(one), adapter.state)
    assert (adapter.rank, adapter.alpha, adapter.target_modules) == (RANK, ALPHA, TARGETS)
    assert all(tensor.dtype == torch.float32 for tensor in adapter.state.values())


def test_the_held_out_pair_never_reaches_the_encoder_or_h(tmp_path):
    """Only ``DemoSet.pairs`` is read: a state that still carries its held-out pair gives the
    adapter of the state without it, and the pair is never encoded (ADR-013)."""
    model = live_model()
    provider = make_hyper(model, tmp_path / "cache")
    demos = demo_set()
    assert demos.held_out is not None
    with_pair = provider.emit(demos).digest()
    assert with_pair == provider.emit(demos.without_held_out()).digest()
    encoded = {h for call in provider.encoder.calls for h in call}
    assert pair_hash(demos.held_out.task_pair()) not in encoded


def test_a_repeated_pair_is_encoded_once_and_counted_twice(tmp_path):
    """z_bar is the mean over DemoSet.pairs as listed, repeats included, as P3.5's training
    takes it over an episode's demonstrations: (2 e_1 + e_2) / 3, not the mean of the set."""
    model = live_model()
    provider = make_hyper(model, tmp_path / "cache")
    training = demo_set().without_held_out()
    first, second = training.pairs
    repeated = dataclasses.replace(training, pairs=(first, second, first))
    adapter = provider.emit(repeated)
    one, two = (pair_hash(pair.task_pair()) for pair in (first, second))
    assert provider.encoder.calls == [[h] for h in sorted((one, two))]  # each pair once
    z_bar = provider.cache.get(sorted((one, two, one))).mean(dim=0)
    with torch.no_grad():
        expected = to_peft_state_dict(provider.hypernet(z_bar[None])[0])
    assert same(adapter.state, expected)
    assert adapter.digest() != provider.emit(training).digest()  # not the set's mean
    assert len(provider.encoder.calls) == 2


def test_two_real_pairs_with_one_held_out_and_an_empty_pair_set(tmp_path):
    """k = 2 under the k2 flag leaves one pair: H reads it alone; no pair at all raises."""
    model = live_model()
    provider = make_hyper(model, tmp_path / "cache")
    task = {"train": TASKS["0934a4d8"]["train"][:2], "test": TASKS["0934a4d8"]["test"]}
    demos = DemoSet.from_task("0934a4d8", task)
    assert len(demos.pairs) == 1 and demos.held_out is not None
    provider.adapt(demos.without_held_out(), model, 0.0)
    assert provider.encoder.calls == [[pair_hash(demos.pairs[0].task_pair())]]
    empty = DemoSet("0934a4d8", (), None, demos.test_inputs)
    before = copy_state(model)
    with pytest.raises(ValueError, match="no pair"):
        provider.adapt(empty, model, 100.0)
    assert provider.rollback(model) is model and same(copy_state(model), before)


@pytest.mark.parametrize(
    "rank, alpha, targets, rslora",
    [
        (8, ALPHA, TARGETS, True),
        (RANK, 32, TARGETS, True),
        (RANK, ALPHA, (*TARGETS, "gate_proj"), True),
        (RANK, ALPHA, TARGETS, False),
    ],
    ids=["rank", "alpha", "targets", "rslora"],
)
def test_a_live_adapter_of_another_scope_is_refused(rank, alpha, targets, rslora, tmp_path):
    """H's adapter loads only into an adapter of H's scope; nothing changes otherwise. Without
    rsLoRA the keys and shapes are H's, but PEFT scales the delta by alpha / r, not
    alpha / sqrt(r): half of H's scale here, a quarter at H's r = 16."""
    model = live_model(rank=rank, alpha=alpha, targets=targets, perturb=3, rslora=rslora)
    scales = {m.scaling["default"] for m in model.modules() if isinstance(m, LoraLinear)}
    assert scales == {alpha / (rank**0.5 if rslora else rank)}
    reference = live_model()
    provider = make_hyper(reference, tmp_path / "cache")
    before = copy_state(model)
    with pytest.raises(ValueError, match="live adapter"):
        provider.adapt(demo_set().without_held_out(), model, 100.0)
    assert provider.rollback(model) is model and same(copy_state(model), before)
    assert provider.encoder.calls == []  # refused before any encode


def test_a_live_adapter_on_another_base_is_refused(tmp_path):
    """H's scope in the config but another base (2 layers, not 4): the tensors' keys differ."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        config = Qwen3Config(
            vocab_size=16, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        )  # fmt: skip
        lora = PeftConfig.from_peft_type(**peft_config(RANK, ALPHA, list(TARGETS)))
        model = get_peft_model(Qwen3ForCausalLM(config).eval(), lora)
    provider = make_hyper(live_model(), tmp_path / "cache")
    before = copy_state(model)
    with pytest.raises(ValueError, match="tensors are not H's"):
        provider.adapt(demo_set().without_held_out(), model, 100.0)
    assert same(copy_state(model), before) and provider.encoder.calls == []


def test_a_model_without_the_peft_adapter_is_refused(tmp_path):
    model = live_model()
    provider = make_hyper(model, tmp_path / "cache")
    with pytest.raises(TypeError, match="PEFT model"):
        provider.adapt(demo_set().without_held_out(), tr.tiny_base(0), 100.0)


def test_the_parts_must_agree(tmp_path):
    """H's width, the cache's namespace, and an encoder that can see the frozen base."""
    model = live_model()
    encoder = PairEncoder(model, tr.standin_tokenizer(), layer=LAYER, views=VIEWS)
    cache = make_cache(tmp_path / "cache", encoder)
    wide = Hypernet(h_config(embed_dim=32), BaseShapes.from_config(model.config))
    with pytest.raises(ValueError, match="width"):
        HyperProvider(wide, encoder, cache)
    other = PairEncoder(model, tr.standin_tokenizer(), layer=1, views=VIEWS)
    with pytest.raises(ValueError, match="cache holds"):
        HyperProvider(hypernet(model), other, cache)
    inner = PairEncoder(model.base_model.model, tr.standin_tokenizer(), layer=LAYER, views=VIEWS)
    with pytest.raises(ValueError, match="cannot disable"):
        HyperProvider(hypernet(model), inner, cache)


def test_a_swap_that_raised_midway_is_rolled_back(tmp_path, monkeypatch):
    """A load that changed a weight and then raised: the rollback restores the snapshot."""
    model = live_model(perturb=3)
    provider = make_hyper(model, tmp_path / "cache")
    before = copy_state(model)
    real = providers.set_peft_model_state_dict

    def failing(target, weights, adapter_name):
        with torch.no_grad():
            next(p for n, p in target.named_parameters() if "lora_B" in n).add_(1.0)
        raise RuntimeError("CUDA error (stand-in load)")

    monkeypatch.setattr(providers, "set_peft_model_state_dict", failing)
    with pytest.raises(RuntimeError, match="stand-in load"):
        provider.adapt(demo_set().without_held_out(), model, 100.0)
    assert not same(copy_state(model), before)
    monkeypatch.setattr(providers, "set_peft_model_state_dict", real)
    provider.rollback(model)
    assert same(copy_state(model), before)


def test_encode_shape_counts_the_tokens_of_the_cache_misses(tmp_path):
    """A 30 by 30 pair is 1,866 tokens under each of the 8 views (``docs/modules/encoder.md``);
    once encoded, nothing is left to price; the encode record carries the same tokens."""
    logger = timing.TimingLogger(tmp_path / "timing", "rank0")
    model = live_model()
    provider = make_hyper(model, tmp_path / "cache", logger=logger)
    big = [[(row * 30 + col) % 10 for col in range(30)] for row in range(30)]
    flipped = [row[::-1] for row in big]
    demos = DemoSet("00000003", (Pair(big, flipped),), None, (big,))
    assert provider.encode_shape(demos) == EncodeShape(pairs=1, prefills=8, tokens=8 * 1866)
    provider.adapt(demos, model, 100.0)
    (record,) = [r for r in timing.read_records([logger.path])[0] if r.op == ENCODE]
    assert record.tokens == 8 * 1866 and record.puzzle_id == "00000003"
    assert provider.encode_shape(demos) == EncodeShape(pairs=0, prefills=0, tokens=0)
    assert provider.cache.get([pair_hash({"input": big, "output": flipped})]).isfinite().all()


def test_h_prices_no_train_term(tmp_path):
    """``fit_shape`` is (0, 0), which the governor's train model prices at 0 seconds, so a round
    of H costs its decodes; the encode is ``encode_shape``'s, for P4.3 (ADR-017)."""
    model = live_model()
    provider = make_hyper(model, tmp_path / "cache")
    training = demo_set().without_held_out()
    tokens, steps = provider.fit_shape(training)
    cost = CostModel.from_mapping(GOVERNOR)
    assert (tokens, steps) == (0, 0) and cost.estimate(timing.TRAIN, tokens, steps=steps) == 0.0
    costs = cost.round_costs(train_tokens=tokens, train_steps=steps, test_batches=4, gate_batches=4)
    assert costs[GRADIENT] == cost.decode_cost(8) + (1 + cost.margin) * cost.overhead
    assert provider.encode_shape(training).pairs == 2


def test_timing_records_name_the_ops_of_h_and_the_polish(ttt, tmp_path):
    logger = timing.TimingLogger(tmp_path / "timing", "rank0")
    model = live_model()
    provider = make_hyper(model, tmp_path / "cache", logger=logger)
    training = demo_set().without_held_out()
    shape = provider.encode_shape(training)
    provider.fingerprint(model)
    provider.adapt(training, model, 100.0)
    provider.rollback(model)
    records = timing.read_records([logger.path])[0]
    assert [r.op for r in records] == [FINGERPRINT, ENCODE, H_FORWARD, SNAPSHOT, SWAP, RESTORE]
    assert records[0].puzzle_id is None and {r.puzzle_id for r in records[1:]} == {"0934a4d8"}
    assert records[1].tokens == shape.tokens > 0
    polish = HyperPolishProvider(provider, make_sgd(ttt, "0934a4d8", POLISH_STEPS, logger))
    polish.adapt(training, model, 100.0)
    ops = [r.op for r in timing.read_records([logger.path])[0][len(records) :]]
    assert ops == [H_FORWARD, SNAPSHOT, SWAP, SNAPSHOT, CANDIDATE_TRAIN]  # pairs cached


def test_from_checkpoint_reads_h_and_the_encoder_settings_of_training(tmp_path):
    """H's weights, the encoder's layer, views, and batch, and the cache's model id come from the
    P3.5 checkpoint, so test time encodes as training did."""
    model = live_model()
    h = hypernet(model)
    raw = yaml.safe_load(TOY.read_text(encoding="utf-8"))
    raw.pop("smoke")
    settings = tr.EncoderSettings(
        layer=1, pool="output", n_perm=1, view_seed=2, batch_tokens=4096, chunk_pairs=8
    )
    cfg = dataclasses.replace(
        tr.TrainConfig.from_dict(raw), hypernet=h_config(), encoder=settings, model_id=MODEL_ID
    )
    payload = {"schema": tr.CHECKPOINT_SCHEMA, "step": 3, "config": cfg.to_dict()}
    tr.save_checkpoint(tmp_path / "ckpt", {**payload, "hypernet": h.state_dict()}, keep=2)
    tokenizer = tr.standin_tokenizer()
    provider = HyperProvider.from_checkpoint(
        tmp_path / "ckpt", model=model, tokenizer=tokenizer, cache_root=tmp_path / "cache"
    )
    encoder = provider.encoder
    assert (encoder.layer, encoder.pool, encoder.views, encoder.batch_tokens) == (
        1, "output", view_descriptors(1, 2), 4096
    )  # fmt: skip
    assert provider.cache.namespace["model_id"] == MODEL_ID and not provider.hypernet.training
    by_hand_encoder = PairEncoder(model, tokenizer, layer=1, views=view_descriptors(1, 2))
    by_hand = HyperProvider(h, by_hand_encoder, make_cache(tmp_path / "other", by_hand_encoder))
    training = demo_set().without_held_out()
    assert provider.emit(training).digest() == by_hand.emit(training).digest()
    with pytest.raises(FileNotFoundError):
        HyperProvider.from_checkpoint(
            tmp_path / "none", model=model, tokenizer=tokenizer, cache_root=tmp_path / "cache"
        )


def test_the_adapter_is_saved_in_b4s_format(tmp_path):
    model = live_model()
    adapter = make_hyper(model, tmp_path / "cache").emit(demo_set().without_held_out())
    adapter.save(tmp_path / "adapter")
    tensors = load_file(tmp_path / "adapter" / ADAPTER_WEIGHTS)
    assert tensors.keys() == adapter.state.keys()
    assert all(torch.equal(tensors[key], adapter.state[key]) for key in tensors)
    config = json.loads((tmp_path / "adapter" / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    assert (config["r"], config["lora_alpha"], config["use_rslora"]) == (RANK, ALPHA, True)
    assert config["target_modules"] == list(TARGETS)


# --- HyperPolishProvider -------------------------------------------------------------------------


def test_the_polish_recomputes_from_h_never_from_init(ttt, tmp_path):
    """Two live models with different adapters give one polished adapter: s steps of the fit
    from $H(D)$, which they then differ from."""
    one, two = live_model(perturb=3), live_model(lora_seed=9, perturb=4)
    h = hypernet(one)
    training = demo_set().without_held_out()
    first = make_hyper(one, tmp_path / "a", h=h)
    emitted = first.emit(training).digest()
    start = len(TRAININGS)
    HyperPolishProvider(first, make_sgd(ttt, "0934a4d8", POLISH_STEPS)).adapt(training, one, 1.0)
    second = make_hyper(two, tmp_path / "b", h=h)
    HyperPolishProvider(second, make_sgd(ttt, "0934a4d8", POLISH_STEPS)).adapt(training, two, 1.0)
    assert same(copy_state(one), copy_state(two)) and digest(one) != emitted
    assert [t["before"] for t in TRAININGS[start:]] == [emitted, emitted]
    assert [t["samples"] for t in TRAININGS[start:]] == [POLISH_STEPS] * 2


def test_the_polish_rollback_after_a_fit_that_raised_mid_training(ttt, tmp_path, monkeypatch):
    """The fit trained two steps and raised: H's adapter, then the round's starting weights, come
    back, and the model is in inference mode."""
    model = live_model(perturb=3)
    polish = make_provider("polish", model, tmp_path, ttt)
    before = copy_state(model)
    monkeypatch.setattr(StubTrainer, "raise_after", 2)
    with pytest.raises(MemoryError, match="stand-in fit"):
        polish.adapt(demo_set().without_held_out(), model, 100.0)
    assert model.training and not same(copy_state(model), before)
    polish.rollback(model)
    assert same(copy_state(model), before) and not model.training


def test_the_polish_rollback_after_h_raised_keeps_the_earlier_round(ttt, tmp_path, monkeypatch):
    """Round 1 kept a polished adapter; round 2's encode raises before any polish: the rollback
    must not load round 1's copy of $H(D_1)$ (the SGDProvider's) over round 1's adapter."""
    model = live_model()
    polish = make_provider("polish", model, tmp_path, ttt)
    demos = demo_set()
    training = demos.without_held_out()
    polish.adapt(training, model, 100.0)
    kept = copy_state(model)

    def raising(pairs, puzzle_id=None):
        raise MemoryError("CUDA out of memory (stand-in encode)")

    monkeypatch.setattr(polish.hyper.encoder, "encode_pairs", raising)
    with pytest.raises(MemoryError, match="stand-in encode"):
        polish.adapt(training.enlarged([pseudo_pair(demos)]), model, 100.0)
    polish.rollback(model)
    assert same(copy_state(model), kept)


class FakeSGD:
    """The polish's parts as the composition sees them: adapt, rollback, fit_shape, steps."""

    steps = 3

    def __init__(self):
        self.budgets = []

    def adapt(self, demos, init, budget_s):
        self.budgets.append(budget_s)
        return init

    def rollback(self, previous):
        return previous

    def fit_shape(self, demos):
        return 11, self.steps


def test_the_polish_gets_what_is_left_of_the_budget(tmp_path):
    """H's step comes out of the budget; the fit's share is never negative (a timeout)."""
    model = live_model()
    sgd = FakeSGD()
    polish = HyperPolishProvider(make_hyper(model, tmp_path / "cache"), sgd)
    training = demo_set().without_held_out()
    polish.adapt(training, model, 0.0)
    polish.adapt(training, model, 50.0)
    assert sgd.budgets[0] == 0.0 and 0.0 <= sgd.budgets[1] <= 50.0
    assert polish.fit_shape(training) == (11, 3) and polish.steps == 3
    assert polish.encode_shape(training) == EncodeShape(0, 0, 0)  # encoded by the adapts


def test_the_polish_provider_checks_its_parts(tmp_path):
    model = live_model()
    hyper = make_hyper(model, tmp_path / "cache")
    with pytest.raises(TypeError, match="HyperProvider"):
        HyperPolishProvider(object(), FakeSGD())
    no_steps = FakeSGD()
    no_steps.steps = 3.0
    with pytest.raises(TypeError, match="steps"):
        HyperPolishProvider(hyper, no_steps)
    with pytest.raises(TypeError, match="rollback"):
        HyperPolishProvider(hyper, types.SimpleNamespace(adapt=len, fit_shape=len, steps=3))


def test_the_default_polish_is_ten_percent_of_an_epoch():
    """B2 Q3: s about 10% of the notebook's 128 one-epoch steps, rounded up."""
    assert POLISH_STEPS == 13


# --- the solver's bf16 cast ----------------------------------------------------------------------


def cast_as_solver_load(model):
    """``arc26.solver.load``'s cast: every float32 parameter to bf16, the adapter's included."""
    for _, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
    return model


@pytest.mark.parametrize("kind", ["hyper", "polish"])
def test_under_the_bf16_cast_swaps_and_rollbacks_stay_bit_for_bit(kind, ttt, tmp_path):
    """Under solver.load's cast the swap rounds H's float32 export to bf16, exactly the cast of the
    emitted tensors, so the live digest is not the emitted one's (ADR-017); each provider's
    rollback gives the bf16 weights back bit for bit, after round 1 and after a later round."""
    model = cast_as_solver_load(live_model(perturb=3))
    provider = make_provider(kind, model, tmp_path, ttt)
    hyper = provider if kind == "hyper" else provider.hyper
    demos = demo_set()
    training = demos.without_held_out()
    before = copy_state(model)
    assert {tensor.dtype for tensor in before.values()} == {torch.bfloat16}
    provider.adapt(training, model, 100.0)
    emitted = hyper.emit(training)
    if kind == "hyper":
        rounded = {key: tensor.to(torch.bfloat16) for key, tensor in emitted.state.items()}
        assert same(copy_state(model), rounded)
    assert provider.fingerprint(model) != emitted.digest()
    provider.rollback(model)
    assert same(copy_state(model), before)
    provider.adapt(training, model, 100.0)
    kept = copy_state(model)  # round 1 kept
    provider.adapt(training.enlarged([pseudo_pair(demos)]), model, 100.0)
    assert not same(copy_state(model), kept)
    provider.rollback(model)
    assert same(copy_state(model), kept) and not model.training


def test_zero_initialized_h_gives_the_base_logits_under_the_bf16_cast(ttt, tmp_path):
    """B = 0 stays 0 in bf16, so H at initialization swapped into the cast live model gives the
    cast base's logits and rescoring NLLs bit for bit; its digest is not the export's (ADR-017)."""
    decode = importlib.import_module("arc26.decode")
    base = cast_as_solver_load(tr.tiny_base(0))
    model = cast_as_solver_load(live_model(perturb=3))
    training = demo_set().without_held_out()
    tokenizer, ids, queries, answers = decode_inputs(training)
    assert not torch.equal(logits(model, ids), logits(base, ids))  # not vacuous
    provider = make_hyper(model, tmp_path / "cache", h=hypernet(model, trained=False))
    provider.adapt(training, model, 100.0)
    assert logits(model, ids).dtype == torch.bfloat16
    assert torch.equal(logits(model, ids), logits(base, ids))
    found = decode.calc_scores(queries, answers, tokenizer, model)
    assert found == decode.calc_scores(queries, answers, tokenizer, base)
    assert provider.fingerprint(model) != provider.emit(training).digest()


# --- P3.3b: a set H --------------------------------------------------------------------------


def test_a_set_h_reads_the_cached_pairs_in_pair_hash_order(tmp_path):
    """With ``aggregator: set`` the provider passes the cached embeddings themselves, in pair-hash
    order, beside their mean; the adapter still ignores the pairs' order, differs from the mean
    H's of the same shared weights, and ``from_checkpoint`` builds the set H from the config."""
    model = live_model()
    agg = {"kind": "set", "width": 8, "blocks": 1, "heads": 2}
    shapes = BaseShapes.from_config(model.config)
    h = Hypernet(h_config(aggregator=agg), shapes)
    generator = torch.Generator().manual_seed(6)
    with torch.no_grad():
        for param in h.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.1)
    demos = demo_set()
    enlarged = demos.without_held_out().enlarged([pseudo_pair(demos)])
    provider = make_hyper(model, tmp_path / "a", h=h)
    first = provider.emit(enlarged).digest()
    rows = provider.cache.get(sorted(pair_hash(pair.task_pair()) for pair in enlarged.pairs))
    with torch.no_grad():
        by_hand = to_peft_state_dict(h.eval()(rows.mean(dim=0)[None], rows[None])[0])
    assert adapter_digest(by_hand) == first
    shuffled = dataclasses.replace(enlarged, pairs=enlarged.pairs[::-1])
    assert make_hyper(model, tmp_path / "b", h=h).emit(shuffled).digest() == first
    mean = Hypernet(h_config(), shapes)
    shared = {k: v for k, v in h.state_dict().items() if not k.startswith("aggregator.")}
    mean.load_state_dict(shared)
    assert make_hyper(model, tmp_path / "c", h=mean).emit(enlarged).digest() != first

    raw = yaml.safe_load(TOY.read_text(encoding="utf-8"))
    raw.pop("smoke")
    settings = tr.EncoderSettings(
        layer=LAYER, pool="output", n_perm=1, view_seed=2, batch_tokens=4096, chunk_pairs=8
    )
    cfg = dataclasses.replace(
        tr.TrainConfig.from_dict(raw),
        hypernet=h_config(aggregator=agg),
        encoder=settings,
        model_id=MODEL_ID,
    )
    assert cfg.to_dict()["hypernet"]["aggregator"]["kind"] == "set"
    payload = {"schema": tr.CHECKPOINT_SCHEMA, "step": 3, "config": cfg.to_dict()}
    tr.save_checkpoint(tmp_path / "ckpt", {**payload, "hypernet": h.state_dict()}, keep=2)
    loaded = HyperProvider.from_checkpoint(
        tmp_path / "ckpt", model=model, tokenizer=tr.standin_tokenizer(), cache_root=tmp_path / "d"
    )
    assert loaded.hypernet.aggregator is not None
    assert loaded.emit(enlarged).digest() == first


def test_a_mean_checkpoint_has_no_aggregator_block():
    """Checkpoints of the mean H (MT-02's, MT-03's) carry no aggregator block, and build H."""
    raw = yaml.safe_load(TOY.read_text(encoding="utf-8"))
    raw.pop("smoke")
    plain = dataclasses.replace(tr.TrainConfig.from_dict(raw), hypernet=h_config()).to_dict()
    assert "aggregator" not in plain["hypernet"]
    assert HypernetConfig(**plain["hypernet"]) == h_config()
