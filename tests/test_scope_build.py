"""The build's LoRA scope in the notebook builder, and run P2.6-01's build (task P2.6, ADR-018).

A submission config may set ``scope``, one LoRA scope (``lora_r``, ``lora_targets``) for the single
run of a build without a series. Unlike an arm's, it applies in both modes: the Config cell holds
it as ``SCOPE`` and the Run cell passes it to ``arc26.runner.RunConfig``, so a competition rerun
trains what a development run trains. Against the task brief's acceptance, this file checks:

- builds without a scope stay byte for byte as before: every committed build config reproduces
  the digests recorded at ``2748a6a`` (``tests/testdata/p2_6_golden_builds.json``), and a scoped
  build differs from its unscoped twin only in its first two cells and by ``SCOPE_EDITS``;
- the scope reaches the solver in both modes: on the simulated Kaggle tree of
  ``tests/test_notebook_local.py`` the generated notebook starts ``python -m arc26.solver`` with
  the scope in its ``--config`` JSON in a competition rerun exactly as in a development run, and
  the split solver's ``load`` builds its LoRA settings from that JSON;
- the refusals: a malformed scope (an arm's rules and messages), the reference's own scope, a
  scope beside arms, a scope in a script config;
- run P2.6-01: its config (P1.1-01's with the scope, on its own code dataset), its build (rebuilt
  byte for byte, the gateway's checks, fit to submit, the scope stated), and its brief's header.

Nothing reaches Kaggle.
"""

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

import arc26.submit as submit
from arc26.runner import RunConfig, solver_command

REPO = Path(__file__).resolve().parent.parent
P26_CONFIG = "configs/submission_P2.6-01.yaml"
P11_CONFIG = "configs/submission_P1.1-01.yaml"
P13_CONFIG = "configs/submission_P1.3-01.yaml"
P26_BRIEF = REPO / "docs" / "prerun" / "P2.6-01.md"
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
SCOPE = {"lora_r": 16, "lora_targets": ATTENTION}
GOLDEN = json.loads(
    (REPO / "tests" / "testdata" / "p2_6_golden_builds.json").read_text(encoding="utf-8")
)
# Every committed build config before task P2.6: their builds must not change.
GOLDEN_CONFIGS = (
    "configs/script_P3.1-01.yaml",
    "configs/script_P3.2-01.yaml",
    "configs/script_P3.2-02.yaml",
    "configs/script_P3.5-01.yaml",
    "configs/script_P3.5-02.yaml",
    "configs/script_smoke.yaml",
    "configs/submission_P0.3-01.yaml",
    "configs/submission_P0.5-01.yaml",
    "configs/submission_P1.1-01.yaml",
    "configs/submission_P1.3-01.yaml",
    "configs/submission_P2.0-01.yaml",
    "configs/submission_P2.1-01.yaml",
    "configs/submission_P2.3-01.yaml",
    "configs/submission_P2.5-01.yaml",
    "configs/submission_P2.5-02.yaml",
    "configs/submission_cpu_check.yaml",
)
# Fixed provenance for the golden builds: no wheel is built and the validator is a stand-in, so
# the bytes depend only on the builders, the committed configs, and the files they embed
# (configs/governor.yaml for a round arm, reference/kernel-metadata.json).
GOLDEN_COMMIT = "0" * 40
GOLDEN_WHEEL = "arc26-0.0.1-py3-none-any.whl"
GOLDEN_WHEEL_SHA256 = "1" * 64
SUBMIT_STAND_IN = "# arc26/submit.py: a stand-in for the golden builds\n"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def sha256(data):
    return hashlib.sha256(data).hexdigest()


BUILD_TESTS = load("_scope_build_submission", REPO / "tests" / "test_build_submission.py")
LOCAL = load("_scope_notebook_local", REPO / "tests" / "test_notebook_local.py")
LORA_SCOPE = load("_scope_lora_scope", REPO / "tests" / "test_lora_scope.py")
nbb, bs, ops = BUILD_TESTS.nbb, BUILD_TESTS.bs, BUILD_TESTS.ops
NINE = list(LOCAL.NINE)
DROP = BUILD_TESTS.DROP
BUILD_STUB = {
    "tag": "t", "commit": "0" * 40, "config": P26_CONFIG, "config_sha256": "2" * 64,
    "wheel": GOLDEN_WHEEL, "wheel_sha256": GOLDEN_WHEEL_SHA256,
}  # fmt: skip


def parsed(name):
    path = REPO / name
    return nbb.parse_config(path.read_bytes(), path)


def config_text(changes, name=P26_CONFIG):
    """A committed config with changes (DROP removes a key), as text for parse_config."""
    raw = yaml.safe_load((REPO / name).read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is DROP:
            raw.pop(key)
        else:
            raw[key] = value
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


def golden_files(builder, name):
    """What ``builder`` (a ``scripts/build_submission.py`` module) writes for a committed config,
    from the fixed provenance above: the notebook, both metadata files, and ``MANIFEST.md``."""
    notebooks = builder.notebook_builder
    data = (REPO / name).read_bytes()
    config = notebooks.resolve_rounds(
        notebooks.parse_config(data, Path(name)), lambda path: (REPO / path).read_bytes()
    )
    script = config.kind == "script"
    hashes = {}
    if script:
        hashes = {
            "script_sha256": sha256((REPO / config.script).read_bytes()),
            "script_config_sha256": sha256((REPO / config.script_config).read_bytes()),
        }
    info = notebooks.build_info(
        config,
        tag="golden",
        commit=GOLDEN_COMMIT,
        config_file=name,
        config_sha256=sha256(data),
        wheel=GOLDEN_WHEEL,
        wheel_sha256=GOLDEN_WHEEL_SHA256,
        **hashes,
    )
    if script:
        notebook = notebooks.build_script_notebook(config, info)
    else:
        notebook = notebooks.build_notebook(config, info, SUBMIT_STAND_IN)
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = builder.kernel_metadata(config, pulled)
    dataset = builder.dataset_metadata(config.code_dataset, config.script if script else None)
    files = {
        notebooks.notebook_file(config): notebooks.notebook_bytes(notebook),
        builder.METADATA_FILE: builder._json_bytes(meta),
        f"{builder.CODE_DIR}/{builder.DATASET_METADATA_FILE}": builder._json_bytes(dataset),
    }
    files[builder.MANIFEST_FILE] = builder.render_manifest(
        tag="golden",
        build=info,
        commit_time=0,
        wheel_commit=GOLDEN_COMMIT,
        wheel_epoch=0,
        config=config,
        builder="uv build --wheel --offline",
        generator="hatchling 1.27.0",
        metadata=meta,
        files=dict(files),
    )
    return files


def golden_digests(builder, name):
    return {file: sha256(data) for file, data in golden_files(builder, name).items()}


def golden_inputs(builder, name):
    """Every repository file a golden build reads besides the builders' own modules, with its
    sha256: the config, the reference metadata, a script build's script and script config (both
    copied beside the wheel), and a round arm's governor config (embedded)."""
    notebooks = builder.notebook_builder
    config = notebooks.parse_config((REPO / name).read_bytes(), Path(name))
    files = [name, config.reference_metadata]
    if config.kind == "script":
        files += [config.script, config.script_config]
    else:
        files += sorted({arm.rounds.governor for arm in config.arms or () if arm.rounds})
    return {file: sha256((REPO / file).read_bytes()) for file in files}


def cell_sources(notebook):
    """Each cell's text by its name, the cell id without ``arc26-NN-``."""
    return {cell["id"].split("-", 2)[2]: "".join(cell["source"]) for cell in notebook["cells"]}


# Builds without a scope: byte for byte as before task P2.6.


def test_the_golden_digests_cover_every_config_recorded():
    assert set(GOLDEN["configs"]) == set(GOLDEN["inputs"]) == set(GOLDEN_CONFIGS)
    assert all(
        parsed(name).kind == "script" or parsed(name).scope is None for name in GOLDEN_CONFIGS
    )


@pytest.mark.parametrize("name", GOLDEN_CONFIGS)
def test_every_committed_build_writes_the_bytes_recorded_before_the_scope(name):
    """The notebook, kernel-metadata.json, dataset-metadata.json, and MANIFEST.md of every
    committed build config equal, byte for byte, what the builders wrote before task P2.6, as
    long as the build's inputs (golden_inputs) hash as recorded then. A config whose input another
    task changed, as task MT changes scripts/cache_embeddings.py, is skipped with the file named:
    its goldens no longer apply. With its inputs unchanged a build must not change, so the test
    keeps guarding the builders."""
    recorded = GOLDEN["inputs"][name]
    changed = [
        file
        for file, digest in recorded.items()
        if not (REPO / file).is_file() or sha256((REPO / file).read_bytes()) != digest
    ]
    if changed:
        pytest.skip(f"{name}: changed since the goldens were recorded: {', '.join(changed)}")
    assert golden_inputs(bs, name) == recorded
    assert golden_digests(bs, name) == GOLDEN["configs"][name]


def test_a_scoped_build_differs_from_its_unscoped_twin_only_in_the_scope_lines():
    """The first two cells state and hold the scope; four fixed cells take SCOPE_EDITS, each old
    text once; every other cell is the same; the unscoped twin never names SCOPE."""
    config = parsed(P26_CONFIG)
    scoped = cell_sources(nbb.build_notebook(config, BUILD_STUB, SUBMIT_STAND_IN))
    plain = cell_sources(
        nbb.build_notebook(config._replace(scope=None), BUILD_STUB, SUBMIT_STAND_IN)
    )
    assert list(scoped) == list(plain)
    assert set(nbb.SCOPE_EDITS) == {"helpers", "environment", "run-note", "run"}
    changed = {name for name in plain if scoped[name] != plain[name]}
    assert changed == {"config-note", "config", *nbb.SCOPE_EDITS}
    for name, edits in nbb.SCOPE_EDITS.items():
        text = plain[name]
        for old, new in edits:
            assert text.count(old) == 1, (name, old)
            text = text.replace(old, new)
        assert text == scoped[name], name
    assert "SCOPE" not in "".join(plain.values())
    # The environment check's series line in a rerun, and the scope line, as SCOPE_EDITS give them.
    for line in (
        'print("series: ignored in a competition rerun, which runs SCOPE once")',
        'print(f"LoRA scope, both modes: {SCOPE}")',
    ):
        assert scoped["environment"].count(line) == 1, line
    stale = "series: ignored in a competition rerun, which runs the reference settings once"
    assert plain["environment"].count(f'print("{stale}")') == 1
    assert stale not in scoped["environment"]
    # The Config cell: SCOPE with its comment, and ARMS' comment naming it.
    lines, plain_lines = scoped["config"].splitlines(), plain["config"].splitlines()
    literal = "SCOPE = {'lora_r': 16, 'lora_targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj']}"
    assert set(lines) - set(plain_lines) == {
        *nbb.SCOPE_COMMENT, literal, *nbb.SERIES_SCOPE_COMMENT[1:]
    }  # fmt: skip
    assert set(plain_lines) - set(lines) == set(nbb.SERIES_FIRST_COMMENT[1:])
    assert lines.index(literal) == lines.index("SOLVER = 'split'") + len(nbb.SCOPE_COMMENT) + 1
    # The first cell: the scope paragraph, and the Config paragraph's rerun naming SCOPE.
    paragraph = (
        "Scoped build (task P2.6): `SCOPE` sets the LoRA scope of every solver launch, r = 16 on "
        "q_proj, k_proj, v_proj, o_proj (embed_tokens and lm_head frozen), instead of the "
        "reference's r = 256 on all nine target modules, in both modes: a competition rerun "
        "trains what a development run trains "
        "(`docs/decisions/ADR-018-submission-2-without-the-loop.md`).\n\n"
    )
    assert scoped["config-note"] == (
        plain["config-note"]
        .replace("## 1. Config\n", paragraph + "## 1. Config\n")
        .replace("with the reference's LoRA settings and", "with `SCOPE` and")
        .replace("`SOLVER` applies in both modes.", "`SOLVER` and `SCOPE` apply in both modes.")
    )
    assert (REPO / "docs" / "decisions" / nbb.SCOPE_ADR).is_file()


def test_a_scope_edit_that_no_longer_matches_its_cell_stops_the_build(monkeypatch):
    edits = {**nbb.SCOPE_EDITS, "run": (("    scope = {}\n", "x"), ("    scope = {}\n", "y"))}
    monkeypatch.setattr(nbb, "SCOPE_EDITS", edits)
    with pytest.raises(RuntimeError, match="the run cell changed: update SCOPE_EDITS"):
        nbb.build_notebook(parsed(P26_CONFIG), BUILD_STUB, SUBMIT_STAND_IN)


# The config rules: an arm's for the scope itself, and three of its own.


@pytest.mark.parametrize(
    ("rank", "targets"),
    [
        (0, ATTENTION), (True, ATTENTION), (16.0, ATTENTION), ("16", ATTENTION), (16, []),
        (16, "q_proj"), (16, ["qkv_proj"]), (16, ["q_proj", "q_proj"]),
        (16, ["k_proj", "q_proj"]), (16, ["embed_tokens", "lm_head"]), (16, ["lm_head"]),
    ],
)  # fmt: skip
def test_a_scope_is_refused_by_an_arms_rules_with_an_arms_message(rank, targets):
    with pytest.raises(nbb.BuildError) as scope_refused:
        nbb.parse_config(
            config_text({"scope": {"lora_r": rank, "lora_targets": targets}}), Path("c.yaml")
        )
    arms = [BUILD_TESTS.REFERENCE_ARM, BUILD_TESTS.arm(lora_r=rank, lora_targets=targets)]
    with pytest.raises(nbb.BuildError) as arm_refused:
        nbb.parse_config(config_text({"arms": arms, "scope": DROP}), Path("c.yaml"))
    scope_message = str(scope_refused.value)
    assert scope_message.startswith("c.yaml: scope: ")
    assert scope_message.removeprefix("c.yaml: scope: ") == str(arm_refused.value).removeprefix(
        "c.yaml: arm 2: "
    )


@pytest.mark.parametrize(
    ("scope", "message"),
    [
        ([], "c.yaml: scope must be null or have exactly the keys lora_r, lora_targets"),
        ("r16-attn", "scope must be null or have exactly the keys lora_r, lora_targets"),
        ({"lora_r": 16}, "scope must be null or have exactly the keys lora_r, lora_targets"),
        ({**SCOPE, "reason": "x"}, "scope must be null or have exactly the keys lora_r, lora_"),
        ({"rank": 16, "target_modules": ATTENTION}, "scope must be null or have exactly the keys"),
        (
            {"lora_r": 256, "lora_targets": NINE},
            "c.yaml: scope is the reference's, lora_r 256 on all nine targets; leave it out, "
            "which builds exactly that",
        ),
    ],
)
def test_a_malformed_scope_or_the_references_own_is_refused(scope, message):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(config_text({"scope": scope}), Path("c.yaml"))
    assert message in str(refused.value)


@pytest.mark.parametrize(
    "arms",
    [
        [BUILD_TESTS.REFERENCE_ARM],
        [BUILD_TESTS.REFERENCE_ARM, BUILD_TESTS.arm()],
        [{**BUILD_TESTS.arm(), "holdout": {"rule": "last", "k2": True, "in_test_prompts": False}}],
    ],
    ids=["reference-arm", "two-arms", "hold-out-arm"],
)
def test_a_scope_beside_arms_is_refused(arms):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(config_text({"arms": arms}), Path("c.yaml"))
    assert str(refused.value) == (
        "c.yaml: scope and arms exclude each other: a development run trains each arm's scope, "
        "while a competition rerun ignores the arms and would train the build's scope, which no "
        "development run of the build trained"
    )


def test_a_script_config_cannot_set_a_scope():
    with pytest.raises(nbb.BuildError, match=re.escape("unknown keys ['scope']")):
        nbb.parse_config(BUILD_TESTS.script_text({"scope": SCOPE}), Path("c.yaml"))


@pytest.mark.parametrize(
    "scope",
    [
        {"lora_r": 1, "lora_targets": ["q_proj"]},
        {"lora_r": 256, "lora_targets": NINE[:7]},
        {"lora_r": 16, "lora_targets": NINE},
        {"lora_r": 64, "lora_targets": ["down_proj", "lm_head"]},
        {"lora_r": 512, "lora_targets": ["o_proj"]},
    ],
)
def test_any_scope_an_arm_may_have_is_accepted_and_runconfig_takes_it(scope):
    config = nbb.parse_config(config_text({"scope": scope}), Path("c.yaml"))
    assert config.scope == nbb.LoraScope(scope["lora_r"], tuple(scope["lora_targets"]))
    assert config.scope.settings() == scope
    cfg = RunConfig(data_dir="/d", model_dir="/m", **config.scope.settings())
    passed = json.loads(solver_command(cfg, 0.0, "python")[6])
    assert {key: passed[key] for key in SCOPE} == scope


def test_a_null_scope_is_no_scope():
    null = nbb.parse_config(config_text({"scope": None}), Path("c.yaml"))
    assert null.scope is None
    assert null == nbb.parse_config(config_text({"scope": DROP}), Path("c.yaml"))


@pytest.mark.parametrize(
    "name",
    [BUILD_TESTS.GPU_CONFIG, BUILD_TESTS.CPU_CONFIG, BUILD_TESTS.TIMING_CONFIG, P11_CONFIG],
)
def test_a_scope_is_no_reason_for_or_against_submitting(name):
    """submission_blocker reads the production settings, not the scope: a scoped build of a
    production config is fit to submit, and a development build keeps exactly its reasons."""
    plain = parsed(name)
    scoped = nbb.parse_config(config_text({"scope": SCOPE}, name), Path("c.yaml"))
    assert scoped.scope == nbb.LoraScope(16, tuple(ATTENTION))
    assert nbb.submission_blocker(scoped) == nbb.submission_blocker(plain)
    assert nbb.submission_blocker(parsed(P26_CONFIG)) is None


def test_scope_text_names_the_frozen_modules():
    assert nbb.scope_text(nbb.LoraScope(16, tuple(ATTENTION))) == (
        "r = 16 on q_proj, k_proj, v_proj, o_proj (embed_tokens and lm_head frozen)"
    )
    assert nbb.scope_text(nbb.LoraScope(64, ("down_proj", "lm_head"))) == (
        "r = 64 on down_proj, lm_head (embed_tokens frozen)"
    )
    assert nbb.scope_text(nbb.LoraScope(16, tuple(NINE))) == f"r = 16 on {', '.join(NINE)}"


# The scope reaches the solver: the generated notebook on the simulated Kaggle tree.


def p26_notebook(dest, wheel, name, **changes):
    path = REPO / P26_CONFIG
    config = nbb.parse_config(path.read_bytes(), path)._replace(**changes)
    info = nbb.build_info(
        config,
        tag=f"local-{name}",
        commit=LOCAL.COMMIT,
        config_file=P26_CONFIG,
        config_sha256=LOCAL.sha256(path),
        wheel=wheel.name,
        wheel_sha256=LOCAL.sha256(wheel),
    )
    out = dest / f"{name}.ipynb"
    out.write_bytes(nbb.notebook_bytes(nbb.build_notebook(config, info, nbb.wheel_member(wheel))))
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    dest = tmp_path_factory.mktemp("build")
    wheel = LOCAL.make_wheel(dest)
    return LOCAL.Built(
        wheel,
        {
            "P2.6-01": p26_notebook(dest, wheel, "P2.6-01"),
            "P2.6-01-unscoped": p26_notebook(dest, wheel, "P2.6-01-unscoped", scope=None),
        },
    )


# Stand-in hidden test tasks: ids no evaluation task has, so the submission shows which
# challenges file a run read (tests/test_notebook_local.py's tree writes the DEV4 stand-ins to
# both files, which its competition tests rely on).
HIDDEN = {"f0000001": 1, "f0000002": 2, "f0000003": 1}


def hidden_tree(tmp, wheel):
    """The simulated Kaggle tree of tests/test_notebook_local.py, its test challenges the hidden
    stand-ins."""
    root = LOCAL.make_tree(tmp, wheel)
    data_dir = root / "input" / "competitions" / LOCAL.COMPETITION
    LOCAL.write_json(data_dir / "arc-agi_test_challenges.json", LOCAL.standin_challenges(HIDDEN))
    return root


def hand_edited(notebook, dest, old, new):
    """A copy of a built notebook whose Config cell a human edited, ``old`` replaced by ``new``."""
    data = json.loads(notebook.read_text(encoding="utf-8"))
    (cell,) = [c for c in data["cells"] if c["id"] == "arc26-02-config"]
    text = "".join(cell["source"])
    assert text.count(old) == 1
    cell["source"] = text.replace(old, new).splitlines(keepends=True)
    out = dest / f"edited-{notebook.name}"
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def runs(built, tmp_path_factory):
    """P2.6-01's notebook as built, in a development run and in a competition rerun on 4 L4s, on
    the tree with the hidden test stand-ins: the solver starts and fails at its import of the GPU
    stack, as in every local run."""
    found = {}
    for mode in ("development", "competition"):
        tmp = tmp_path_factory.mktemp(mode)
        root = hidden_tree(tmp, built.wheel)
        found[mode] = LOCAL.simulate(
            tmp, built.notebooks["P2.6-01"], root, competition=mode == "competition", gpus=4
        )
    return found


def solver_call(run):
    """The one ``python -m arc26.solver`` call: its soft deadline after the start, its JSON."""
    (start,) = LOCAL.split_starts(run)
    assert start["argv"][2] == "--end-time" and start["argv"][4] == "--config"
    assert start["env"] == LOCAL.LAUNCH_ENV
    return float(start["argv"][3]) - run.record["NOTEBOOK_START"], json.loads(start["argv"][5])


def test_a_competition_rerun_starts_the_split_solver_with_the_scope(built, runs):
    run = runs["competition"]
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and LOCAL.solver_starts(run) == []
    namespace = {}
    exec(LOCAL.notebook_code(built.notebooks["P2.6-01"])["arc26-02-config"], namespace)
    assert namespace["SCOPE"] == SCOPE and namespace["ARMS"] is None
    end_s, config = solver_call(run)
    assert end_s == nbb.COMPETITION_SOLVER_END_S
    assert config == {
        "data_dir": str(run.root / "input" / "competitions" / LOCAL.COMPETITION),
        "model_dir": str(run.root.joinpath("input", *LOCAL.MODEL_DIR)),
        "outputs_dir": str(run.root / "inference_outputs"),
        "marker_dir": str(run.root),
        "subset": None,
        "workers": 4,
        "timing_dir": None,
        "lora_r": 16,
        "lora_targets": ATTENTION,
    }
    assert f"LoRA scope, both modes: {SCOPE!r}" in run.output
    assert LOCAL.GPU_STACK_MISSING.search(run.output)
    assert sorted(p.name for p in (run.root / "working").iterdir()) == ["submission.json"]
    # The rerun submits the hidden set, exactly, and reads no evaluation file.
    written = LOCAL.submission(run)
    assert sorted(written) == sorted(HIDDEN)
    assert submit.validate(written, LOCAL.standin_challenges(HIDDEN)) == []
    assert run.record["CHALLENGES"].endswith("arc-agi_test_challenges.json")
    opened = [path for _, path in run.record["opened"]]
    assert any(path.endswith("arc-agi_test_challenges.json") for path in opened)
    assert not [path for path in opened if "solutions" in path or "evaluation" in path]
    LOCAL.assert_kernel_never_imported_torch(run)


def test_a_development_run_starts_the_solver_with_the_same_scope_and_records_it(runs):
    """The solver's JSON of the development run equals the competition rerun's but for the
    timing folder (a rerun records no timing); both start at the same soft deadline."""
    dev, rerun = runs["development"], runs["competition"]
    assert dev.record["error"] is None, dev.output
    assert dev.record["PROBLEMS"] == []
    (dev_end, dev_config), (rerun_end, rerun_config) = solver_call(dev), solver_call(rerun)
    assert dev_end == rerun_end == 41400

    def rooted(config, root):
        return {k: v.replace(str(root), "<root>") if isinstance(v, str) else v
                for k, v in config.items()}  # fmt: skip

    assert dev_config["timing_dir"] == str(dev.root / "working" / "run_logs" / "timing")
    assert rooted(dev_config, dev.root) | {"timing_dir": None} == rooted(rerun_config, rerun.root)
    assert f"LoRA scope, both modes: {SCOPE!r}" in dev.output
    stages = json.loads((dev.root / "working" / "run_logs" / "stages.json").read_text())
    assert stages["stages"]["run"]["scope"] == SCOPE
    assert stages["settings"]["solver"] == "split" and stages["settings"]["arms"] is None
    assert stages["build"]["code_dataset"] == "alyahya737/arc26-code-p26"
    # The development run submits the evaluation stand-ins, not the hidden ones.
    assert sorted(LOCAL.submission(dev)) == sorted(LOCAL.DEV4)
    assert submit.validate(LOCAL.submission(dev), LOCAL.standin_challenges()) == []


def test_a_rerun_of_a_scoped_build_with_arms_edited_in_runs_scope_and_says_so(built, tmp_path):
    """A human who sets ARMS in a scoped build's Config cell: a competition rerun ignores the
    series, says it runs SCOPE once (the environment check's line from SCOPE_EDITS), and trains
    SCOPE once in the single run's folders on the hidden set."""
    arms = [{"name": "ref", "lora_r": 256, "lora_targets": NINE, "reason": "edited in"}]
    notebook = hand_edited(built.notebooks["P2.6-01"], tmp_path, "ARMS = None", f"ARMS = {arms!r}")
    root = hidden_tree(tmp_path, built.wheel)
    run = LOCAL.simulate(tmp_path, notebook, root, competition=True, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == []
    lines = run.output.splitlines()
    assert "series: ignored in a competition rerun, which runs SCOPE once" in lines
    assert not [line for line in lines if "reference settings" in line]
    _, config = solver_call(run)
    assert (config["lora_r"], config["lora_targets"]) == (16, ATTENTION)
    assert config["outputs_dir"] == str(root / "inference_outputs")
    assert "arms" not in run.record["STAGES"]["run"]
    assert sorted(LOCAL.submission(run)) == sorted(HIDDEN)


def split_solver_peft_params(cfg):
    """Evaluate the ``peft_params = dict(...)`` statement of ``arc26.solver.load`` with ``cfg``,
    as ``tests/test_lora_scope.py`` evaluates the moved reference's."""
    tree = ast.parse((REPO / "arc26" / "solver.py").read_text(encoding="utf-8"))
    (load_fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load"]
    (assign,) = [
        node
        for node in load_fn.body
        if isinstance(node, ast.Assign)
        and [getattr(target, "id", None) for target in node.targets] == ["peft_params"]
    ]
    return eval(compile(ast.Expression(assign.value), "<peft_params>", "eval"), {"cfg": cfg})


def test_the_split_solver_trains_the_scope_that_the_competition_rerun_passes(runs):
    """The JSON the rerun passed gives the split solver's PEFT settings rank 16 on q, k, v, o,
    everything else the reference's; RunConfig's defaults give the reference's own."""
    reference = LORA_SCOPE.peft_params(LORA_SCOPE.reference_cell())
    _, config = solver_call(runs["competition"])
    assert split_solver_peft_params(config) == {**reference, "r": 16, "target_modules": ATTENTION}
    default = json.loads(solver_command(RunConfig(data_dir="/d", model_dir="/m"), 0.0, "py")[6])
    assert split_solver_peft_params(default) == reference


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_the_run_cell_gives_runconfig_the_scope_in_both_modes(
    built, tmp_path, monkeypatch, competition
):
    namespace, cell, calls = LOCAL.run_cell_namespace(
        built, tmp_path, monkeypatch, lambda: 0, competition=competition, notebook="P2.6-01"
    )
    exec(compile(cell, "run", "exec"), namespace)
    ((fields, end_time, _, _),) = calls
    assert (fields["lora_r"], fields["lora_targets"]) == (16, ATTENTION)
    assert (fields["solver"], fields["workers"], fields["subset"]) == ("split", 4, None)
    timing = None if competition else str(tmp_path / "working" / "run_logs" / "timing")
    assert fields["timing_dir"] == timing
    assert end_time == namespace["NOTEBOOK_START"] + 41400
    assert namespace["STAGES"]["run"]["scope"] == SCOPE


@pytest.mark.parametrize("scope", [None, {}], ids=["none", "empty"])
def test_a_scope_edited_away_in_the_config_cell_runs_the_reference_settings(
    built, tmp_path, monkeypatch, scope
):
    namespace, cell, calls = LOCAL.run_cell_namespace(
        built, tmp_path, monkeypatch, lambda: 0, competition=True, notebook="P2.6-01"
    )
    namespace["SCOPE"] = scope
    exec(compile(cell, "run", "exec"), namespace)
    ((fields, _, _, _),) = calls
    assert not {"lora_r", "lora_targets"} & set(fields)
    assert namespace["STAGES"]["run"]["scope"] == scope


def test_the_unscoped_twin_reruns_the_reference_settings(built, tmp_path, monkeypatch):
    namespace, cell, calls = LOCAL.run_cell_namespace(
        built, tmp_path, monkeypatch, lambda: 0, competition=True, notebook="P2.6-01-unscoped"
    )
    assert "SCOPE" not in namespace and cell == nbb.RUN_CODE
    exec(compile(cell, "run", "exec"), namespace)
    ((fields, _, _, _),) = calls
    assert not {"lora_r", "lora_targets"} & set(fields)
    assert "scope" not in namespace["STAGES"]["run"]


# Run P2.6-01: Submission 2, P1.1-01's DEV120 run with the scope.


def test_p2_6_01_config_is_p1_1_01s_with_the_scope_on_its_own_code_dataset():
    p26, p11 = parsed(P26_CONFIG), parsed(P11_CONFIG)
    assert p26.scope == nbb.LoraScope(16, tuple(ATTENTION))
    assert p26.code_dataset == "alyahya737/arc26-code-p26" != p11.code_dataset
    assert (p26.solver, p11.solver) == ("split", "nvarc")
    assert p26._replace(code_dataset=p11.code_dataset, solver=p11.solver, scope=None) == p11
    assert (p26.kaggle_notebook, p26.accelerator) == (nbb.SUBMISSION_NOTEBOOK, "NvidiaL4")
    assert (p26.subset, p26.workers, p26.hash_seed, p26.run_solver) == (None, 4, 0, True)
    assert (p26.solver_end_s, p26.solver_kill_s) == (
        nbb.COMPETITION_SOLVER_END_S, nbb.COMPETITION_SOLVER_KILL_S
    )  # fmt: skip
    # P1.3-01 arm 2's scope (r16-attn), H's (ADR-002): 10.5 of 40 on DEV40.
    arm2 = next(a for a in parsed(P13_CONFIG).arms if a.name == "r16-attn")
    assert (arm2.lora_r, arm2.lora_targets) == tuple(p26.scope)
    assert nbb.submission_blocker(p26) is None


def test_every_p2_6_01_config_value_has_a_comment():
    lines = (REPO / P26_CONFIG).read_text(encoding="utf-8").splitlines()
    keys = [i for i, line in enumerate(lines) if re.match(r"\s*[a-z_]+:", line)]
    assert len(keys) == 17
    for index in keys:
        assert lines[index - 1].lstrip().startswith("#"), f"no comment above {lines[index]!r}"
    assert "TODO" not in "\n".join(lines)


def test_p2_6_01_metadata_attach_its_own_code_dataset_and_pass_the_gateway():
    config = parsed(P26_CONFIG)
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(config, pulled)  # runs the gateway's metadata checks
    assert meta["id"] == nbb.SUBMISSION_NOTEBOOK and meta["machine_shape"] == "NvidiaL4"
    assert meta["dataset_sources"] == ["alyahya737/arc26-code-p26"]
    assert meta["kernel_sources"] == ["sorokin/pip-install-unsloth-flash-patch/2"]
    dataset = bs.dataset_metadata(config.code_dataset)
    assert dataset["id"] == "alyahya737/arc26-code-p26" and dataset["title"] == "arc26 code p26"


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    """dist/P2.6-01/ twice and dist/P1.1-01/ once, from one throwaway repository holding the
    working tree, as tests/test_build_submission.py builds."""
    if not BUILD_TESTS.BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    repo = BUILD_TESTS.snapshot(tmp_path_factory.mktemp("repo"))
    first = BUILD_TESTS.read_files(bs.build("P2.6-01", repo / P26_CONFIG, repo))
    again = bs.build("P2.6-01", repo / P26_CONFIG, repo)
    p11 = bs.build("P1.1-01", repo / P11_CONFIG, repo)
    return repo, again, first, p11


def test_the_p2_6_01_build_is_a5s_folder_rebuilt_byte_for_byte(builds):
    _, folder, first, p11 = builds
    files = BUILD_TESTS.read_files(folder)
    assert set(files) == BUILD_TESTS.FILES and files == first
    # The scope is a run setting: the wheel is every build's of the commit.
    wheel = f"arc26-code/{BUILD_TESTS.WHEEL}"
    assert files[wheel] == BUILD_TESTS.read_files(p11)[wheel]


def test_the_p2_6_01_build_passes_the_gateway_and_is_fit_to_submit(builds):
    repo, folder, _, _ = builds
    meta = ops.load_kernel_metadata(folder.resolve())
    assert meta["id"] == nbb.SUBMISSION_NOTEBOOK and meta["machine_shape"] == "NvidiaL4"
    assert meta["dataset_sources"] == ["alyahya737/arc26-code-p26"]
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    ops.check_code_file(folder.resolve(), meta)
    ops.check_dataset_folder((folder / "arc26-code").resolve())
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    for row in (
        "| fit to submit | yes |",
        "| solver (both modes) | `split`: ",
        "| scope (both modes) | `SCOPE`, r = 16 on q_proj, k_proj, v_proj, o_proj (embed_tokens "
        "and lm_head frozen); task P2.6, ADR-018 |",
        "| subset (outside a competition rerun) | every task |",
        "| arms (series, development runs only) | none (one run, the scope above) |",
        "| private dataset `alyahya737/arc26-code-p26` | first build: `python "
        "scripts/kaggle_ops.py dataset-create dist/P2.6-01/arc26-code`",
    ):
        assert row in manifest, row
    cells = BUILD_TESTS.cells(folder)
    note = "".join(cells[0]["source"])
    assert "Not for submission" not in note and "Scoped build (task P2.6)" in note
    namespace = {}
    exec(BUILD_TESTS.code(folder)["arc26-02-config"], namespace)
    assert namespace["SCOPE"] == SCOPE and namespace["ARMS"] is None
    build = namespace["BUILD"]
    assert build["config"] == P26_CONFIG and build["code_dataset"] == "alyahya737/arc26-code-p26"
    assert build["commit"] == BUILD_TESTS.git(repo, "rev-parse", "HEAD").strip()
    assert "    scope = dict(SCOPE or {})\n" in BUILD_TESTS.code(folder)["arc26-12-run"]


# The pre-run brief.


def test_the_p2_6_01_brief_header_parses_with_the_gateways_reader():
    text = P26_BRIEF.read_text(encoding="utf-8")
    assert "TODO" not in text
    header = ops.brief_header(text)
    config = parsed(P26_CONFIG)
    assert header["Run-id"] == "P2.6-01" and header["Config"] == P26_CONFIG
    assert header["Kaggle notebook"] == config.kaggle_notebook == nbb.SUBMISSION_NOTEBOOK
    assert header["Accelerator"] == config.accelerator == "NvidiaL4"
    assert int(header["Timeout"]) == nbb.SESSION_LIMIT_S == 43200
    assert header["Previous run"].startswith("P1.1-01")
    assert config.kaggle_notebook in ops.briefed_notebooks()
