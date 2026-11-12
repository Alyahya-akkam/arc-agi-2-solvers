"""JSONL timing logger of arc26.timing (P0.3a, BRIEF B4 conventions)."""

import json
import math
import multiprocessing
import os
import threading
from types import SimpleNamespace

import pytest

from arc26 import timing
from arc26.timing import DECODE, PUZZLE, TRAIN, TimingLogger, TimingRecord, read_records


@pytest.fixture(autouse=True)
def no_process_logger(monkeypatch):
    monkeypatch.setattr(timing, "_process_logger", None)


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_record_appends_one_line_with_the_schema_fields_in_order(tmp_path):
    logger = TimingLogger(tmp_path / "logs", worker=2)
    logger.record("0934a4d8", TRAIN, 12.3456789, 4096)
    logger.record(None, "model_load", 150, None)
    assert logger.path == tmp_path / "logs" / "timing-2.jsonl"
    first, second = (json.loads(line) for line in lines(logger.path))
    assert list(first) == ["puzzle_id", "op", "seconds", "tokens"]
    assert first == {"puzzle_id": "0934a4d8", "op": "train", "seconds": 12.345679, "tokens": 4096}
    assert second == {"puzzle_id": None, "op": "model_load", "seconds": 150.0, "tokens": None}
    assert '"tokens": null' in lines(logger.path)[1]


def test_default_worker_label_is_the_process_id(tmp_path):
    logger = TimingLogger(tmp_path)
    assert logger.worker == f"pid{os.getpid()}"
    assert logger.path.name == f"timing-pid{os.getpid()}.jsonl"


@pytest.mark.parametrize("worker", ["../up", "a/b", ".", "..", "", "rank 0"])
def test_worker_labels_must_be_plain_file_name_parts(tmp_path, worker):
    with pytest.raises(ValueError, match="worker label"):
        TimingLogger(tmp_path, worker=worker)


def test_nothing_is_written_before_the_first_record(tmp_path):
    TimingLogger(tmp_path / "logs", worker=0)
    assert not (tmp_path / "logs").exists()


def test_timed_block_records_its_duration_and_late_token_count(tmp_path):
    logger = TimingLogger(tmp_path, worker=0)
    with logger.timed("t1", DECODE) as timer:
        assert timer.seconds is None
        timer.tokens = 931
    assert timer.seconds is not None and timer.seconds >= 0
    (record,) = (json.loads(line) for line in lines(logger.path))
    assert record["op"] == "decode" and record["tokens"] == 931
    assert record["seconds"] == round(timer.seconds, 6)


def test_timed_block_measures_wall_time(tmp_path, monkeypatch):
    clock = iter([100.0, 102.5])
    monkeypatch.setattr(timing, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    logger = TimingLogger(tmp_path, worker=0)
    with logger.timed("t1", TRAIN, tokens=10):
        pass
    assert json.loads(lines(logger.path)[0])["seconds"] == 2.5


def test_a_block_that_raises_writes_nothing_and_propagates(tmp_path):
    logger = TimingLogger(tmp_path, worker=0)
    with pytest.raises(RuntimeError, match="out of memory"), logger.timed("t1", TRAIN):
        raise RuntimeError("out of memory")
    assert not logger.path.exists()


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((1, "train", 1.0, None), TypeError),
        (("t", 3, 1.0, None), TypeError),
        (("t", "", 1.0, None), ValueError),
        (("t", "train", True, None), TypeError),
        (("t", "train", "1.0", None), TypeError),
        (("t", "train", -0.1, None), ValueError),
        (("t", "train", math.nan, None), ValueError),
        (("t", "train", math.inf, None), ValueError),
        (("t", "train", 1.0, True), TypeError),
        (("t", "train", 1.0, 1.5), TypeError),
        (("t", "train", 1.0, "12"), TypeError),
        (("t", "train", 1.0, -1), ValueError),
    ],
)
def test_invalid_records_raise_and_write_nothing(tmp_path, args, error):
    logger = TimingLogger(tmp_path, worker=0)
    with pytest.raises(error):
        logger.record(*args)
    assert not logger.path.exists()


class IntegerScalar:
    """Stands in for numpy.int64 or a torch integer scalar: not an int, but has __index__."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def test_integer_scalar_token_counts_are_stored_as_plain_ints(tmp_path):
    logger = TimingLogger(tmp_path, worker=0)
    entry = logger.record("t1", TRAIN, 1.0, IntegerScalar(4096))
    assert entry["tokens"] == 4096 and type(entry["tokens"]) is int
    with logger.timed("t1", DECODE) as timer:
        timer.tokens = IntegerScalar(931)  # set inside the block, converted on exit
    records, skipped = read_records([logger.path])
    assert skipped == 0 and [record.tokens for record in records] == [4096, 931]
    with pytest.raises(ValueError, match="non-negative"):
        logger.record("t1", TRAIN, 1.0, IntegerScalar(-1))


def test_module_timed_is_a_stopwatch_until_configured(tmp_path):
    assert timing.get_logger() is None
    with timing.timed("t1", TRAIN, tokens=5) as timer:
        pass
    assert timer.seconds is not None and timer.tokens == 5
    assert list(tmp_path.iterdir()) == []


def test_module_timed_writes_to_the_configured_logger(tmp_path):
    logger = timing.configure(tmp_path, worker="rank1")
    assert timing.get_logger() is logger
    with timing.timed("t1", PUZZLE) as timer:
        timer.tokens = 7
    (record,) = (json.loads(line) for line in lines(tmp_path / "timing-rank1.jsonl"))
    assert record["op"] == "puzzle" and record["tokens"] == 7


def test_threads_sharing_a_logger_write_whole_lines(tmp_path):
    logger = TimingLogger(tmp_path, worker=0)

    def work(n):
        for i in range(200):
            logger.record(f"p{n}", DECODE, i * 0.001, i)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records, skipped = read_records([logger.path])
    assert skipped == 0 and len(records) == 1600


def _worker_process(log_dir, rank, count):
    timing.configure(log_dir, worker=rank)
    for i in range(count):
        with timing.timed(f"puzzle{rank}", TRAIN, tokens=i):
            pass


def test_four_worker_processes_write_four_separate_files(tmp_path):
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("fork start method unavailable")
    processes = [context.Process(target=_worker_process, args=(tmp_path, r, 50)) for r in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    files = sorted(tmp_path.iterdir())
    assert [f.name for f in files] == [f"timing-{r}.jsonl" for r in range(4)]
    for rank, path in enumerate(files):
        records, skipped = read_records([path])
        assert skipped == 0 and len(records) == 50
        assert {(r.puzzle_id, r.worker) for r in records} == {(f"puzzle{rank}", str(rank))}
        assert [r.tokens for r in records] == list(range(50))


def test_read_records_skips_invalid_lines_and_keeps_the_rest(tmp_path):
    path = tmp_path / "timing-3.jsonl"
    good = {"puzzle_id": "t1", "op": "train", "seconds": 1.5, "tokens": 10}
    path.write_text(
        "\n".join(
            [
                json.dumps(good),
                "",
                json.dumps({**good, "extra": "kept"}),
                json.dumps({"puzzle_id": "t1", "op": "train", "seconds": 1.5}),  # no tokens key
                json.dumps({**good, "seconds": -1}),
                json.dumps([1, 2, 3]),
                "not json",
                '{"puzzle_id": "t2", "op": "dec',  # truncated by a hard kill
            ]
        ),
        encoding="utf-8",
    )
    records, skipped = read_records([path])
    assert skipped == 5
    assert records == [TimingRecord("t1", "train", 1.5, 10, "3")] * 2


def test_worker_label_comes_from_the_file_name(tmp_path):
    other = tmp_path / "old-log.jsonl"
    other.write_text(json.dumps({"puzzle_id": None, "op": "x", "seconds": 0, "tokens": None}))
    (record,), skipped = read_records([other])
    assert record.worker == "old-log" and skipped == 0
    assert timing.worker_label(tmp_path / "timing-pid42.jsonl") == "pid42"


def test_round_trip_through_the_logger(tmp_path):
    logger = TimingLogger(tmp_path, worker=1)
    logger.record("t1", TRAIN, 3.25, 100)
    logger.record(None, "model_load", 0, None)
    records, skipped = read_records([logger.path])
    assert skipped == 0
    assert records == [
        TimingRecord("t1", "train", 3.25, 100, "1"),
        TimingRecord(None, "model_load", 0.0, None, "1"),
    ]
