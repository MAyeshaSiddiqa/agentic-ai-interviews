"""Each test first reproduces the planted bug, then proves the fix."""

import pytest

from evalkit.debug import judge_service_buggy as buggy
from evalkit.debug import judge_service_fixed as fixed


class Judge:
    def __init__(self):
        self.calls = []

    def __call__(self, item, version):
        self.calls.append((item["item_id"], version))
        return {"score": 1 if version == "v1" else 5}


class FlakyWriter:
    """First write commits, then times out (lost ack)."""

    def __init__(self):
        self.rows = []
        self.keyed = {}
        self.failed_once = False

    def insert(self, row):
        self.rows.append(row)
        if not self.failed_once:
            self.failed_once = True
            raise TimeoutError("ack lost")

    def upsert(self, key, row):
        self.keyed[key] = row
        if not self.failed_once:
            self.failed_once = True
            raise TimeoutError("ack lost")


ITEM = {"item_id": "x1", "prompt": "p", "response": "r"}


class OkWriter:
    def __init__(self):
        self.rows, self.keyed = [], {}

    def insert(self, row):
        self.rows.append(row)

    def upsert(self, key, row):
        self.keyed[key] = row


@pytest.fixture(autouse=True)
def clear_buggy_cache():
    buggy._CACHE.clear()


def test_bug1_cache_key_ignores_judge_version():
    j, w = Judge(), OkWriter()
    buggy.score_items([ITEM], j, w, "v1")
    buggy.score_items([ITEM], j, w, "v2")
    assert j.calls == [("x1", "v1")]  # v2 never ran: the rollout was a no-op
    assert [r["score"] for r in w.rows] == [1]

    j, w, cache, m = Judge(), OkWriter(), {}, fixed.Metrics()
    v1 = fixed.JudgeConfig("judge-model", "v1", "rubric-a")
    v2 = fixed.JudgeConfig("judge-model", "v2", "rubric-a")
    fixed.score_items([ITEM], j, w, v1, cache, m)
    out = fixed.score_items([ITEM], j, w, v2, cache, m)
    assert j.calls == [("x1", "v1"), ("x1", "v2")] and out["x1"]["score"] == 5
    assert fixed.cache_key(ITEM, v1) != fixed.cache_key(ITEM, fixed.JudgeConfig("judge-model", "v1", "rubric-b"))


def test_bug1b_concatenation_collision():
    a = {"item_id": "a", "prompt": "ab", "response": "c"}
    b = {"item_id": "b", "prompt": "a", "response": "bc"}
    assert buggy.cache_key(a) == buggy.cache_key(b)
    cfg = fixed.JudgeConfig("m", "v1", "r")
    assert fixed.cache_key(a, cfg) != fixed.cache_key(b, cfg)


def test_bug2_write_timeout_reruns_judge_and_duplicates_rows():
    j, w = Judge(), FlakyWriter()
    buggy.score_items([ITEM], j, w, "v1")
    assert len(j.calls) == 2 and len(w.rows) == 2

    j, w, m = Judge(), FlakyWriter(), fixed.Metrics()
    fixed.score_items([ITEM], j, w, fixed.JudgeConfig("m", "v1", "r"), {}, m)
    assert len(j.calls) == 1 and len(w.keyed) == 1
    assert m.counters[("retry", (("attempt", 1), ("op", "write")))] == 1


def test_bug2b_exhausted_retries_silently_cache_none():
    def always_timeout(item, version):
        raise TimeoutError

    buggy.score_items([ITEM], always_timeout, OkWriter(), "v1")
    assert buggy._CACHE[buggy.cache_key(ITEM)] is None  # poisoned: never retried again
    with pytest.raises(TimeoutError):
        fixed.score_items([ITEM], always_timeout, OkWriter(), fixed.JudgeConfig("m", "v1", "r"), {}, fixed.Metrics())


def _rows():
    # 40 traces x 5 judged steps each; steps within a trace are near-duplicates.
    return [{"trace_id": f"t{t}", "step": s} for t in range(40) for s in range(5)]


def test_bug3_row_split_leaks_traces_and_mutates_input():
    rows = _rows()
    original = list(rows)
    calib, hold = buggy.split_for_calibration(rows)
    leaked = {r["trace_id"] for r in calib} & {r["trace_id"] for r in hold}
    assert len(leaked) > 20
    assert rows != original  # caller's list shuffled in place

    rows = _rows()
    calib, hold = fixed.split_for_calibration(rows)
    assert not ({r["trace_id"] for r in calib} & {r["trace_id"] for r in hold})
    assert rows == _rows()
    assert 0.15 < len(hold) / len(rows) < 0.45


def test_fixed_split_is_stable_when_data_grows():
    _, hold_before = fixed.split_for_calibration(_rows())
    more = _rows() + [{"trace_id": f"t{t}", "step": 0} for t in range(40, 80)]
    _, hold_after = fixed.split_for_calibration(more)
    before = {r["trace_id"] for r in hold_before}
    after = {r["trace_id"] for r in hold_after}
    assert before <= after  # no trace changes side
