from collections import Counter

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from evalkit.lg_resumable_scoring import (
    FatalJudgeError,
    IdempotentSink,
    TransientJudgeError,
    build_scoring_graph,
    merge_first_write_wins,
)

ITEMS = [{"item_id": f"i{k}", "trace": f"trace {k}"} for k in range(5)]


class CountingJudge:
    def __init__(self, fail_transient=None, fail_fatal=None):
        self.calls = Counter()
        self.fail_transient = dict(fail_transient or {})  # item_id -> remaining transient failures
        self.fail_fatal = set(fail_fatal or ())

    def __call__(self, item, version):
        iid = item["item_id"]
        self.calls[iid] += 1
        if iid in self.fail_fatal:
            raise FatalJudgeError(f"judge backend down for {iid}")
        if self.fail_transient.get(iid, 0) > 0:
            self.fail_transient[iid] -= 1
            raise TransientJudgeError("429")
        return {"score": float(int(iid[1:]) % 3)}


def cfg(thread="run-1", **kw):
    return {"configurable": {"thread_id": thread}, **kw}


def inputs(run_id="run-1", version="judge-v3", items=ITEMS):
    return {"run_id": run_id, "judge_version": version, "items": items}


def test_happy_path_fans_out_and_publishes_once():
    judge, sink = CountingJudge(), IdempotentSink()
    g = build_scoring_graph(judge, sink, checkpointer=InMemorySaver())
    out = g.invoke(inputs(), cfg())
    assert set(out["scores"]) == {it["item_id"] for it in ITEMS}
    assert out["summary"]["mean"] == pytest.approx((0 + 1 + 2 + 0 + 1) / 5)
    assert all(v == 1 for v in judge.calls.values())
    assert len(sink.records) == 1 and "published" in out["log"]


def test_transient_errors_retried_by_retry_policy():
    judge, sink = CountingJudge(fail_transient={"i2": 2}), IdempotentSink()
    out = build_scoring_graph(judge, sink).invoke(inputs())
    assert judge.calls["i2"] == 3 and len(out["scores"]) == 5


def test_retry_budget_exhausted_surfaces_error():
    judge, sink = CountingJudge(fail_transient={"i2": 5}), IdempotentSink()
    with pytest.raises(TransientJudgeError):
        build_scoring_graph(judge, sink, max_attempts=3).invoke(inputs())
    assert judge.calls["i2"] == 3 and not sink.records


def test_crash_then_resume_rescores_only_failed_item():
    judge, sink, saver = CountingJudge(fail_fatal={"i3"}), IdempotentSink(), InMemorySaver()
    g = build_scoring_graph(judge, sink, checkpointer=saver)
    with pytest.raises(FatalJudgeError):
        g.invoke(inputs(), cfg())
    assert judge.calls["i3"] == 1  # fatal errors are not retried
    snap = g.get_state(cfg())
    assert snap.next == ("score_item",)  # pending: only the failed Send task
    assert not sink.records

    judge.fail_fatal.clear()
    out = g.invoke(None, cfg())  # resume from checkpoint, no new input
    assert out["summary"]["n_scored"] == 5
    assert judge.calls == Counter({"i0": 1, "i1": 1, "i2": 1, "i3": 2, "i4": 1})
    assert len(sink.records) == 1


def test_rerun_same_thread_is_idempotent():
    judge, sink = CountingJudge(), IdempotentSink()
    g = build_scoring_graph(judge, sink, checkpointer=InMemorySaver())
    g.invoke(inputs(), cfg())
    out = g.invoke(inputs(), cfg())  # e.g. an orchestrator retry after a lost ack
    assert sum(judge.calls.values()) == 5  # nothing re-scored
    assert sink.attempts == 2 and len(sink.records) == 1
    assert out["log"][-1] == "publish_noop"


def test_new_judge_version_needs_new_thread_and_gets_new_record():
    judge, sink, saver = CountingJudge(), IdempotentSink(), InMemorySaver()
    g = build_scoring_graph(judge, sink, checkpointer=saver)
    g.invoke(inputs(version="judge-v3"), cfg("run-1"))
    g.invoke(inputs(run_id="run-2", version="judge-v4"), cfg("run-2"))
    assert len(sink.records) == 2 and sum(judge.calls.values()) == 10


def test_empty_batch_and_duplicate_ids():
    judge, sink = CountingJudge(), IdempotentSink()
    g = build_scoring_graph(judge, sink)
    out = g.invoke(inputs(items=[]))
    assert out["summary"]["n_items"] == 0 and out["summary"]["mean"] is None
    with pytest.raises(ValueError):
        g.invoke(inputs(items=[ITEMS[0], ITEMS[0]]))


def test_reducer_is_idempotent_and_first_write_wins():
    a = {"i1": {"score": 1.0}}
    assert merge_first_write_wins(a, a) == a
    assert merge_first_write_wins(a, {"i1": {"score": 5.0}, "i2": {"score": 2.0}}) == {"i1": {"score": 1.0}, "i2": {"score": 2.0}}
    assert merge_first_write_wins(None, None) == {}
