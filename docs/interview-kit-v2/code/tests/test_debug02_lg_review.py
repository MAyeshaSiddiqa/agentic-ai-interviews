"""Each test reproduces a planted bug in the shipped graph, then checks the fix."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from evalkit.debug import lg_review_buggy as buggy
from evalkit.debug import lg_review_fixed as fixed

ITEMS = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}]


class Judge:
    """Item 'b' is low-confidence on first pass; rescoring raises its score."""

    def __init__(self):
        self.seen = set()

    def __call__(self, item):
        iid = item["item_id"]
        first = iid not in self.seen
        self.seen.add(iid)
        if iid == "b" and first:
            return {"score": 1.0, "confidence": 0.2}
        return {"score": {"a": 4.0, "b": 3.0, "c": 5.0}[iid], "confidence": 0.9}


def run_buggy(dataset="nightly", saver=None, tickets=None):
    saver = saver or InMemorySaver()
    tickets = [] if tickets is None else tickets
    g = buggy.build(Judge(), tickets, saver)
    cfg = buggy.run_config(dataset)
    g.invoke({"items": ITEMS}, cfg)
    out = g.invoke(Command(resume="approve"), cfg)
    return out, tickets, saver


def run_fixed(run_id, dataset="nightly", saver=None, tickets=None):
    saver = saver or InMemorySaver()
    tickets = tickets or fixed.TicketSystem()
    g = fixed.build(Judge(), tickets, saver)
    cfg = fixed.run_config(dataset, run_id)
    g.invoke({"run_id": run_id, "items": ITEMS}, cfg)
    out = g.invoke(Command(resume="approve"), cfg)
    return out, tickets, saver


def test_bug_reducer_duplicates_whole_list():
    out, _, _ = run_buggy()
    # operator.add + returning state["scores"] + fresh => old list appended to itself.
    assert out["summary"]["n"] == 7  # 3 items -> 3 + (3 + 1)
    ok, _, _ = run_fixed("r1")
    assert ok["summary"]["n"] == 3
    assert ok["summary"]["mean"] == (4.0 + 3.0 + 5.0) / 3


def test_bug_side_effect_before_interrupt_runs_twice():
    _, tickets, _ = run_buggy()
    assert len(tickets) == 2  # node body re-executes from the top on resume
    _, t, _ = run_fixed("r1")
    assert t.create_calls == 1 and len(t.tickets) == 1


def test_bug_thread_id_reuse_accumulates_across_runs():
    saver, tickets = InMemorySaver(), []
    first, _, _ = run_buggy(saver=saver, tickets=tickets)
    second, _, _ = run_buggy(saver=saver, tickets=tickets)
    assert second["summary"]["n"] > first["summary"]["n"]  # "drifts run over run"

    saver = InMemorySaver()
    a, _, _ = run_fixed("2026-09-29", saver=saver)
    b, _, _ = run_fixed("2026-09-30", saver=saver)
    assert a["summary"] == b["summary"]
