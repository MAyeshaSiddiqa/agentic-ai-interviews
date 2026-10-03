import itertools

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from evalkit.lg_adaptive_routing import TransientJudgeError, build_routing_graph


def const(score, conf):
    calls = []

    def j(item):
        calls.append(item["item_id"])
        return {"score": score, "confidence": conf}

    j.calls = calls
    return j


def item(risk="low"):
    return {"item": {"item_id": "t1", "trace": "...", "risk": risk}}


def cfg(t="t1", **kw):
    return {"configurable": {"thread_id": t}, **kw}


def test_confident_cheap_judge_finalizes_directly():
    strong = const(1, 0.99)
    out = build_routing_graph(const(4, 0.95), strong).invoke(item())
    assert out["final"] == {"item_id": "t1", "score": 4, "source": "cheap"}
    assert out["route"] == ["cheap", "final:cheap"] and strong.calls == []


def test_unsure_cheap_resamples_then_escalates_and_strong_agrees():
    cheap = const(3, 0.5)
    out = build_routing_graph(cheap, const(4, 0.9), max_cheap_rounds=2).invoke(item())
    assert out["route"] == ["cheap", "cheap", "strong", "final:strong"]
    assert len(cheap.calls) == 2


def test_disagreement_interrupts_for_human_and_resumes():
    g = build_routing_graph(const(1, 0.5), const(5, 0.9), checkpointer=InMemorySaver())
    out = g.invoke(item(), cfg())
    assert "__interrupt__" in out
    payload = out["__interrupt__"][0].value
    assert payload["item_id"] == "t1" and payload["strong"]["score"] == 5
    assert g.get_state(cfg()).next == ("human_review",)

    out = g.invoke(Command(resume={"score": 2, "reviewer": "r-17"}), cfg())
    assert out["final"]["source"] == "human" and out["final"]["score"] == 2
    assert out["route"][-2:] == ["human", "final:human"]


def test_invalid_human_label_reasks():
    g = build_routing_graph(const(1, 0.5), const(5, 0.9), checkpointer=InMemorySaver())
    g.invoke(item(), cfg())
    out = g.invoke(Command(resume={"score": "five"}), cfg())
    assert "__interrupt__" in out and "final" not in out
    out = g.invoke(Command(resume={"score": 5, "reviewer": "r-1"}), cfg())
    assert out["final"]["score"] == 5 and "human_invalid" in out["route"]


def test_high_risk_always_gets_human_even_if_judges_agree():
    g = build_routing_graph(const(5, 0.99), const(5, 0.99), checkpointer=InMemorySaver())
    out = g.invoke(item(risk="high"), cfg())
    assert "__interrupt__" in out and out["route"] == ["cheap", "strong"]


def test_threads_are_isolated():
    g = build_routing_graph(const(1, 0.5), const(5, 0.9), checkpointer=InMemorySaver())
    g.invoke(item(), cfg("a"))
    assert g.get_state(cfg("b")).next == ()


def test_unbounded_loop_hits_recursion_limit():
    never_sure = const(3, 0.1)
    g = build_routing_graph(never_sure, const(3, 0.9), max_cheap_rounds=None)
    with pytest.raises(GraphRecursionError):
        g.invoke(item(), {"recursion_limit": 8})
    assert len(never_sure.calls) == 8


def test_strong_judge_transient_errors_retried():
    attempts = itertools.count(1)

    def flaky(it):
        if next(attempts) < 3:
            raise TransientJudgeError("503")
        return {"score": 3, "confidence": 0.9}

    out = build_routing_graph(const(3, 0.2), flaky, max_cheap_rounds=1).invoke(item())
    assert out["final"]["source"] == "strong"
