"""DEBUG-2 (as shipped): LangGraph rescoring + human review. Planted bugs.

Symptom report: "Nightly mean score drifts upward run over run on an
unchanged dataset; the dashboard shows more than twice as many scores as items; reviewers
get two tickets for every escalation."
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class State(TypedDict, total=False):
    items: list[dict]
    scores: Annotated[list[dict], operator.add]
    summary: dict


def build(judge, tickets, checkpointer):
    def score(state):
        return {"scores": [{"item_id": it["item_id"], **judge(it)} for it in state["items"]]}

    def rescore_low_confidence(state):
        redo = [s for s in state["scores"] if s["confidence"] < 0.5]
        fresh = [{"item_id": s["item_id"], **judge({"item_id": s["item_id"]}), "rescored": True} for s in redo]
        return {"scores": state["scores"] + fresh}

    def review(state):
        worst = min(state["scores"], key=lambda s: s["score"])
        tickets.append({"item_id": worst["item_id"]})
        decision = interrupt({"item_id": worst["item_id"]})
        return {"summary": {"reviewed": worst["item_id"], "decision": decision}}

    def summarize(state):
        vals = [s["score"] for s in state["scores"]]
        return {"summary": {**state.get("summary", {}), "n": len(vals), "mean": sum(vals) / len(vals)}}

    g = StateGraph(State)
    for name, fn in [("score", score), ("rescore", rescore_low_confidence), ("review", review), ("summarize", summarize)]:
        g.add_node(name, fn)
    g.add_edge(START, "score")
    g.add_edge("score", "rescore")
    g.add_edge("rescore", "review")
    g.add_edge("review", "summarize")
    g.add_edge("summarize", END)
    return g.compile(checkpointer=checkpointer)


def run_config(dataset: str) -> dict:
    return {"configurable": {"thread_id": dataset}}
