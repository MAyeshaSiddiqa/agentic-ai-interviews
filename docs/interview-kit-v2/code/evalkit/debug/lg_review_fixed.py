"""DEBUG-2 (fixed)."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


def latest_by_item(left: dict | None, right: dict | None) -> dict:
    """Keyed reducer: a rescore replaces the item's entry instead of appending."""
    return {**(left or {}), **(right or {})}


class State(TypedDict, total=False):
    run_id: str
    items: list[dict]
    scores: Annotated[dict[str, dict], latest_by_item]
    review_target: str
    summary: dict


class TicketSystem:
    def __init__(self):
        self.tickets: dict[str, dict] = {}
        self.create_calls = 0

    def create(self, idempotency_key: str, payload: dict) -> None:
        self.create_calls += 1
        self.tickets.setdefault(idempotency_key, payload)


def build(judge, tickets: TicketSystem, checkpointer):
    def score(state):
        return {"scores": {it["item_id"]: judge(it) for it in state["items"]}}

    def rescore_low_confidence(state):
        # Return only the delta; the reducer merges it.
        return {"scores": {iid: {**judge({"item_id": iid}), "rescored": True}
                           for iid, s in state["scores"].items() if s["confidence"] < 0.5}}

    def open_ticket(state):
        # Side effect in its own node: once this node completes it is
        # checkpointed and does not re-run when review resumes.
        worst = min(state["scores"], key=lambda iid: state["scores"][iid]["score"])
        tickets.create(f"{state['run_id']}:{worst}", {"item_id": worst})
        return {"review_target": worst}

    def review(state):
        decision = interrupt({"item_id": state["review_target"]})
        return {"summary": {"reviewed": state["review_target"], "decision": decision}}

    def summarize(state):
        ids = [it["item_id"] for it in state["items"]]
        assert set(state["scores"]) == set(ids), "score/item cardinality mismatch"
        vals = [state["scores"][i]["score"] for i in ids]
        return {"summary": {**state.get("summary", {}), "n": len(vals), "mean": sum(vals) / len(vals)}}

    g = StateGraph(State)
    for name, fn in [("score", score), ("rescore", rescore_low_confidence), ("open_ticket", open_ticket),
                     ("review", review), ("summarize", summarize)]:
        g.add_node(name, fn)
    g.add_edge(START, "score")
    g.add_edge("score", "rescore")
    g.add_edge("rescore", "open_ticket")
    g.add_edge("open_ticket", "review")
    g.add_edge("review", "summarize")
    g.add_edge("summarize", END)
    return g.compile(checkpointer=checkpointer)


def run_config(dataset: str, run_id: str) -> dict:
    # One thread per run. Reusing the dataset name resumes the previous run's
    # state, and an accumulating reducer then adds last night's scores.
    return {"configurable": {"thread_id": f"{dataset}:{run_id}"}}
