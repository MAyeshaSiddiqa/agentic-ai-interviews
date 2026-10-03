"""Exercise LG-1: resumable map-reduce scoring graph with idempotent publish.

START -> plan --(Send per unscored item)--> score_item --> aggregate -> publish -> END
                `--(nothing to score)-------------------'

* scores use a keyed-merge reducer, so replayed or retried writes cannot
  duplicate an item
* score_item has a RetryPolicy for transient judge errors only
* with a checkpointer, a crash mid-fan-out keeps the writes of the tasks that
  finished; resuming with invoke(None, config) re-runs only the failed ones
* publish writes under a content-derived key, so re-running it is a no-op
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send


class TransientJudgeError(Exception):
    pass


class FatalJudgeError(Exception):
    pass


def merge_first_write_wins(left: dict | None, right: dict | None) -> dict:
    """Keyed merge. Re-delivered updates for an existing key are ignored, so a
    published score cannot change underneath its summary hash."""
    out = dict(left or {})
    for k, v in (right or {}).items():
        out.setdefault(k, v)
    return out


class ScoringState(TypedDict, total=False):
    run_id: str
    judge_version: str
    items: list[dict[str, Any]]  # {"item_id": str, "trace": str}
    scores: Annotated[dict[str, dict], merge_first_write_wins]
    summary: dict[str, Any]
    log: Annotated[list[str], operator.add]


class ItemTask(TypedDict):
    run_id: str
    judge_version: str
    item: dict[str, Any]


class IdempotentSink:
    """Stand-in for a results table with a unique key."""

    def __init__(self):
        self.records: dict[str, dict] = {}
        self.attempts = 0

    def upsert(self, key: str, record: dict) -> bool:
        self.attempts += 1
        if key in self.records:
            return False
        self.records[key] = record
        return True


Judge = Callable[[dict[str, Any], str], dict]  # (item, judge_version) -> {"score": float, ...}


def build_scoring_graph(judge: Judge, sink: IdempotentSink, *, checkpointer=None, max_attempts: int = 3):
    def plan(state: ScoringState) -> dict:
        ids = [it["item_id"] for it in state.get("items", [])]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate item_id in batch: keyed merge would drop one silently")
        return {"log": [f"plan:{len(ids)}"]}

    def fan_out(state: ScoringState):
        done = state.get("scores", {})
        todo = [it for it in state.get("items", []) if it["item_id"] not in done]
        if not todo:
            return "aggregate"
        return [Send("score_item", {"run_id": state["run_id"], "judge_version": state["judge_version"], "item": it})
                for it in todo]

    def score_item(task: ItemTask) -> dict:
        v = judge(task["item"], task["judge_version"])
        return {"scores": {task["item"]["item_id"]: {**v, "judge_version": task["judge_version"]}},
                "log": [f"scored:{task['item']['item_id']}"]}

    def aggregate(state: ScoringState) -> dict:
        items = state.get("items", [])
        scores = state.get("scores", {})
        missing = sorted(it["item_id"] for it in items if it["item_id"] not in scores)
        vals = [scores[it["item_id"]]["score"] for it in items if it["item_id"] in scores]
        body = {"run_id": state["run_id"], "judge_version": state["judge_version"],
                "n_items": len(items), "n_scored": len(vals), "missing": missing,
                "mean": (sum(vals) / len(vals)) if vals else None,
                "scores": {k: scores[k]["score"] for k in sorted(scores)}}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
        return {"summary": {**body, "digest": digest}}

    def publish(state: ScoringState) -> dict:
        s = state["summary"]
        if s["missing"]:
            raise RuntimeError(f"refusing to publish incomplete run: missing {s['missing']}")
        created = sink.upsert(f"{s['run_id']}:{s['judge_version']}:{s['digest']}", s)
        return {"log": ["published" if created else "publish_noop"]}

    g = StateGraph(ScoringState)
    g.add_node("plan", plan)
    g.add_node(
        "score_item",
        score_item,
        retry_policy=RetryPolicy(max_attempts=max_attempts, initial_interval=0.001, jitter=False,
                                 retry_on=TransientJudgeError),
    )
    g.add_node("aggregate", aggregate)
    g.add_node("publish", publish)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["score_item", "aggregate"])
    g.add_edge("score_item", "aggregate")
    g.add_edge("aggregate", "publish")
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)
