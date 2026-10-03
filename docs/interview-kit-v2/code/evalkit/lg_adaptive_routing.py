"""Exercise LG-2: adaptive judge routing with escalation to a human.

START -> cheap_judge --(confident)--------------------------> finalize -> END
             ^  |--(unsure, rounds left: resample)--'
             |  `--(unsure, out of rounds / high risk)--> strong_judge
                                                       |--(agree)--> finalize
                                                       `--(disagree)--> human_review
human_review: interrupt() for a label; Command(goto=...) routes on the result.

The resample loop is bounded by max_cheap_rounds; recursion_limit is the
backstop that turns a routing bug into GraphRecursionError instead of an
unbounded bill.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt


class TransientJudgeError(Exception):
    pass


class RoutingState(TypedDict, total=False):
    item: dict[str, Any]  # {"item_id", "trace", "risk": "low"|"high"}
    cheap: dict | None  # {"score": int, "confidence": float}
    cheap_rounds: int
    strong: dict | None
    human: dict | None
    final: dict | None
    route: Annotated[list[str], operator.add]


Judge = Callable[[dict[str, Any]], dict]


def build_routing_graph(
    cheap_judge: Judge,
    strong_judge: Judge,
    *,
    checkpointer=None,
    confident: float = 0.85,
    max_cheap_rounds: int | None = 2,
    agree_within: int = 1,
):
    def cheap(state: RoutingState) -> dict:
        v = cheap_judge(state["item"])
        return {"cheap": v, "cheap_rounds": state.get("cheap_rounds", 0) + 1, "route": ["cheap"]}

    def after_cheap(state: RoutingState) -> Literal["finalize", "cheap_judge", "strong_judge"]:
        if state["item"].get("risk") == "high":
            return "strong_judge"
        if state["cheap"]["confidence"] >= confident:
            return "finalize"
        if max_cheap_rounds is None or state["cheap_rounds"] < max_cheap_rounds:
            return "cheap_judge"
        return "strong_judge"

    def strong(state: RoutingState) -> dict:
        return {"strong": strong_judge(state["item"]), "route": ["strong"]}

    def after_strong(state: RoutingState) -> Literal["finalize", "human_review"]:
        c, s = state["cheap"], state["strong"]
        if state["item"].get("risk") == "high":
            return "human_review"
        if abs(c["score"] - s["score"]) <= agree_within and s["confidence"] >= confident:
            return "finalize"
        return "human_review"

    def human_review(state: RoutingState) -> Command[Literal["finalize", "human_review"]]:
        # Everything above interrupt() re-executes on resume. Keep this node
        # free of side effects; notify reviewers from an upstream node.
        answer = interrupt({
            "item_id": state["item"]["item_id"],
            "cheap": state.get("cheap"),
            "strong": state.get("strong"),
            "ask": "Provide an integer score 1-5 and your reviewer id.",
        })
        ok = (isinstance(answer, dict) and isinstance(answer.get("score"), int)
              and not isinstance(answer.get("score"), bool) and 1 <= answer["score"] <= 5
              and isinstance(answer.get("reviewer"), str) and answer["reviewer"])
        if not ok:
            return Command(goto="human_review", update={"route": ["human_invalid"]})
        return Command(goto="finalize", update={"human": answer, "route": ["human"]})

    def finalize(state: RoutingState) -> dict:
        if state.get("human"):
            src, score = "human", state["human"]["score"]
        elif state.get("strong"):
            src, score = "strong", state["strong"]["score"]
        else:
            src, score = "cheap", state["cheap"]["score"]
        return {"final": {"item_id": state["item"]["item_id"], "score": score, "source": src},
                "route": [f"final:{src}"]}

    g = StateGraph(RoutingState)
    g.add_node("cheap_judge", cheap)
    g.add_node("strong_judge", strong,
               retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.001, jitter=False,
                                        retry_on=TransientJudgeError))
    g.add_node("human_review", human_review, destinations=("finalize", "human_review"))
    g.add_node("finalize", finalize)
    g.add_edge(START, "cheap_judge")
    g.add_conditional_edges("cheap_judge", after_cheap, ["finalize", "cheap_judge", "strong_judge"])
    g.add_conditional_edges("strong_judge", after_strong, ["finalize", "human_review"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
