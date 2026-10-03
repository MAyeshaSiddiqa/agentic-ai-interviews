"""Exercise ADK-1: multi-judge evaluator agent.

SequentialAgent[
    ParallelAgent[judge_1 .. judge_k],   # LlmAgents, one output_key each
    VerdictAggregator,                   # deterministic BaseAgent, no LLM
]

The trace under evaluation arrives in session state ("trace"). Judges run with
include_contents="none" and an instruction provider, so they see only the
rubric and the trace: not the chat history, not each other.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

RAW_KEY = "judge_raw__{}"  # not "judge:x" style: app:/user:/temp: prefixes change persistence scope
META_KEY = "judge_meta__{}"


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    model: BaseLlm | str
    rubric: str
    version: str


def _instruction(spec: JudgeSpec):
    def provider(ctx: ReadonlyContext) -> str:
        # An instruction *provider* is not run through {state_var} templating,
        # so braces in rubric text (e.g. "output {score}") cannot become state
        # lookups that raise KeyError or pull in an unrelated state value.
        return (
            f"You are evaluation judge '{spec.name}' (rubric {spec.version}).\n"
            f"Rubric: {spec.rubric}\n"
            "Reply with a single JSON object with keys score (integer 1-5) and rationale (string).\n"
            "=== TRACE START ===\n"
            f"{ctx.state.get('trace', '')}\n"
            "=== TRACE END ==="
        )

    return provider


def _guard(spec: JudgeSpec, max_trace_chars: int):
    def before_model(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
        callback_context.state[META_KEY.format(spec.name)] = {
            "judge_version": spec.version,
            "model": spec.model if isinstance(spec.model, str) else spec.model.model,
        }
        trace = callback_context.state.get("trace", "")
        if not trace:
            return _abstain("empty_trace")
        if len(trace) > max_trace_chars:
            # Judging a silently truncated trace is worse than abstaining.
            return _abstain("trace_too_long")
        return None  # proceed to the model

    return before_model


def _abstain(reason: str) -> LlmResponse:
    body = json.dumps({"abstain": True, "reason": reason})
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=body)]))


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def parse_judgement(raw: Any) -> tuple[int | None, str]:
    """Return (score, reason). score None means abstain: never coerce to 0."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "missing_output"
    m = _FENCE.match(raw)
    body = m.group(1) if m else raw
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None, "unparseable"
    if not isinstance(obj, dict):
        return None, "not_an_object"
    if obj.get("abstain"):
        return None, f"abstained:{obj.get('reason', '')}"
    s = obj.get("score")
    if isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= 5:
        return None, "score_out_of_schema"
    return s, "ok"


class VerdictAggregator(BaseAgent):
    judge_names: list[str]
    min_quorum: int = 2
    disagreement_threshold: int = 2

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        per_judge: dict[str, dict] = {}
        for name in self.judge_names:
            score, reason = parse_judgement(ctx.session.state.get(RAW_KEY.format(name)))
            per_judge[name] = {"score": score, "status": reason, **(ctx.session.state.get(META_KEY.format(name)) or {})}
        valid = [v["score"] for v in per_judge.values() if v["score"] is not None]
        if len(valid) < self.min_quorum:
            verdict = {"status": "insufficient_quorum", "score": None, "n_valid": len(valid)}
        else:
            spread = max(valid) - min(valid)
            verdict = {
                "status": "needs_human" if spread >= self.disagreement_threshold else "ok",
                "score": statistics.median(valid),
                "spread": spread,
                "n_valid": len(valid),
            }
        verdict["judges"] = per_judge
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
            actions=EventActions(state_delta={"verdict": verdict}),
        )


def build_multi_judge(
    judges: list[JudgeSpec],
    *,
    min_quorum: int = 2,
    disagreement_threshold: int = 2,
    max_trace_chars: int = 20_000,
) -> SequentialAgent:
    names = [j.name for j in judges]
    if len(set(names)) != len(names):
        # Duplicate names would share an output_key: under ParallelAgent the
        # last writer wins and one judge silently disappears.
        raise ValueError(f"judge names must be unique: {names}")
    if min_quorum > len(judges):
        raise ValueError("min_quorum exceeds number of judges")
    judge_agents = [
        LlmAgent(
            name=j.name,
            model=j.model,
            instruction=_instruction(j),
            include_contents="none",
            output_key=RAW_KEY.format(j.name),
            generate_content_config=types.GenerateContentConfig(temperature=0.0),
            before_model_callback=_guard(j, max_trace_chars),
        )
        for j in judges
    ]
    return SequentialAgent(
        name="multi_judge",
        sub_agents=[
            ParallelAgent(name="judge_panel", sub_agents=judge_agents),
            VerdictAggregator(
                name="aggregator",
                judge_names=names,
                min_quorum=min_quorum,
                disagreement_threshold=disagreement_threshold,
            ),
        ],
    )
