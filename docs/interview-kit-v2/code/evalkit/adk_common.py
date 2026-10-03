"""Offline test double for ADK: a BaseLlm whose replies come from a Python function.

ADK accepts a BaseLlm instance as LlmAgent.model, so agents run end to end
(runner, callbacks, tools, state, AgentEvaluator) without any API key.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import ConfigDict, Field

Responder = Callable[[LlmRequest], LlmResponse]


class ScriptedLlm(BaseLlm):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Any = None  # Responder
    requests: list = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        await asyncio.sleep(0)
        yield self.responder(llm_request)


def text(s: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=s)]))


def call(name: str, **args: Any) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])
    )


def function_responses(req: LlmRequest) -> list[types.FunctionResponse]:
    return [p.function_response for c in req.contents for p in (c.parts or []) if p.function_response]


def user_text(req: LlmRequest) -> str:
    return " ".join(p.text or "" for c in req.contents if c.role == "user" for p in (c.parts or []) if p.text)


def system_text(req: LlmRequest) -> str:
    si = req.config.system_instruction if req.config else None
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    parts = si.parts if isinstance(si, types.Content) else si
    return " ".join(getattr(p, "text", "") or "" for p in parts)


async def run_once(agent: BaseAgent, message: str, state: dict | None = None, app_name: str = "kit"):
    """Run one user turn; return (events, final session state)."""
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(app_name=app_name, user_id="u", state=state or {})
    events = []
    async for ev in runner.run_async(
        user_id="u", session_id=session.id, new_message=types.Content(role="user", parts=[types.Part(text=message)])
    ):
        events.append(ev)
    session = await runner.session_service.get_session(app_name=app_name, user_id="u", session_id=session.id)
    return events, dict(session.state)
