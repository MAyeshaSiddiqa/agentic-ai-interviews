"""Exercise ADK-2: a fail-closed safety gate on agent tool calls.

A before_tool_callback runs before every tool execution. Returning a dict
skips the tool and hands that dict to the model as the tool result;
returning None lets the call proceed. The gate:
  * denies tools with no policy (default deny)
  * validates arguments per tool (path sandbox, refund limits, ownership)
  * caps tool calls per invocation via temp: state
  * fails closed when a validator raises
  * appends an audit record by reassigning state (in-place mutation is lost)
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.tools import FunctionTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

POLICY_VERSION = "tool-policy-2026-09"
Validator = Callable[[dict[str, Any], dict[str, Any]], str | None]  # (args, state) -> deny reason


@dataclass(frozen=True)
class ToolPolicy:
    validator: Validator | None = None
    requires_approval: bool = False


def sandboxed_path(root: str) -> Validator:
    root_real = os.path.realpath(root)

    def check(args, state):
        p = args.get("path")
        if not isinstance(p, str) or not p or "\x00" in p:
            return "invalid_path"
        # realpath resolves "..", and symlinks that point outside the root.
        real = os.path.realpath(os.path.join(root_real, p))
        if os.path.commonpath([root_real, real]) != root_real:
            return "path_outside_sandbox"
        return None

    return check


def refund_limit(max_usd: float) -> Validator:
    def check(args, state):
        amt = args.get("amount_usd")
        if isinstance(amt, bool) or not isinstance(amt, (int, float)):
            return "amount_not_numeric"
        # NaN compares False against everything, so "amt > max" alone lets it through.
        if not math.isfinite(amt) or amt <= 0 or amt > max_usd:
            return "amount_out_of_bounds"
        if args.get("order_id") not in set(state.get("user_order_ids", [])):
            return "order_not_owned_by_user"
        return None

    return check


def make_safety_gate(policies: dict[str, ToolPolicy], *, max_calls: int = 5):
    def gate(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> dict | None:
        state = tool_context.state
        n = state.get("temp:tool_calls", 0) + 1
        state["temp:tool_calls"] = n
        policy = policies.get(tool.name)
        try:
            if n > max_calls:
                reason = "tool_call_budget_exhausted"
            elif policy is None:
                reason = "tool_not_allowlisted"
            elif policy.requires_approval and not _approved(state, tool.name, args):
                reason = "requires_human_approval"
            elif policy.validator is not None:
                reason = policy.validator(dict(args), state.to_dict())
            else:
                reason = None
        except Exception as e:  # a crashing validator must not become an allow
            reason = f"validator_error:{type(e).__name__}"
        state["audit_log"] = [
            *state.get("audit_log", []),
            {"tool": tool.name, "args": dict(args), "decision": "deny" if reason else "allow",
             "reason": reason, "policy_version": POLICY_VERSION, "call_index": n},
        ]
        if reason:
            return {"status": "blocked", "reason": reason, "policy_version": POLICY_VERSION}
        return None

    return gate


def _approved(state, tool_name: str, args: dict) -> bool:
    approval = state.get("approved_action")
    return bool(approval) and approval.get("tool") == tool_name and approval.get("args") == dict(args)


class SideEffects:
    """Records real tool executions so tests can prove a blocked call never ran."""

    def __init__(self):
        self.executed: list[tuple[str, dict]] = []


def build_tools(sandbox_root: str, fx: SideEffects):
    def read_file(path: str) -> dict:
        """Read a UTF-8 text file from the agent's workspace."""
        fx.executed.append(("read_file", {"path": path}))
        with open(os.path.join(sandbox_root, path), encoding="utf-8") as f:
            return {"content": f.read()}

    def issue_refund(order_id: str, amount_usd: float) -> dict:
        """Refund an order."""
        fx.executed.append(("issue_refund", {"order_id": order_id, "amount_usd": amount_usd}))
        return {"status": "refunded", "order_id": order_id, "amount_usd": amount_usd}

    def delete_records(table: str, where: str) -> dict:
        """Delete rows from a table."""
        fx.executed.append(("delete_records", {"table": table, "where": where}))
        return {"deleted": 1}

    def run_shell(cmd: str) -> dict:
        """Run a shell command."""
        fx.executed.append(("run_shell", {"cmd": cmd}))
        return {"exit": 0}

    return [FunctionTool(read_file), FunctionTool(issue_refund), FunctionTool(delete_records), FunctionTool(run_shell)]


def build_gated_agent(model: BaseLlm | str, sandbox_root: str, fx: SideEffects, *, max_calls: int = 5,
                      refund_max_usd: float = 100.0) -> LlmAgent:
    policies = {
        "read_file": ToolPolicy(sandboxed_path(sandbox_root)),
        "issue_refund": ToolPolicy(refund_limit(refund_max_usd)),
        "delete_records": ToolPolicy(requires_approval=True),
        # run_shell deliberately has no policy: default deny.
    }
    return LlmAgent(
        name="ops_agent",
        model=model,
        instruction="You are a support operations agent.",
        tools=build_tools(sandbox_root, fx),
        before_tool_callback=make_safety_gate(policies, max_calls=max_calls),
    )
