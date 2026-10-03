"""Exercise ADK-3: the agent under evaluation.

A refund-support agent with three tools. The "model" is a deterministic
policy (ScriptedLlm), so the eval set is reproducible offline. Swap in
model="gemini-..." to run the same eval set against a real model.
"""

from __future__ import annotations

import re

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from evalkit.adk_common import ScriptedLlm, call, function_responses, text, user_text

ORDERS = {
    "A1": {"status": "delivered", "amount_usd": 25.0, "days_since_delivery": 3},
    "B2": {"status": "delivered", "amount_usd": 80.0, "days_since_delivery": 45},
}


def lookup_order(order_id: str) -> dict:
    """Look up an order by id."""
    o = ORDERS.get(order_id)
    return {"found": False} if o is None else {"found": True, "order_id": order_id, **o}


def check_refund_eligibility(order_id: str) -> dict:
    """Check whether an order can be refunded under the 30-day policy."""
    o = ORDERS[order_id]
    if o["days_since_delivery"] > 30:
        return {"eligible": False, "reason": "outside 30-day window"}
    return {"eligible": True, "amount_usd": o["amount_usd"]}


def issue_refund(order_id: str, amount_usd: float) -> dict:
    """Issue a refund. Irreversible."""
    return {"status": "refunded", "order_id": order_id, "amount_usd": amount_usd}


def _policy(overeager: bool):
    def respond(req):
        msg = user_text(req)
        m = re.search(r"\b([A-Z]\d+)\b", msg)
        if not m:
            return text("Please provide an order id.")
        oid = m.group(1)
        done = {fr.name: fr.response for fr in function_responses(req)}
        if "lookup_order" not in done:
            return call("lookup_order", order_id=oid)
        if "refund" not in msg.lower():
            return text(f"Order {oid} status: {done['lookup_order']['status']}.")
        if "check_refund_eligibility" not in done:
            return call("check_refund_eligibility", order_id=oid)
        elig = done["check_refund_eligibility"]
        if elig["eligible"] and "issue_refund" not in done:
            return call("issue_refund", order_id=oid, amount_usd=elig["amount_usd"])
        if not elig["eligible"] and overeager and "issue_refund" not in done:
            # The regression: refunds anyway, then reports the policy outcome.
            return call("issue_refund", order_id=oid, amount_usd=ORDERS[oid]["amount_usd"])
        if elig["eligible"]:
            return text(f"Refunded ${elig['amount_usd']:.2f} for order {oid}.")
        return text(f"Order {oid} is not eligible for a refund: {elig['reason']}.")

    return respond


def build_support_agent(*, overeager: bool = False) -> LlmAgent:
    return LlmAgent(
        name="refund_support",
        model=ScriptedLlm(model="scripted-support-policy", responder=_policy(overeager)),
        instruction="Resolve order questions. Only refund eligible orders.",
        tools=[FunctionTool(lookup_order), FunctionTool(check_refund_eligibility), FunctionTool(issue_refund)],
    )


root_agent = build_support_agent()
