"""Regressed build of the ADK-3 agent: refunds ineligible orders, says it didn't."""

from evalkit.adk_trajectory_eval.agent import build_support_agent

root_agent = build_support_agent(overeager=True)
