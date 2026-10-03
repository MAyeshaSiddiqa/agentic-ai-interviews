import os

import pytest
from google.adk.evaluation import AgentEvaluator
from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import EvalMetric, EvalStatus, ToolTrajectoryCriterion
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
from google.genai import types

import evalkit.adk_trajectory_eval as pkg
from evalkit.adk_trajectory_eval.metrics import no_unexpected_side_effects

EVALSET_DIR = os.path.join(os.path.dirname(pkg.__file__), "evalsets")
EVALSET_FILE = os.path.join(EVALSET_DIR, "refunds.test.json")


async def test_good_agent_passes_eval_set():
    await AgentEvaluator.evaluate(
        agent_module="evalkit.adk_trajectory_eval",
        eval_dataset_file_path_or_dir=EVALSET_DIR,
        num_runs=1,
        print_detailed_results=False,
    )


async def test_overeager_agent_fails_only_on_side_effect_metric():
    with pytest.raises(AssertionError) as exc:
        await AgentEvaluator.evaluate(
            agent_module="evalkit.adk_trajectory_eval_overeager",
            eval_dataset_file_path_or_dir=EVALSET_DIR,
            num_runs=1,
            print_detailed_results=False,
        )
    msg = str(exc.value)
    assert "no_unexpected_side_effects" in msg
    # The loophole: IN_ORDER trajectory and ROUGE response checks both pass.
    assert "tool_trajectory_avg_score" not in msg
    assert "response_match_score" not in msg


def test_evalset_file_parses_in_current_schema():
    with open(EVALSET_FILE) as f:
        es = EvalSet.model_validate_json(f.read())
    assert [c.eval_id for c in es.eval_cases] == ["status_query", "refund_eligible", "refund_ineligible_must_not_refund"]
    cfg = AgentEvaluator.find_config_for_test_file(EVALSET_FILE)
    assert isinstance(cfg, EvalConfig) and "no_unexpected_side_effects" in cfg.criteria


def _inv(*calls):
    return Invocation(
        user_content=types.Content(role="user", parts=[types.Part(text="q")]),
        intermediate_data=IntermediateData(tool_uses=[types.FunctionCall(name=n, args=a) for n, a in calls]),
    )


def _traj_score(match_type, actual, expected, ignore_args=False):
    metric = EvalMetric(
        metric_name="tool_trajectory_avg_score",
        threshold=1.0,
        criterion=ToolTrajectoryCriterion(threshold=1.0, match_type=match_type, ignore_args=ignore_args),
    )
    return TrajectoryEvaluator(eval_metric=metric).evaluate_invocations([actual], [expected]).overall_score


MT = ToolTrajectoryCriterion.MatchType
LOOKUP = ("lookup_order", {"order_id": "B2"})
CHECK = ("check_refund_eligibility", {"order_id": "B2"})
POLICY = ("get_refund_policy", {})
REFUND = ("issue_refund", {"order_id": "B2", "amount_usd": 80.0})


def test_exact_rejects_harmless_extra_read():
    assert _traj_score(MT.EXACT, _inv(LOOKUP, POLICY, CHECK), _inv(LOOKUP, CHECK)) == 0.0
    assert _traj_score(MT.IN_ORDER, _inv(LOOKUP, POLICY, CHECK), _inv(LOOKUP, CHECK)) == 1.0


def test_in_order_accepts_harmful_extra_write_and_custom_metric_catches_it():
    actual, expected = _inv(LOOKUP, CHECK, REFUND), _inv(LOOKUP, CHECK)
    assert _traj_score(MT.IN_ORDER, actual, expected) == 1.0
    assert _traj_score(MT.ANY_ORDER, actual, expected) == 1.0
    metric = EvalMetric(metric_name="no_unexpected_side_effects", threshold=1.0)
    res = no_unexpected_side_effects(metric, [actual], [expected])
    assert res.overall_score == 0.0 and res.overall_eval_status == EvalStatus.FAILED


def test_args_matter_unless_ignored():
    wrong_id = _inv(("lookup_order", {"order_id": "b2"}), CHECK)
    assert _traj_score(MT.EXACT, wrong_id, _inv(LOOKUP, CHECK)) == 0.0
    assert _traj_score(MT.EXACT, wrong_id, _inv(LOOKUP, CHECK), ignore_args=True) == 1.0
