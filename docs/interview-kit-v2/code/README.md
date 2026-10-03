# Reference solutions and tests for 04-coding-exercises

Reference implementations for the 13 exercises in [../04-coding-exercises.md](../04-coding-exercises.md), plus pytest suites. Every LLM call is mocked. The ADK agents use `ScriptedLlm`, a `BaseLlm` subclass whose replies come from a Python function; the LangGraph judges are plain callables. The suite runs with no API key and no network (checked with networking disabled).

Verified against **google-adk 2.10.0** and **langgraph 1.2.12** on Python 3.12. Result: **117 passed in ~15 s**.

## Run

```bash
cd code
python3 -m venv .venv && . .venv/bin/activate    # or: uv venv .venv && . .venv/bin/activate
pip install -r requirements.txt                   # exact transitive pins: requirements-lock.txt
python -m pytest                                  # all suites
python -m pytest tests/test_py04_trajectory.py    # one exercise
```

`google-adk[eval]` is required: `AgentEvaluator` refuses to run without the eval extra. The extra pulls in pandas, scipy and scikit-learn, about 650 MB installed.

## Layout

| Exercise | Solution | Tests |
|---|---|---|
| PY-1 Cohen's kappa, Krippendorff's alpha | `evalkit/agreement.py` | `tests/test_py01_agreement.py` |
| PY-2 Clustered bootstrap CI | `evalkit/bootstrap.py` | `tests/test_py02_bootstrap.py` |
| PY-3 Bradley-Terry ranking | `evalkit/bradley_terry.py` | `tests/test_py03_bradley_terry.py` |
| PY-4 Trajectory matching | `evalkit/trajectory.py` | `tests/test_py04_trajectory.py` |
| PY-5 Async judge runner | `evalkit/judge_runner.py` | `tests/test_py05_judge_runner.py` |
| PY-6 Position-bias probe | `evalkit/position_bias.py` | `tests/test_py06_position_bias.py` |
| ADK-1 Multi-judge evaluator agent | `evalkit/adk_multi_judge.py` | `tests/test_adk01_multi_judge.py` |
| ADK-2 Tool-call safety gate | `evalkit/adk_safety_gate.py` | `tests/test_adk02_safety_gate.py` |
| ADK-3 Eval set for trajectory scoring | `evalkit/adk_trajectory_eval/` (agent, `evalsets/*.test.json`, `test_config.json`, custom metric), `evalkit/adk_trajectory_eval_overeager/` | `tests/test_adk03_evalset.py` |
| LG-1 Resumable scoring graph | `evalkit/lg_resumable_scoring.py` | `tests/test_lg01_resumable_scoring.py` |
| LG-2 Adaptive routing and human escalation | `evalkit/lg_adaptive_routing.py` | `tests/test_lg02_adaptive_routing.py` |
| DEBUG-1 Judge service | `evalkit/debug/judge_service_buggy.py`, `..._fixed.py` | `tests/test_debug01_judge_service.py` |
| DEBUG-2 LangGraph review graph | `evalkit/debug/lg_review_buggy.py`, `..._fixed.py` | `tests/test_debug02_lg_review.py` |

`evalkit/adk_common.py` holds the offline test double (`ScriptedLlm`) and the `run_once` helper.

## Practising against the tests

Copy `code/` somewhere else. Replace the body of the module you are practising with your own implementation, keeping the public names the tests import. Then run that module's test file. The debug tests import both the buggy and the fixed module: each test first reproduces the bug, then checks the fix.

## Notes on the pinned versions

- In ADK 2.x, `SequentialAgent`, `ParallelAgent` and `LoopAgent` emit a `DeprecationWarning` pointing at `google.adk.workflow.Workflow`. They still work, and `Workflow` cannot yet be an `LlmAgent` sub-agent. `pytest.ini` filters these warnings.
- ADK's `AgentEvaluator` imports the module named by `agent_module` and needs it to expose `agent.root_agent`. That is why each eval package's `__init__.py` does `from . import agent`.
- ADK clears `eval_metric.threshold` before calling a custom metric function. Read the threshold from `eval_metric.criterion.threshold` (see `adk_trajectory_eval/metrics.py`).
- `conftest.py` sets `sys.dont_write_bytecode`. On synced filesystems with coarse timestamps, stale `.pyc` files can shadow edited sources.
