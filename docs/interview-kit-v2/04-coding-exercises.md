# 04 — Coding Exercises (v2)

Thirteen exercises: six in Python, three in Google ADK, two in LangGraph, and two debug-under-pressure rounds. Every exercise has a reference solution and a pytest suite in [code/](code/). The suite mocks every LLM call and runs offline with no API key: **117 tests pass** against **google-adk 2.10.0** and **langgraph 1.2.12** (Python 3.12). Pins are in [code/requirements.txt](code/requirements.txt); run instructions are in [code/README.md](code/README.md).

Each exercise has the same parts: **Prompt**, **Time box**, **Starter code**, **Edge cases**, **Hidden failure mode** (the thing most candidates miss), **Test strategy**, **Senior vs Staff**, and **Reference**.

How to use this file: write the solution yourself against the starter code, run the matching test file, and only then read the reference. A solution that "looks right" but has not been run against the tests does not count.

---

## API reference (verified against the installed versions)

The v1 kit hedged with "version-dependent naming". These are the exact names, checked against the installed packages.

**Google ADK 2.10.0**

| Primitive | Import / signature | Used in |
|---|---|---|
| `LlmAgent` (alias `Agent`) | `from google.adk.agents import LlmAgent`. Fields: `model` (str or a `BaseLlm` instance), `instruction` (str or `InstructionProvider(ReadonlyContext) -> str`), `tools`, `output_key`, `include_contents` (`"default"` or `"none"`), `generate_content_config`, `before_model_callback`, `before_tool_callback`, `after_tool_callback` | ADK-1, 2, 3 |
| `SequentialAgent`, `ParallelAgent`, `LoopAgent` | `from google.adk.agents import ...`. `LoopAgent(max_iterations=...)` stops on `max_iterations` or on an event with `EventActions(escalate=True)`. All three emit a `DeprecationWarning` in 2.x (pointing at `google.adk.workflow.Workflow`) and still work. | ADK-1 |
| Custom agent | subclass `BaseAgent` and implement `async def _run_async_impl(self, ctx: InvocationContext)`, yielding `Event(author=self.name, invocation_id=ctx.invocation_id, actions=EventActions(state_delta={...}))` | ADK-1 |
| `FunctionTool` | `from google.adk.tools import FunctionTool, ToolContext`. `FunctionTool(func, *, require_confirmation=False)`. The docstring becomes the tool description. A `tool_context: ToolContext` parameter is injected, not exposed to the model. | ADK-2, 3 |
| `before_tool_callback` | `(tool: BaseTool, args: dict, tool_context: ToolContext) -> dict \| None`. Returning a dict **skips the tool**, and that dict is sent to the model as the tool result. `None` lets the call proceed. | ADK-2 |
| `before_model_callback` | `(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse \| None`. Returning an `LlmResponse` **skips the model call**. | ADK-1 |
| Session state | `tool_context.state[...]` / `callback_context.state[...]`. The `temp:` prefix is invocation-scoped; `user:` and `app:` are shared across sessions. `output_key` writes the agent's final text into `state[output_key]`. `{var}` in a string instruction is filled from state; an instruction *provider* is not templated. | all ADK |
| Offline model | subclass `google.adk.models.base_llm.BaseLlm` and implement `async def generate_content_async(self, llm_request, stream=False)`, yielding `LlmResponse` objects. See `code/evalkit/adk_common.py`. | all ADK |
| Runner | `from google.adk.runners import InMemoryRunner`; `runner.session_service.create_session(...)`; `runner.run_async(user_id=, session_id=, new_message=types.Content(...))` | all ADK |
| Evaluation | `from google.adk.evaluation import AgentEvaluator`; `await AgentEvaluator.evaluate(agent_module, eval_dataset_file_path_or_dir, num_runs=...)`. It finds `*.test.json` files (EvalSet schema) and a sibling `test_config.json`. Criteria: `tool_trajectory_avg_score` (`match_type`: `EXACT` / `IN_ORDER` / `ANY_ORDER`, plus `ignore_args`) and `response_match_score` (ROUGE-1 F). Custom metrics use `custom_metrics.<name>.code_config.name`. Requires `pip install "google-adk[eval]"`. | ADK-3 |

**LangGraph 1.2.12**

| Primitive | Import / signature | Used in |
|---|---|---|
| `StateGraph`, `START`, `END` | `from langgraph.graph import StateGraph, START, END` | LG-1, 2, DEBUG-2 |
| Typed state with reducers | `class S(TypedDict): scores: Annotated[dict, merge_fn]`. A channel without a reducer is last-write-wins, and **raises `InvalidUpdateError`** if two tasks write it in the same superstep. | all LG |
| `Send` | `from langgraph.types import Send`. A conditional-edge function returns `[Send("node", payload), ...]` for map fan-out; each task receives `payload`, not the graph state. | LG-1 |
| Conditional edges | `g.add_conditional_edges(source, path_fn, path_map)`. `path_fn` returns a node name, a list of names, or `Send`s. | LG-1, 2 |
| `Command` | A node returns `Command(goto="x", update={...})` to route and update together. Declare `add_node(..., destinations=(...))` for graph rendering. `Command(resume=value)` is the input that resumes an interrupt. | LG-2 |
| `interrupt` | `from langgraph.types import interrupt`; `value = interrupt(payload)`. Needs a checkpointer. `invoke` returns a state containing `"__interrupt__"`. **On resume, the node re-executes from its first line.** | LG-2, DEBUG-2 |
| `RetryPolicy` | `from langgraph.types import RetryPolicy`; `RetryPolicy(initial_interval=0.5, backoff_factor=2.0, max_interval=128.0, max_attempts=3, jitter=True, retry_on=...)`, passed as `add_node(..., retry_policy=...)` | LG-1, 2 |
| Checkpointer | `from langgraph.checkpoint.memory import InMemorySaver`; `g.compile(checkpointer=...)`; `config={"configurable": {"thread_id": ...}}`; `g.get_state(config).next`; `g.invoke(None, config)` resumes after a crash | LG-1, 2, DEBUG-2 |
| `recursion_limit` | `config={"recursion_limit": N}` caps supersteps. Exceeding it raises `langgraph.errors.GraphRecursionError`. | LG-2 |

---

## Python

### PY-1 — Chance-corrected agreement: Cohen's kappa and Krippendorff's alpha

**Prompt.** You have LLM-judge labels and human labels for the same traces. Implement `cohen_kappa(a, b, labels=None, weights=None|"linear"|"quadratic")` for two raters, and `krippendorff_alpha(units, level="nominal"|"ordinal"|"interval"|"ratio")` for any number of raters with missing labels. No statistics libraries.

**Time box.** 45 min (kappa 15, alpha 30).

**Starter code.**

```python
from collections.abc import Hashable, Sequence
from typing import Literal

def cohen_kappa(rater_a: Sequence[Hashable | None], rater_b: Sequence[Hashable | None], *,
                labels: Sequence[Hashable] | None = None,
                weights: Literal["linear", "quadratic"] | None = None) -> float:
    """kappa = 1 - D_obs / D_exp. Drop pairs with a missing label."""
    raise NotImplementedError

def krippendorff_alpha(units: Sequence[Sequence[float | Hashable | None]], *,
                       level: Literal["nominal", "ordinal", "interval", "ratio"] = "nominal") -> float:
    """units[u] = the values each rater gave unit u (None = not rated).
    Build the coincidence matrix; alpha = 1 - D_o / D_e."""
    raise NotImplementedError
```

**Edge cases.**
- A rater is constant, and both raters use the same single label: expected disagreement is 0 and kappa is **undefined**. Return NaN, not 0 or 1.
- Missing labels: kappa drops the pair. Alpha keeps the unit if it still has at least 2 values; units with a single value are unpairable and must not affect the result.
- Weighted kappa on string labels needs an explicit order. Refuse rather than sort alphabetically ("fail" < "partial" < "pass" happens to work; "bad" < "good" < "ok" does not).
- The ordinal distance in alpha depends on the observed marginals, not on the numeric gaps between values.
- `True == 1` in Python. Don't let booleans pass as numeric ratings.

**Hidden failure mode.** The **kappa paradox**. With 95% "pass" prevalence, 96% raw agreement can give kappa below 0.5. The candidate who reports only kappa, or only raw agreement, misleads either way. The test builds exactly this case. Report prevalence, per-class agreement (or PABAK), and kappa together.

**Test strategy.**
- Krippendorff's published worked example (4 coders × 12 units, with missing data): nominal 0.743, ordinal 0.815, interval 0.849, ratio 0.797.
- A hand-computed 2×2 table (kappa 0.40).
- Cross-check against `sklearn.metrics.cohen_kappa_score`, all three weightings, on random data.
- Properties: adding single-rated units leaves alpha unchanged; for 2 raters with no missing data and large n, alpha ≈ kappa.

**Senior vs Staff.**
- *Senior:* correct formulas, NaN on degenerate input, the published values reproduced.
- *Staff:* also explains which coefficient to put on the dashboard and why. Alpha handles missing labels and several raters. Quadratic-weighted kappa equals an ICC under mild conditions. Kappa's ceiling is the human–human agreement, so a judge at κ = 0.62 against humans who agree with each other at κ = 0.65 is at the noise ceiling. Adds a bootstrap CI clustered by trace (PY-2) before claiming that one judge beats another.

**Reference.** [code/evalkit/agreement.py](code/evalkit/agreement.py) · tests: [code/tests/test_py01_agreement.py](code/tests/test_py01_agreement.py)

---

### PY-2 — Clustered bootstrap confidence interval

**Prompt.** Each row is one judged step, and steps belong to traces (clusters). Implement `clustered_bootstrap_ci(values, clusters, n_boot, confidence, seed)` for the row-level mean, and `paired_clustered_bootstrap_diff(a, b, clusters)` for comparing two systems scored on the same rows. Vectorise it: 2,000 resamples of 10k rows should take well under a second.

**Time box.** 35 min.

**Starter code.**

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class BootstrapCI:
    estimate: float; low: float; high: float
    n_clusters: int; n_rows: int; n_boot: int

def clustered_bootstrap_ci(values, clusters, *, n_boot=2000, confidence=0.95, seed=0) -> BootstrapCI:
    # Hint: precompute per-cluster sums and counts, then resample cluster indices
    # with rng.integers(0, G, size=(n_boot, G)).
    raise NotImplementedError

def paired_clustered_bootstrap_diff(values_a, values_b, clusters, **kw) -> BootstrapCI:
    raise NotImplementedError
```

**Edge cases.** A single cluster (the CI is not estimable, so raise). NaN or inf values (make the caller choose a policy). Unequal cluster sizes. Mixed-type cluster ids: casting to `str` merges `1` with `"1"`. Fewer than about 30 clusters, where the percentile interval under-covers; mention BCa or a t-based cluster-robust interval.

**Hidden failure mode.** Two, and both are tested:
1. The **iid bootstrap** over rows produces intervals 2–3× too narrow when the intra-cluster correlation is high. In the test's Monte Carlo (30 clusters × 20 rows, 120 replicates), nominal 95% intervals cover 92% of the time with the cluster bootstrap and 54% with the iid bootstrap.
2. **Ratio of sums vs mean of cluster means.** Resample clusters, but compute the statistic over the pooled rows (Σ sums / Σ counts). Averaging per-cluster means answers a different question, and the two diverge whenever cluster size correlates with quality: long traces fail more.

**Test strategy.** The cluster CI is more than 2× wider than the iid CI on strongly clustered data. A 120-replicate coverage simulation: cluster ≥ 0.85, iid ≤ 0.65. The row-weighting test uses one big bad cluster. The paired CI is under 25% of the unpaired width when items share difficulty. Guards for one cluster and NaN. Same seed gives an identical CI.

**Senior vs Staff.**
- *Senior:* resamples clusters, is vectorised, and uses a paired difference for A/B comparisons.
- *Staff:* names the design effect, \(1 + (m-1)\rho\), and uses it to size datasets *before* collecting them ("40 traces × 25 steps at ρ = 0.3 has a design effect of 1 + 24 × 0.3 = 8.2, so it is worth about 122 independent rows, not 1,000"). Knows when the cluster is the *user* rather than the trace, and that two-way clustering (task × model seed) needs a different scheme. Refuses to report a CI from 5 clusters.

**Reference.** [code/evalkit/bootstrap.py](code/evalkit/bootstrap.py) · tests: [code/tests/test_py02_bootstrap.py](code/tests/test_py02_bootstrap.py)

---

### PY-3 — Bradley-Terry ranking from pairwise judgments

**Prompt.** Given `Comparison(a, b, outcome)` records (1 = a wins, 0 = b wins, 0.5 = tie) from a pairwise judge, fit Bradley-Terry strengths and return a ranking, centred log-strengths, and `win_prob(i, j)`.

**Time box.** 45 min.

**Starter code.**

```python
from dataclasses import dataclass
from collections.abc import Hashable, Iterable

@dataclass(frozen=True)
class Comparison:
    a: Hashable; b: Hashable; outcome: float  # 1, 0 or 0.5

@dataclass(frozen=True)
class BTResult:
    log_strength: dict[Hashable, float]   # centred: mean 0
    iterations: int
    converged: bool
    def ranking(self) -> list[Hashable]: ...
    def win_prob(self, i, j) -> float: ...

def fit_bradley_terry(comparisons: Iterable[Comparison], *, prior_games: float = 1.0,
                      max_iter: int = 200, tol: float = 1e-10,
                      allow_disconnected: bool = False) -> BTResult:
    raise NotImplementedError
```

**Edge cases.**
- An undefeated or winless item: the maximum-likelihood estimate diverges to ±∞. Use a prior (virtual games against a phantom item of strength 1), or refuse.
- A **disconnected comparison graph**: strengths are not comparable across components. Raise; don't rank them silently.
- Ties: half a win each way is a defensible default; Davidson's model is the principled alternative. Dropping ties biases the fit toward decisive pairs.
- Self-comparisons and out-of-range outcomes.
- Orientation invariance: `(a, b, 1)` must mean the same as `(b, a, 0)`.

**Hidden failure mode.** **Convergence along the scale direction.** Hunter's MM updates are the textbook answer, but with a weak prior they converge linearly and very slowly along the common-scale direction. The first draft of the reference solution hit 10,000 iterations without meeting `tol`, even though the ranking was already right. A Newton-Raphson step on the penalised log-likelihood (the Hessian is a weighted graph Laplacian plus the prior diagonal) converges in about 6 iterations. The test asserts `converged`, so "the ranking looks right" is not enough to pass.

**Test strategy.** Recover 5 known strengths from simulated games (±0.2 in log-strength, exact ranking). Orientation invariance. Finite strengths for an undefeated item with a prior, and an error without one. Disconnected graph raises. Ties shrink the gap. Centring, and `win_prob(i,j) + win_prob(j,i) = 1`.

**Senior vs Staff.**
- *Senior:* correct fit, handles separation and disconnection.
- *Staff:* treats the judge as the measurement instrument. Correct position bias before fitting (PY-6 feeds PY-3). Bootstrap the *comparisons clustered by prompt* to get rank intervals, and say "A and B are statistically tied" when the rank CIs overlap. Choose which pairs to compare next by expected information gain, not uniformly. Notes that Elo with a K-factor depends on the order of games while BT does not, which is why leaderboards moved to BT.

**Reference.** [code/evalkit/bradley_terry.py](code/evalkit/bradley_terry.py) · tests: [code/tests/test_py03_bradley_terry.py](code/tests/test_py03_bradley_terry.py)

---

### PY-4 — Trajectory matching: exact, in-order, any-order, with valid alternatives

**Prompt.** Score an agent's tool-call trajectory against references. Support three modes: `exact`, `in_order` (the reference is a subsequence of the actual calls) and `any_order` (a multiset match, extra calls allowed). Accept **several acceptable reference paths** and **step-level alternatives** (`AnyOf(call1, call2)`). Match arguments with an ignore-list (request ids, timestamps), a float tolerance, `ANY` wildcards and predicate matchers. Accept a `forbidden` set of tools. Return `matched`, a partial-credit `score`, the best alternative's index, the missing steps, the forbidden hits, and the first divergence.

**Time box.** 50 min.

**Starter code.**

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)

ANY = object()

@dataclass(frozen=True)
class Step:            # one expected step that accepts any of several calls
    options: tuple[ToolCall, ...]

@dataclass(frozen=True)
class ArgPolicy:
    ignore: frozenset[str] = frozenset()
    float_tol: float = 1e-9
    allow_extra_args: bool = False

@dataclass(frozen=True)
class MatchResult:
    matched: bool; score: float; mode: str; alternative: int | None
    missing: tuple[int, ...] = (); forbidden_hits: tuple[int, ...] = ()
    first_divergence: int | None = None

def match_trajectory(actual: Sequence[ToolCall], references: Sequence[Sequence[ToolCall | Step]], *,
                     mode: Literal["exact", "in_order", "any_order"] = "in_order",
                     arg_policy: ArgPolicy = ArgPolicy(),
                     forbidden: frozenset[str] = frozenset()) -> MatchResult:
    raise NotImplementedError
```

**Edge cases.** An empty reference means "no tool should be called". Under in-order or any-order, it would otherwise match everything vacuously. Duplicate expected calls (two `lookup_order` calls) need multiset semantics. `True` vs `1`. `NaN == NaN` is false. Extra argument keys. Nested dict and list arguments.

**Hidden failure mode.** Two, and both are tested:
1. **Greedy any-order matching is wrong once wildcards or predicates exist.** With expected `[search(q=ANY), search(q="x")]` and actual `[search(q="x"), search(q="y")]`, the greedy matcher assigns the wildcard to `"x"` and then fails. The right answer is a maximum **bipartite matching** (Kuhn's algorithm is enough). Greedy earliest-match *is* optimal for in-order subsequence matching, so the candidate who says "greedy is fine" is right for one mode and wrong for the other.
2. **In-order and any-order ignore extra calls, including destructive ones.** `[lookup, check, issue_refund]` matches the reference `[lookup, check]` in-order. A forbidden-tool list (or a side-effect metric, as in ADK-3) is required.

**Test strategy.** Mode semantics. Partial credit (an LCS-style DP for in-order). Best-of-alternatives. `AnyOf`. The bipartite counterexample. Multiset duplicates. The forbidden loophole. Empty-reference semantics. Argument policy: ignore, tolerance, booleans, extra args. Predicates.

**Senior vs Staff.**
- *Senior:* all three modes are correct, including the bipartite case, with useful diagnostics.
- *Staff:* questions the reference format itself. Alternatives multiply combinatorially, so a partial order (a DAG of required precedences) or a state-based check ("was the refund issued exactly once, after eligibility was confirmed?") scales better. Separates *side-effect correctness* (a hard gate) from *path efficiency* (a soft score). Calibrates the partial-credit score against human judgments before anyone averages it.

**Reference.** [code/evalkit/trajectory.py](code/evalkit/trajectory.py) · tests: [code/tests/test_py04_trajectory.py](code/tests/test_py04_trajectory.py)

---

### PY-5 — Rate-limited async judge runner (retry, jitter, budget, idempotency)

**Prompt.** Build `JudgeRunner.run(requests)` over an async judge function. Requirements: a concurrency cap; a token-bucket rate limit; retry only on retryable errors, with exponential backoff and full jitter, honouring `Retry-After` as a floor; a hard spend budget that concurrent attempts cannot jointly exceed; and idempotency, meaning completed requests are served from a store and concurrent duplicates share one in-flight call. Clock, sleep and RNG must be injectable.

**Time box.** 60 min.

**Starter code.**

```python
import asyncio, hashlib, json, random, time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, MutableMapping

class RetryableJudgeError(Exception):
    def __init__(self, msg="", retry_after: float | None = None): super().__init__(msg); self.retry_after = retry_after
class FatalJudgeError(Exception): ...

@dataclass(frozen=True)
class JudgeRequest:
    item_id: str; payload: dict; judge_model: str; judge_version: str; rubric_version: str
    def idempotency_key(self) -> str: raise NotImplementedError

@dataclass
class JudgeOutcome:
    item_id: str; key: str
    status: Literal["ok", "cached", "failed", "skipped_budget"]
    verdict: Any = None; attempts: int = 0; cost: float = 0.0
    error: str | None = None; backoffs: list[float] = field(default_factory=list)

class JudgeRunner:
    def __init__(self, judge: Callable[[JudgeRequest], Awaitable[tuple[Any, float]]], *, store: MutableMapping,
                 budget_limit: float, estimate_cost: Callable[[JudgeRequest], float], max_concurrency=8,
                 rate_per_sec=10.0, burst=10, max_attempts=4, base_delay=0.5, max_delay=30.0,
                 clock=time.monotonic, sleep=asyncio.sleep, rng: random.Random | None = None): ...
    async def run(self, requests) -> list[JudgeOutcome]: raise NotImplementedError
```

**Edge cases.**
- An idempotency key that omits the judge version or rubric version (see DEBUG-1).
- An `item_id` inside the key, which defeats content-level deduplication.
- Two identical requests in the same batch: single-flight them, don't call twice.
- A failed or timed-out attempt is **still billed**.
- An unknown exception type must not kill the whole `gather`.
- `Retry-After` longer than the backoff cap.
- Budget exhaustion mid-batch: return `skipped_budget` outcomes, don't raise.

**Hidden failure mode.** **Check-then-spend races on the budget.** If you check `spent < limit` before the call and add the cost after it, N concurrent tasks all pass the check and overshoot by up to N × cost. Reserve the estimated cost *before* the attempt and settle the actual cost after, all within one event-loop step. The test runs 12 requests with concurrency 10 against a budget of 5 and asserts exactly 5 `ok` results, with spend ≤ 5 and zero outstanding reservations.

**Test strategy.** A fake clock whose `sleep` advances time instantly, so the tests are deterministic. Idempotency within and across runs. The judge version changes the key and the item id does not. Retry to success with jitter bounds `[0, base·2^(k-1)]`. `Retry-After` as a floor. Fatal and unexpected errors get exactly 1 attempt. The budget test above. A peak-concurrency probe. Token-bucket timing: the burst goes at t=0, then one call every 1/rate, and the last of 10 calls lands at t=4.0 for rate 2 and burst 2.

**Senior vs Staff.**
- *Senior:* all guarantees hold and are tested with a fake clock.
- *Staff:* treats cost as a first-class SLO. Budgets are per tenant and per judge version, and a canary judge gets a capped slice. Retries are budgeted separately, because a retry storm during a provider brownout is when the budget matters most. Adds a circuit breaker that stops sending after an error-rate threshold instead of retrying every item. Makes the store write-once under a content key, so a replay after a crash never re-bills. Knows that "temperature 0" judges are still nondeterministic, so the cache is *also* a consistency mechanism: the same input gets the same verdict within a judge version.

**Reference.** [code/evalkit/judge_runner.py](code/evalkit/judge_runner.py) · tests: [code/tests/test_py05_judge_runner.py](code/tests/test_py05_judge_runner.py)

---

### PY-6 — Judge position-bias probe (swap the answer order)

**Prompt.** Given a pairwise judge `judge(question, first, second) -> "first" | "second" | "tie"`, run every pair in both orders. Map each verdict back to the canonical frame (A or B). Report: the swap-consistency rate; P(judge picks slot 1) over both orders, with a Wilson interval and an exact two-sided binomial test against 0.5; and a debiased verdict per pair (the consistent verdict, otherwise tie).

**Time box.** 35 min.

**Starter code.**

```python
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Sequence

Slot = Literal["first", "second", "tie"]

@dataclass(frozen=True)
class Pair:
    pair_id: str; question: str; a: str; b: str

@dataclass(frozen=True)
class BiasReport:
    n_pairs: int; consistency: float
    first_slot_rate: float; first_slot_ci: tuple[float, float]; p_value: float
    n_decisions: int; probes: tuple

async def probe_position_bias(judge: Callable[[str, str, str], Awaitable[Slot]],
                              pairs: Sequence[Pair]) -> BiasReport:
    raise NotImplementedError
```

**Edge cases.** An unparseable verdict ("Answer 1"): raise, don't coerce. Ties are excluded from the slot-rate denominator. Identical answers should tie. With n = 0 decisions, the Wilson interval is (0, 1).

**Hidden failure mode.** Two:
1. Measuring "P(first wins)" on **one order only** confounds bias with quality whenever the dataset puts the better answer, often the reference, in slot A. A perfectly fair judge then shows 100% "first-slot preference". Only the order-balanced rate is a bias estimate. Across both orders, true quality cancels, so an unbiased judge gives exactly 0.5.
2. **A cache keyed without order.** If the judge sits behind a cache keyed on the unordered pair, the swapped call never reaches the model, and the probe reports perfect consistency. The test builds this cache and shows a judge with 100% first-slot bias on close pairs reported as consistency 1.0.

**Test strategy.** A fair judge gives consistency 1.0, rate 0.5 and p = 1. A biased judge (90% slot-1 preference on close pairs) gives consistency < 0.7, the lower CI bound > 0.5, and p < 1e-6. Debiased verdicts keep clear wins and tie the flips. The single-order confound. The order-blind cache. Checks on the Wilson and binomial helpers.

**Senior vs Staff.**
- *Senior:* order-balanced, correct statistics, debiased verdicts.
- *Staff:* stratifies bias by the human-judged quality margin, because bias concentrates on close pairs, which are the ones that decide launches. Checks length and self-preference bias with the same harness (swap length or model family instead of position). Makes swap-consistency a release gate for every new judge version. Budgets the 2× cost explicitly and proposes swapping only a random 20% slice to *monitor* bias in production.

**Reference.** [code/evalkit/position_bias.py](code/evalkit/position_bias.py) · tests: [code/tests/test_py06_position_bias.py](code/tests/test_py06_position_bias.py)

---

## Google ADK

All three ADK exercises run end to end through `InMemoryRunner` with a `ScriptedLlm`: a `BaseLlm` subclass whose replies come from a Python function of the `LlmRequest`. Build that first (10 minutes). It is the difference between "I think the callback fires" and a test that proves it.

```python
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

class ScriptedLlm(BaseLlm):
    responder: object = None           # Callable[[LlmRequest], LlmResponse]
    requests: list = []
    async def generate_content_async(self, llm_request, stream=False):
        self.requests.append(llm_request)
        yield self.responder(llm_request)

def call(name, **args):
    return LlmResponse(content=types.Content(role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]))
```

### ADK-1 — Multi-judge evaluator agent

**Prompt.** Build an ADK agent that scores an agent trace, supplied in session state as `trace`, with a panel of k LLM judges running in parallel, then aggregates their outputs deterministically. Each judge returns JSON `{"score": 1-5, "rationale": ...}`. The aggregator writes `state["verdict"]` with the median score, the spread, the number of valid judges, per-judge status and judge version, and a status of `ok`, `needs_human` (spread ≥ 2) or `insufficient_quorum`. Oversized or empty traces must not reach any model.

**Time box.** 60 min.

**Starter code.**

```python
from dataclasses import dataclass
from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

@dataclass(frozen=True)
class JudgeSpec:
    name: str; model: object; rubric: str; version: str

class VerdictAggregator(BaseAgent):
    judge_names: list[str]
    min_quorum: int = 2
    disagreement_threshold: int = 2
    async def _run_async_impl(self, ctx: InvocationContext):
        # read ctx.session.state, then:
        # yield Event(author=self.name, invocation_id=ctx.invocation_id,
        #             actions=EventActions(state_delta={"verdict": ...}))
        raise NotImplementedError

def build_multi_judge(judges: list[JudgeSpec], *, min_quorum=2, max_trace_chars=20_000) -> SequentialAgent:
    # SequentialAgent[ParallelAgent[LlmAgent per judge (output_key=...)], VerdictAggregator]
    raise NotImplementedError
```

**Edge cases.** A judge wraps its JSON in a markdown fence. A score of `3.0`, `"3"` or `true`. A judge that abstains. Every judge invalid. Duplicate judge names. A trace containing `{braces}`.

**Hidden failure mode.** Four, and each is tested:
1. **Duplicate `output_key`s under `ParallelAgent`.** The last writer wins and one judge silently disappears. Reject duplicate names at build time.
2. **Coercing parse failures to 0.** One malformed judge drags the mean from 5 to 3.3. An invalid output is an *abstention*, and quorum is checked on valid outputs.
3. **Judges seeing the conversation.** Without `include_contents="none"`, a judge gets the chat history, including user text such as "please grade leniently". With it, the current user turn is still forwarded (verified in the test), so put the trace in the instruction and send the judge only a neutral trigger.
4. **Braces in the rubric template.** In a *string* instruction, every `{identifier}` is a state reference. A rubric that says "output {score}" raises `KeyError: Context variable not found: score` at run time, or silently fills in an unrelated state value if a key with that name exists. (Values substituted from state are not re-templated; this was verified, so `{...}` inside the trace itself is safe.) An `InstructionProvider` (a callable taking `ReadonlyContext`) skips templating entirely, so rubric text with JSON examples is also safe.

Also: don't name state keys `judge:x`. The `app:`, `user:` and `temp:` prefixes change persistence scope, and a near-miss is a code-review hazard.

**Test strategy.** All through `InMemoryRunner`: an agreeing panel gives the median; a fenced JSON reply parses; a malformed judge gives n_valid = 2 rather than a zero; disagreement gives `needs_human`; all invalid gives `insufficient_quorum`; isolation is checked by inspecting each judge's recorded `LlmRequest`; `before_model_callback` short-circuits oversized traces and the test asserts the model was never called; duplicate names are rejected; a parametrised strict-parse table.

**Senior vs Staff.**
- *Senior:* the correct ADK composition, a deterministic aggregator, strict parsing, callbacks used for guards.
- *Staff:* argues that three judges from the same model family are not three independent votes, and measures inter-judge correlation on the calibration set before claiming a variance reduction. Chooses the median over the mean because of robustness to one bad judge, and explains that trade-off. Records judge version and model per verdict for later recalibration. Puts cost control in the callback: skip the expensive judges when the cheap ones agree, which can be built as `LoopAgent` escalation or a routing agent. Plans the migration path, since `ParallelAgent` and `SequentialAgent` are deprecated in 2.x in favour of `Workflow`.

**Reference.** [code/evalkit/adk_multi_judge.py](code/evalkit/adk_multi_judge.py) · tests: [code/tests/test_adk01_multi_judge.py](code/tests/test_adk01_multi_judge.py)

---

### ADK-2 — Safety gate on tool calls

**Prompt.** An ops agent has the tools `read_file(path)`, `issue_refund(order_id, amount_usd)`, `delete_records(table, where)` and `run_shell(cmd)`. Implement a `before_tool_callback` safety gate with these rules. Tools are denied by default. `read_file` is confined to a sandbox root. Refunds must be within (0, limit] and only for orders the user owns (`state["user_order_ids"]`). `delete_records` requires an approval bound to the exact arguments. There is a per-invocation cap on tool calls. The gate fails closed if a validator crashes. Every decision appends a policy-versioned audit record to session state. A blocked call must never execute.

**Time box.** 50 min.

**Starter code.**

```python
from dataclasses import dataclass
from typing import Any, Callable
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

Validator = Callable[[dict[str, Any], dict[str, Any]], str | None]   # (args, state) -> deny reason

@dataclass(frozen=True)
class ToolPolicy:
    validator: Validator | None = None
    requires_approval: bool = False

def make_safety_gate(policies: dict[str, ToolPolicy], *, max_calls: int = 5):
    def gate(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> dict | None:
        # return None to allow; return {"status": "blocked", "reason": ...} to skip the tool
        raise NotImplementedError
    return gate
```

**Edge cases.** `../secret.txt`; an absolute path; **a symlink inside the sandbox that points outside it**; an empty path. `amount_usd` given as NaN, inf, negative, 0, 1e6, `"20"` or `True`. An approval for `delete where id=7` reused for `where 1=1`. The call cap across several tool calls in one invocation.

**Hidden failure mode.** Three, and each is tested:
1. **NaN passes `amount > limit`.** Every comparison with NaN is False. Check `math.isfinite` first.
2. **In-place mutation of session state is silently lost.** `tool_context.state["audit_log"].append(...)` does not register a state delta, and the audit log stays `[]`. Reassign instead: `state["audit_log"] = [*old, rec]`. The test demonstrates the lost write.
3. **Fail-open on exceptions.** If a validator raises inside the callback, the invocation errors or, in a careless wrapper, the call is allowed. Catch the exception and deny.

Also: `temp:tool_calls` gives a per-invocation counter that is visible across tool calls in the same invocation and is not persisted (the test asserts it is absent from the final state). Path checks must use `realpath` and `commonpath`: `startswith(root)` accepts `/ws-evil` for the root `/ws`.

**Test strategy.** Scripted model calls go through the real runner. A `SideEffects` recorder proves blocked tools never ran. The traversal, symlink, absolute-path and empty-path cases are parametrised, as are the numeric edge cases. Ownership. Default deny. Argument-bound approval. The call cap. A crashing validator. The lost in-place write.

**Senior vs Staff.**
- *Senior:* default deny, fail closed, correct numeric and path checks, audit trail.
- *Staff:* treats the gate as a policy engine with a version, a dry-run ("shadow") mode that logs would-block decisions before enforcing, and a denial-rate metric broken down by tool and policy version. Uses `FunctionTool(require_confirmation=...)` or an interrupt for human approval, not a state flag the model might influence. Explains that the model sees the block reason, so the reason must not leak policy internals that help an attacker probe the policy. Evaluates the gate itself: false-block rate on a benign trace set, and miss rate on a red-team set.

**Reference.** [code/evalkit/adk_safety_gate.py](code/evalkit/adk_safety_gate.py) · tests: [code/tests/test_adk02_safety_gate.py](code/tests/test_adk02_safety_gate.py)

---

### ADK-3 — An ADK eval set for trajectory scoring

**Prompt.** For a refund-support agent with the tools `lookup_order`, `check_refund_eligibility` and `issue_refund`, write an ADK eval set (`*.test.json`, EvalSet schema) and a `test_config.json` covering a status query, an eligible refund and an **ineligible** refund. Run it with `AgentEvaluator.evaluate`. Then show that a regressed agent, which refunds ineligible orders but still *says* "not eligible", passes the built-in trajectory and response checks, and add a custom metric that catches it.

**Time box.** 45 min.

**Starter code.**

```jsonc
// evalsets/refunds.test.json (EvalSet schema)
{
  "eval_set_id": "refund_support_trajectories_v1",
  "eval_cases": [{
    "eval_id": "refund_ineligible_must_not_refund",
    "conversation": [{
      "invocation_id": "refund-no-1",
      "user_content": {"role": "user", "parts": [{"text": "Please refund order B2."}]},
      "final_response": {"role": "model", "parts": [{"text": "Order B2 is not eligible ..."}]},
      "intermediate_data": {"tool_uses": [
        {"name": "lookup_order", "args": {"order_id": "B2"}},
        {"name": "check_refund_eligibility", "args": {"order_id": "B2"}}]}
    }],
    "session_input": {"app_name": "refund_support", "user_id": "eval-user", "state": {}}
  }]
}
// evalsets/test_config.json
{"criteria": {"tool_trajectory_avg_score": {"threshold": 1.0, "match_type": "IN_ORDER"},
              "response_match_score": 0.7,
              "no_unexpected_side_effects": 1.0},
 "custom_metrics": {"no_unexpected_side_effects":
              {"code_config": {"name": "evalkit.adk_trajectory_eval.metrics.no_unexpected_side_effects"}}}}
```

```python
# metrics.py: custom metric signature expected by ADK
from google.adk.evaluation.eval_case import Invocation, get_all_tool_calls
from google.adk.evaluation.eval_metrics import EvalMetric, EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult

def no_unexpected_side_effects(eval_metric: EvalMetric, actual_invocations: list[Invocation],
                               expected_invocations: list[Invocation] | None,
                               conversation_scenario=None) -> EvaluationResult:
    raise NotImplementedError

# test
from google.adk.evaluation import AgentEvaluator
await AgentEvaluator.evaluate(agent_module="evalkit.adk_trajectory_eval",
                              eval_dataset_file_path_or_dir=".../evalsets", num_runs=1)
```

**Edge cases.**
- The agent module must expose `agent.root_agent`: the package `__init__.py` needs `from . import agent`.
- ADK sets `eval_metric.threshold = None` before calling a custom metric, so read `eval_metric.criterion.threshold` and set `eval_status` yourself.
- `test_config.json` applies to every `*.test.json` in its folder, so cases that need different match types go in different folders.
- Arguments are compared exactly unless `ignore_args` is set: `"b2"` ≠ `"B2"`, and a timestamp argument makes EXACT always fail.

**Hidden failure mode.** **IN_ORDER and ANY_ORDER tolerate extra calls, including the irreversible one**, and `response_match_score` is ROUGE-1 lexical overlap, so an agent that refunds and then says "not eligible" passes both. The test runs the over-eager agent and asserts that the failure message names *only* `no_unexpected_side_effects`. Conversely, EXACT rejects a harmless extra read (`get_refund_policy`), and teams react by loosening to IN_ORDER globally. The right structure is IN_ORDER for the path plus a hard gate on side effects.

**Test strategy.** The good agent passes the whole folder. The over-eager agent fails only on the custom metric. The eval set parses in the current schema, and `find_config_for_test_file` loads the config. Direct `TrajectoryEvaluator` unit tests cover EXACT vs IN_ORDER on a harmless extra read, IN_ORDER and ANY_ORDER on a harmful extra write, and `ignore_args`.

**Senior vs Staff.**
- *Senior:* a valid eval set and config, a working custom metric, and an understanding of each match type.
- *Staff:* designs the eval set as a product artefact. Every case is tagged with the policy it protects. Negative cases ("must not refund") are at least as numerous as positive ones. `num_runs > 1` with pass^k reporting handles nondeterministic agents. ROUGE is replaced or augmented with a rubric judge (`final_response_match_v2` or a rubric-based criterion) *that is itself calibrated against humans*. The eval set is versioned with the agent, and a new production incident becomes a new case within one release.

**Reference.** [code/evalkit/adk_trajectory_eval/](code/evalkit/adk_trajectory_eval/) (agent, [evalsets/refunds.test.json](code/evalkit/adk_trajectory_eval/evalsets/refunds.test.json), [test_config.json](code/evalkit/adk_trajectory_eval/evalsets/test_config.json), [metrics.py](code/evalkit/adk_trajectory_eval/metrics.py)), regressed agent in [code/evalkit/adk_trajectory_eval_overeager/](code/evalkit/adk_trajectory_eval_overeager/) · tests: [code/tests/test_adk03_evalset.py](code/tests/test_adk03_evalset.py)

---

## LangGraph

### LG-1 — Resumable scoring graph with idempotent publish

**Prompt.** Build a graph that scores a batch of traces with a judge and publishes one summary record. It must fan out one task per item with `Send`; retry transient judge errors with `RetryPolicy`; survive a crash mid-fan-out, so that resuming re-runs **only** the failed items; stay idempotent when re-invoked on the same thread (no re-scoring, no second record); and refuse to publish an incomplete run.

**Time box.** 60 min.

**Starter code.**

```python
import operator
from typing import Annotated, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

class TransientJudgeError(Exception): ...

def merge_first_write_wins(left: dict | None, right: dict | None) -> dict:
    raise NotImplementedError

class ScoringState(TypedDict, total=False):
    run_id: str
    judge_version: str
    items: list[dict[str, Any]]                                   # {"item_id", "trace"}
    scores: Annotated[dict[str, dict], merge_first_write_wins]
    summary: dict[str, Any]
    log: Annotated[list[str], operator.add]

class IdempotentSink:
    def upsert(self, key: str, record: dict) -> bool: ...          # False if key exists

def build_scoring_graph(judge, sink, *, checkpointer=None, max_attempts=3):
    g = StateGraph(ScoringState)
    # plan -> (Send per unscored item | "aggregate") ; score_item -> aggregate -> publish
    # g.add_node("score_item", score_item, retry_policy=RetryPolicy(max_attempts=..., retry_on=TransientJudgeError))
    # g.add_conditional_edges("plan", fan_out, ["score_item", "aggregate"])
    return g.compile(checkpointer=checkpointer)
```

**Edge cases.** An empty batch: the fan-out function must return `"aggregate"`, not `[]`. Duplicate `item_id`s: a keyed merge would drop one silently, so raise. Retries exhausted. A new judge version: it gets a new thread and a new record, and never overwrites the old one.

**Hidden failure mode.**
- **`operator.add` on scores.** It works on the first run and then duplicates on every retry, replay or re-invocation. A keyed-merge reducer makes the writes idempotent. Choosing *first-write-wins* over last-write-wins is deliberate: a published score must not change underneath its digest.
- **Fatal vs transient errors.** Putting `retry_on=Exception` on the node retries a 400 three times and bills three times.
- **Publish must be keyed by content** (`run_id:judge_version:digest`), because "check whether published, then publish" is a race.

**Test strategy.**
- **Kill and resume.** A fatal error on `i3` makes `invoke` raise. `get_state(cfg).next == ("score_item",)`: only the failed `Send` task is pending, because LangGraph saved the successful tasks' writes. `invoke(None, cfg)` resumes, and the per-item call counts are `{i0:1, i1:1, i2:1, i3:2, i4:1}`.
- Re-invoke on the same thread: `sink.attempts == 2` with 1 record and `publish_noop` logged.
- A transient failure retried to success (3 calls); retries exhausted raises and nothing is published.
- The empty batch, duplicate ids, and reducer properties.

**Senior vs Staff.**
- *Senior:* correct Send fan-out, keyed reducer, retry policy, a checkpointed resume that is demonstrated by a test.
- *Staff:* explains the pending-writes semantics that make partial resume work, and what breaks them (side effects inside `score_item` that aren't idempotent, or changing the graph shape between crash and resume). Chooses `thread_id = run_id`, never the dataset name (see DEBUG-2). Swaps `InMemorySaver` for a durable checkpointer in production, and asks about checkpoint size when the state holds full traces: store references, not payloads. Adds a completeness SLO (percentage scored per run) and alerts on `missing`, rather than letting the average of a partial run reach a dashboard.

**Reference.** [code/evalkit/lg_resumable_scoring.py](code/evalkit/lg_resumable_scoring.py) · tests: [code/tests/test_lg01_resumable_scoring.py](code/tests/test_lg01_resumable_scoring.py)

---

### LG-2 — Adaptive judge routing with escalation to a human

**Prompt.** Route each trace through a cheap judge. If it is confident, finalize. If not, resample up to `max_cheap_rounds` times, then escalate to a strong judge. If cheap and strong agree within 1 point and the strong judge is confident, finalize. Otherwise, **interrupt for human review**; the human's label decides the result. High-risk items always go to a human. Invalid human input is re-asked. The strong judge retries transient errors. A routing bug must end in `GraphRecursionError`, not an unbounded bill.

**Time box.** 50 min.

**Starter code.**

```python
import operator
from typing import Annotated, Any, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, RetryPolicy, interrupt

class RoutingState(TypedDict, total=False):
    item: dict[str, Any]            # {"item_id", "trace", "risk"}
    cheap: dict | None; cheap_rounds: int
    strong: dict | None; human: dict | None; final: dict | None
    route: Annotated[list[str], operator.add]

def build_routing_graph(cheap_judge, strong_judge, *, checkpointer=None,
                        confident=0.85, max_cheap_rounds: int | None = 2, agree_within=1):
    def human_review(state) -> Command[Literal["finalize", "human_review"]]:
        answer = interrupt({"item_id": state["item"]["item_id"], "cheap": state["cheap"], "strong": state["strong"]})
        ...  # validate; Command(goto="finalize", update={"human": answer}) or re-ask
    g = StateGraph(RoutingState)
    # g.add_node("human_review", human_review, destinations=("finalize", "human_review"))
    # g.add_conditional_edges("cheap_judge", after_cheap, ["finalize", "cheap_judge", "strong_judge"])
    return g.compile(checkpointer=checkpointer)

# usage
# out = graph.invoke({"item": ...}, {"configurable": {"thread_id": "t1"}})
# out["__interrupt__"][0].value -> payload for the reviewer
# graph.invoke(Command(resume={"score": 2, "reviewer": "r-17"}), {"configurable": {"thread_id": "t1"}})
```

**Edge cases.** A resume value of `{"score": "five"}` or `True`. Resuming on the wrong `thread_id`, which gets a fresh state. Risk overrides confidence. `max_cheap_rounds=None`, the unbounded loop that `recursion_limit` has to catch.

**Hidden failure mode.** **`interrupt()` re-executes the whole node on resume.** Anything above the `interrupt` call (a ticket, a Slack ping, a billed LLM call) runs twice. Keep review nodes pure, and put side effects in an upstream node that is checkpointed once it completes. DEBUG-2 tests this directly. Also: `interrupt` without a checkpointer cannot resume, and a confidence threshold not calibrated against human labels sends either everything or nothing to humans. Say how you would set τ from a reliability diagram.

**Test strategy.** The confident path skips the strong judge. Resample-then-escalate gives the exact route `["cheap","cheap","strong","final:strong"]`. The disagreement path returns `__interrupt__` with the payload, `get_state().next == ("human_review",)`, and after the resume the source is human. An invalid label re-interrupts. High risk always interrupts. Threads are isolated. `recursion_limit=8` with an unbounded loop raises `GraphRecursionError` after exactly 8 cheap calls. Strong-judge transient retries.

**Senior vs Staff.**
- *Senior:* correct conditional edges, `Command` routing, interrupt and resume, bounded loops, tests for each route.
- *Staff:* frames the router as a cost/quality policy with measurable operating points: the escalation rate, human load per 1k items, and agreement with humans *per route*. The cheap judge's accuracy on auto-finalized items is the number that matters, and it is estimated by auditing a random slice of confident items. Without that slice, selection bias hides the error rate. Adds reviewer-disagreement handling (two reviewers on a sample, adjudication) and treats human labels as data for recalibrating the judge.

**Reference.** [code/evalkit/lg_adaptive_routing.py](code/evalkit/lg_adaptive_routing.py) · tests: [code/tests/test_lg02_adaptive_routing.py](code/tests/test_lg02_adaptive_routing.py)

---

## Debug under pressure

Format: 25 minutes. The interviewer pastes the code and a symptom report. You find the root causes, name the observability gap that let each one ship, and propose a *safe* fix: code change, data repair and rollout. Each test file first **reproduces** every planted bug in the shipped code, then checks the fix.

### DEBUG-1 — Judge scoring service

**Symptom report.** "After the judge-v2 rollout, agreement with humans did not move at all. The scores table has more rows than items. The calibration report shows kappa 0.91, but production disagreement looks much worse."

```python
_CACHE: dict[str, dict] = {}

def cache_key(item: dict) -> str:
    return hashlib.md5((item["prompt"] + item["response"]).encode()).hexdigest()

def score_items(items, judge, writer, judge_version, max_retries=3):
    for item in items:
        k = cache_key(item)
        if k in _CACHE:
            verdict = _CACHE[k]
        else:
            verdict = None
            for attempt in range(max_retries):
                try:
                    verdict = judge(item, judge_version)
                    writer.insert({"item_id": item["item_id"], "score": verdict["score"],
                                   "judge_version": judge_version})
                    break
                except TimeoutError:
                    continue
            _CACHE[k] = verdict

def split_for_calibration(rows, holdout_frac=0.3, seed=0):
    random.seed(seed)
    random.shuffle(rows)
    cut = int(len(rows) * (1 - holdout_frac))
    return rows[:cut], rows[cut:]
```

**Time box.** 25 min.

**Root causes.**

| # | Bug | Consequence | Test |
|---|---|---|---|
| 1 | The cache key omits the judge model, judge version and rubric. | v2 is served v1 verdicts, so the rollout was a no-op, and every "v2 agreement" number is v1's. | `test_bug1_cache_key_ignores_judge_version` |
| 1b | The key concatenates `prompt + response`. | `("ab","c")` collides with `("a","bc")`, so different items share a verdict. | `test_bug1b_concatenation_collision` |
| 2 | `writer.insert` sits inside the judge retry loop. | A write whose commit succeeded but whose ack timed out re-runs the **judge** (billed twice, possibly a different verdict) and inserts a duplicate row. | `test_bug2_write_timeout_reruns_judge_and_duplicates_rows` |
| 2b | Exhausted retries cache `None`. | The item is poisoned: it is never re-judged and never written. | `test_bug2b_exhausted_retries_silently_cache_none` |
| 3 | The split is row-level, while steps of the same trace are near-duplicates. | More than 20 of 40 traces land on both sides, so the holdout kappa is inflated (0.91). `random.seed` also mutates global RNG state, and `shuffle` reorders the caller's list. | `test_bug3_row_split_leaks_traces_and_mutates_input` |

**Observability gap.**
- No cache hit-rate metric **by judge version**. A new version at a 100% hit rate on day one is an alarm.
- No uniqueness constraint on `(item_id, judge_version)`, and no rows-per-item gauge.
- No retry or attempt counter split by operation (judge vs write).
- No overlap assertion between the calibration and holdout groups.
- No "calibration kappa vs production audit kappa" comparison. A gap that large is itself the signal of leakage.

**Safe fix** ([judge_service_fixed.py](code/evalkit/debug/judge_service_fixed.py)).
1. Key = sha256 of canonical JSON over `{prompt, response, judge fingerprint}`, where the fingerprint hashes model, version, rubric and temperature.
2. Retry the judge and the write *separately*. The write is an upsert on `(item_id, judge_fingerprint)`. Retry exhaustion raises and caches nothing.
3. Group split by `trace_id` with a salted hash bucket. The assignment is a pure function of the id, stable as data grows (tested), with an overlap assertion.

**Rollout.** Purge the cache entries written since the v2 flag flipped, or bump the cache namespace. Deduplicate existing rows by keeping the earliest per `(item_id, judge_version)`, and write a data-repair ticket for downstream aggregates. Re-run the v2 evaluation. Recompute calibration on the group split and publish the corrected kappa with a note on why it dropped.

**Senior vs Staff.**
- *Senior:* finds bugs 1–3 and fixes them.
- *Staff:* also finds 1b and 2b. Quantifies the blast radius: which dashboards and decisions used v2 numbers. Adds the metrics that would have caught each bug. Turns the whole class of problem into a design rule: "every cache key and every idempotency key is derived from one `JudgeConfig.fingerprint`."

**Reference.** buggy: [code/evalkit/debug/judge_service_buggy.py](code/evalkit/debug/judge_service_buggy.py) · fixed: [code/evalkit/debug/judge_service_fixed.py](code/evalkit/debug/judge_service_fixed.py) · tests: [code/tests/test_debug01_judge_service.py](code/tests/test_debug01_judge_service.py)

---

### DEBUG-2 — LangGraph rescoring and human-review graph

**Symptom report.** "The nightly mean score drifts upward run over run on an unchanged dataset. The dashboard shows more than twice as many scores as items. Reviewers get two tickets for every escalation."

```python
class State(TypedDict, total=False):
    items: list[dict]
    scores: Annotated[list[dict], operator.add]
    summary: dict

def build(judge, tickets, checkpointer):
    def score(state):
        return {"scores": [{"item_id": it["item_id"], **judge(it)} for it in state["items"]]}

    def rescore_low_confidence(state):
        redo = [s for s in state["scores"] if s["confidence"] < 0.5]
        fresh = [{"item_id": s["item_id"], **judge({"item_id": s["item_id"]}), "rescored": True} for s in redo]
        return {"scores": state["scores"] + fresh}

    def review(state):
        worst = min(state["scores"], key=lambda s: s["score"])
        tickets.append({"item_id": worst["item_id"]})
        decision = interrupt({"item_id": worst["item_id"]})
        return {"summary": {"reviewed": worst["item_id"], "decision": decision}}

    def summarize(state):
        vals = [s["score"] for s in state["scores"]]
        return {"summary": {**state.get("summary", {}), "n": len(vals), "mean": sum(vals) / len(vals)}}
    # START -> score -> rescore -> review -> summarize -> END, compiled with checkpointer

def run_config(dataset: str) -> dict:
    return {"configurable": {"thread_id": dataset}}
```

**Time box.** 25 min.

**Root causes.**

| # | Bug | Consequence | Test |
|---|---|---|---|
| 1 | **Reducer misuse.** The channel reducer is `operator.add`, and `rescore` returns `state["scores"] + fresh`, the *full* list plus the delta. | The reducer appends the whole list to itself: 3 items become 3 + (3 + 1) = **7** rows. Both the stale and the rescored entry for item b are in the mean. | `test_bug_reducer_duplicates_whole_list` |
| 2 | **Side effect before `interrupt()`.** | On `Command(resume=...)`, the `review` node re-executes from its first line, so a second ticket is created. | `test_bug_side_effect_before_interrupt_runs_twice` |
| 3 | **`thread_id = dataset`.** | Tonight's run resumes last night's thread. `operator.add` then accumulates both nights' scores, which produces the "drift" on unchanged data. | `test_bug_thread_id_reuse_accumulates_across_runs` |

**Observability gap.**
- No invariant `len(scores) == len(items)` at summarize time.
- No `run_id` in state or in the ticket payload, so duplicate tickets can't be correlated.
- No idempotency key on the ticket API.
- No metric for "checkpoint resumed from a prior run" (a thread with existing state at the start of a nightly job).
- The dashboard plots the mean without n. Had it shown n, the 7-rows-for-3-items inflation would have been visible on day one.

**Safe fix** ([lg_review_fixed.py](code/evalkit/debug/lg_review_fixed.py)).
1. `scores: Annotated[dict[str, dict], latest_by_item]`, and nodes return **only deltas**.
2. Move ticket creation into an `open_ticket` node before `review`, keyed `run_id:item_id`. The `review` node contains only `interrupt`.
3. `thread_id = f"{dataset}:{run_id}"`.
4. Assert the score/item cardinality in `summarize`.

**Rollout.** Close the duplicate tickets. Recompute historical nightly means from the raw judge logs, not from the checkpointed state. Delete or archive the old dataset-named threads so that no job resumes them.

**Senior vs Staff.**
- *Senior:* finds all three and explains the re-execution semantics of `interrupt`.
- *Staff:* generalises them into review rules: every accumulating reducer needs a written reason; every node before or containing an `interrupt` must be idempotent; thread ids come from run ids. Adds a graph-level test harness that runs each graph twice on the same thread and asserts nothing changes.

**Reference.** buggy: [code/evalkit/debug/lg_review_buggy.py](code/evalkit/debug/lg_review_buggy.py) · fixed: [code/evalkit/debug/lg_review_fixed.py](code/evalkit/debug/lg_review_fixed.py) · tests: [code/tests/test_debug02_lg_review.py](code/tests/test_debug02_lg_review.py)

---

## Suggested sequencing

| Session | Exercises | Why this order |
|---|---|---|
| 1 | PY-1, PY-2 | The statistics underpin every later claim ("the judge improved"). |
| 2 | PY-4, PY-6 | Trajectory semantics and judge bias. PY-6 feeds PY-3. |
| 3 | PY-3, PY-5 | Ranking, then the runner that produces the comparisons at scale. |
| 4 | ADK-1, ADK-2 | ADK composition, callbacks and state. Build `ScriptedLlm` first. |
| 5 | ADK-3 | Eval sets and custom metrics, reusing PY-4's lessons. |
| 6 | LG-1, LG-2 | Durability, fan-out, interrupts. |
| 7 | DEBUG-1, DEBUG-2 | Timed and cold, ideally with someone else pasting the code. |
