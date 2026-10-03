# Staff/Senior Applied Scientist & Research Scientist Interview Preparation Kit

## Scope and Interview Context

This kit is tuned for roles building and owning:

- Agent-evaluation platforms
- Judge-based scoring pipelines
- Reliability/safety controls
- Production LLM systems with strict SLAs

The bar here is **staff/senior**: architecture judgment, rigor under ambiguity, operational accountability, and mathematically defensible decisions.

---

## 1) Interview Traps & Conceptual Drifts

Format per item: **Interviewer question -> naive answer -> senior/staff analysis + solution**.

### Trap 1: “How do you know your LLM judge is accurate?”
- **Naive answer:** “We benchmarked it on a labeled set and got 90% agreement with humans.”
- **Senior/staff analysis + solution:**
  - Agreement alone is not enough; you need **calibration**, **bias decomposition**, and **failure stratification**.
  - Require:
    1. Inter-rater reliability (Krippendorff’s alpha / Cohen’s kappa where appropriate)
    2. Per-slice error (domain, language, length, safety category)
    3. Judge-vs-human confusion matrix and threshold tuning
    4. Drift monitoring over time
  - Production answer: “I treat the judge as a probabilistic sensor and continuously re-calibrate it against a rotating gold set with slice-level control charts.”

### Trap 2: “Can we use one universal quality score?”
- **Naive answer:** “Yes, aggregate all criteria into a weighted score.”
- **Senior/staff analysis + solution:**
  - Single scores hide trade-offs and can reward unsafe behavior that boosts helpfulness.
  - Use **vector metrics** (helpfulness, factuality, policy compliance, latency, cost), with **hard safety gates** before composite ranking.
  - Enforce lexicographic ordering: safety constraints -> task success -> quality -> cost/latency.

### Trap 3: “If offline eval improves, are we safe to ship?”
- **Naive answer:** “Yes, offline metrics are up, so launch.”
- **Senior/staff analysis + solution:**
  - Offline gains often fail online due to distribution shift and user interaction loops.
  - Require online guardrail rollout:
    - Shadow + canary + staged ramp
    - Exposure guardrails per cohort
    - Rollback SLO triggers
  - Use pre-registered launch criteria and veto rules.

### Trap 4: “Prompting is enough; why build infrastructure?”
- **Naive answer:** “Prompt improvements can solve most issues quickly.”
- **Senior/staff analysis + solution:**
  - Prompting without infrastructure is brittle and non-reproducible.
  - Need: versioned prompts, datasets, model snapshot IDs, trace IDs, and deterministic replay envelope.
  - Staff-level stance: solve root causes with system controls, not repeated manual prompt patching.

### Trap 5: “How do you reduce hallucination?”
- **Naive answer:** “Just add RAG.”
- **Senior/staff analysis + solution:**
  - RAG introduces retrieval failure modes: stale docs, wrong chunking, irrelevant context flooding, citation laundering.
  - Solve with:
    - Retrieval quality metrics (recall@k, MRR, provenance coverage)
    - Attribution checks (claim -> citation span alignment)
    - Abstention policy when evidence confidence is low

### Trap 6: “What is your definition of reliability?”
- **Naive answer:** “High uptime and low latency.”
- **Senior/staff analysis + solution:**
  - Reliability in LLM systems is multidimensional:
    - Infra reliability (availability, tail latency)
    - Behavioral reliability (consistent outputs under semantically equivalent inputs)
    - Safety reliability (policy compliance under adversarial prompts)
  - Define SLOs for all three classes.

### Trap 7: “Can we trust synthetic eval data?”
- **Naive answer:** “Yes, synthetic data scales labeling cheaply.”
- **Senior/staff analysis + solution:**
  - Synthetic data amplifies generator biases and can leak benchmark style.
  - Use synthetic only with:
    - Human-audited anchor sets
    - Style-diversification transforms
    - Contamination checks against train/eval pools
    - Weighting to avoid overfitting to synthetic artifacts

### Trap 8: “Why not optimize only for cost?”
- **Naive answer:** “Lower model size and token caps reduce spend.”
- **Senior/staff analysis + solution:**
  - Cost-only optimization can increase retries, user abandonment, and incident load.
  - Optimize expected utility:
    - `Utility = task_value - inference_cost - failure_penalty - safety_penalty`
  - Include second-order costs (ops escalation, trust loss).

### Trap 9: “Can we evaluate agents by final answer only?”
- **Naive answer:** “Yes, users care about final output.”
- **Senior/staff analysis + solution:**
  - Agent systems can produce correct outputs via unsafe or non-compliant trajectories.
  - Evaluate trajectory-level properties:
    - Tool misuse
    - Policy bypass attempts
    - Unnecessary external calls
    - Retry-loop pathology

### Trap 10: “Why not hardcode rules instead of model judges?”
- **Naive answer:** “Rules are deterministic and simpler.”
- **Senior/staff analysis + solution:**
  - Rules are useful for hard constraints but fail on semantic nuance.
  - Hybrid architecture: deterministic policy filters + calibrated model judge + human escalation band.

### Trap 11: “Do we need uncertainty estimates?”
- **Naive answer:** “No, top-1 score is enough for ranking.”
- **Senior/staff analysis + solution:**
  - Uncertainty is critical for thresholding and safe abstention.
  - Use score intervals (bootstrap), confidence calibration, and disagreement-based escalation.

### Trap 12: “Can A/B test wins settle model quality?”
- **Naive answer:** “If click-through improves, model is better.”
- **Senior/staff analysis + solution:**
  - Single KPI wins can hide safety regressions or long-term user trust damage.
  - Use balanced scorecards and guardrail metrics with hard fail criteria.

---

## 2) Architectural/System Design Traps

For each: **Challenge -> hidden catch -> ideal solution**.

### A. Build an evaluation platform for 50 teams
- **Challenge:** Central platform for dataset management, runs, scoring, and dashboards.
- **Hidden catch:** Team-specific schemas drift rapidly; strict central schema causes adoption failure.
- **Ideal solution:**
  - Canonical core schema + extensible typed metadata fields
  - Dataset contract validation layer
  - Immutable run artifacts with lineage graph (dataset hash, prompt version, model ID, scorer version)
  - Multi-tenant quotas and RBAC

### B. Judge-based scoring at scale (100M examples/month)
- **Challenge:** Accurate, low-cost scoring.
- **Hidden catch:** Judge variance and API outages create noisy, non-reproducible results.
- **Ideal solution:**
  - Tiered judging:
    1. Deterministic validators (format/policy)
    2. Fast judge model for broad coverage
    3. High-accuracy judge/human for disagreement or high-impact samples
  - Caching and dedup by normalized prompt+response hash
  - Idempotent task queue with retry budgets and dead-letter queues

### C. Reliability-first online evaluation
- **Challenge:** Evaluate changes continuously in production.
- **Hidden catch:** Online traffic skew gives false confidence.
- **Ideal solution:**
  - Stratified sampling with protected cohorts
  - Counterfactual logging for offline replay
  - Sequential testing with pre-registered stopping rules

### D. Safety monitoring for agent tool use
- **Challenge:** Catch harmful tool trajectories in near real-time.
- **Hidden catch:** Post-hoc checks miss fast failure cascades.
- **Ideal solution:**
  - Inline policy gateway before tool execution
  - Stateful risk scorer over trajectory events
  - Circuit breaker that downgrades capability when risk exceeds threshold

### E. “Single metric leaderboard” for model selection
- **Challenge:** Executive asks for one ranking table.
- **Hidden catch:** Teams game the metric; regressions slip through.
- **Ideal solution:**
  - Publish Pareto front instead of single rank
  - Require threshold pass in safety and factuality before ranking on utility/cost
  - Keep anti-gaming test slices hidden and rotated

### F. Reproducibility under rapid model updates
- **Challenge:** Vendor model aliases silently update.
- **Hidden catch:** “Same experiment” becomes non-repeatable.
- **Ideal solution:**
  - Pin snapshot/version IDs whenever provider supports it
  - Archive prompts, tools, retrieval index version, and runtime config
  - Build replay harness that can re-run with frozen dependencies

### G. Multi-region low-latency serving
- **Challenge:** Serve globally with strict p95 latency.
- **Hidden catch:** Safety pipelines add synchronous latency and cross-region chatter.
- **Ideal solution:**
  - Regionalized inference and retrieval
  - Split synchronous hard checks vs asynchronous deep audits
  - Graceful degradation tiers (full policy -> minimal safe mode -> abstain)

### H. Human-in-the-loop adjudication
- **Challenge:** Add humans only where needed.
- **Hidden catch:** Reviewer load explodes without calibrated routing.
- **Ideal solution:**
  - Route only uncertainty/disagreement band
  - Active learning to maximize information gain per reviewed item
  - Reviewer quality scoring and consensus protocols

---

## 3) Arithmetic Back-of-Envelope Drills

Use engineering approximations; show logic fast and clearly.

### Drill 1: Monthly judge inference cost
**Problem:** 40M responses/month. Average 900 tokens judged per response (prompt+completion). Judge model costs $0.60 per 1M tokens. What is monthly cost?

**Math:**
1. Total tokens = `40,000,000 * 900 = 36,000,000,000` tokens = 36B
2. Cost per token = `$0.60 / 1,000,000`
3. Total cost = `36,000 * $0.60` (because 36B / 1M = 36,000) = `$21,600`

**Final:** **~$21.6k/month**

---

### Drill 2: p95 latency budget decomposition
**Problem:** End-to-end p95 must be 1800 ms. Network p95=250 ms, retrieval p95=350 ms, safety checks p95=300 ms. What is max remaining model inference p95 budget?

**Math:**
1. Fixed components = `250 + 350 + 300 = 900 ms`
2. Remaining budget = `1800 - 900 = 900 ms`

**Final:** **Model inference p95 must be <=900 ms**

---

### Drill 3: Canary blast radius
**Problem:** 12M daily requests. Canary at 2% for 3 hours. Estimate exposed requests.

**Math:**
1. Requests/hour = `12,000,000 / 24 = 500,000`
2. For 3 hours total = `1,500,000`
3. Canary exposure = `1,500,000 * 0.02 = 30,000`

**Final:** **~30k canary requests**

---

### Drill 4: Human review staffing
**Problem:** 5M eval items/day. Escalation rate 0.4%. Each reviewer handles 240 items/day. How many reviewers?

**Math:**
1. Escalated/day = `5,000,000 * 0.004 = 20,000`
2. Reviewers needed = `20,000 / 240 = 83.33`
3. Add 20% operational buffer -> `~100`

**Final:** **~84 minimum, ~100 with buffer**

---

### Drill 5: Retry policy impact
**Problem:** Base failure rate 2%. One retry succeeds 70% of failed requests. New effective failure rate?

**Math:**
1. Initial failures = 2%
2. Recovered by retry = `2% * 70% = 1.4%`
3. Residual failure = `2% - 1.4% = 0.6%`

**Final:** **Failure rate drops from 2.0% to 0.6%**

---

### Drill 6: Confidence interval quick check
**Problem:** Model A wins 5,300 out of 10,000 pairwise comps (53%). Is this likely real?

**Approximation:**
1. For Bernoulli proportion `p=0.53`, `n=10,000`
2. Standard error `SE ≈ sqrt(p*(1-p)/n) ≈ sqrt(0.53*0.47/10000) ≈ sqrt(0.2491/10000) ≈ sqrt(0.00002491) ≈ 0.00499`
3. 95% margin `≈ 1.96*SE ≈ 0.0098` (~0.98%)
4. CI ≈ `53% +/- 0.98%` => `[52.0%, 54.0%]`

**Final:** **Likely significant above 50%**

---

### Drill 7: Throughput sizing
**Problem:** Need to process 80M evals/day. Worker handles 12 evals/sec sustained. Required workers at 70% utilization?

**Math:**
1. Daily seconds = 86,400
2. Per-worker/day raw = `12 * 86,400 = 1,036,800 evals`
3. At 70% util effective = `1,036,800 * 0.7 ≈ 725,760`
4. Workers = `80,000,000 / 725,760 ≈ 110.2`
5. Round up + redundancy -> 130

**Final:** **~111 minimum, ~130 practical**

---

### Drill 8: Token truncation tradeoff
**Problem:** Average context currently 2,000 tokens. Truncating to 1,400 reduces quality pass rate from 91.0% to 90.2%. Monthly volume 25M calls, token price $2.00 per 1M tokens. Is truncation worth it (token cost only)?

**Math:**
1. Token reduction/call = 600
2. Monthly tokens saved = `25,000,000 * 600 = 15,000,000,000` = 15B
3. Cost saved = `(15B / 1M) * $2 = 15,000 * 2 = $30,000/month`
4. Quality loss = `0.8 percentage points`
5. Failed additional calls = `25,000,000 * 0.008 = 200,000`
6. Break-even penalty per additional fail = `$30,000 / 200,000 = $0.15`

**Final:** If each extra failure costs **>$0.15** in business impact, truncation is net negative.

---

## 4) Coding Exercises (Python, Google ADK, LangGraph)

Each exercise includes APIs/primitives, edge cases, hidden failure modes, tests, and what separates senior/staff solutions.

## 4.1 Python Exercise 1 — Judge Calibration Service

### Prompt
Implement a Python service that ingests `(prediction_score, human_label)` streams and emits calibrated probabilities plus drift alarms.

### Core primitives
- `pandas`/`numpy` for batch stats
- Calibration models (Platt scaling / isotonic regression)
- Drift detectors (PSI, KS test, rolling z-score)
- Async ingestion queue (`asyncio.Queue` or Kafka consumer abstraction)

### Required capabilities
1. Sliding-window calibration by slice
2. Confidence interval reporting
3. Alert when calibration error exceeds threshold
4. Idempotent handling of duplicate events

### Edge cases
- Severe class imbalance
- Missing or delayed labels
- Distribution regime change mid-window
- Duplicate event IDs with conflicting values

### Hidden failure modes
- Leakage: calibration fitted on future labels
- Silent metric inflation from dropping hard samples
- Window instability from tiny slice sizes

### Test strategy
- Unit tests for calibration math
- Property tests for monotonicity in isotonic mapping
- Replay tests with synthetic drift episodes
- Chaos test: delayed labels + out-of-order delivery

### Senior/staff differentiators
- Explicit data contracts + schema evolution plan
- Online/offline metric parity checks
- Costed operational runbook (SLOs, alert thresholds, rollback)

---

## 4.2 Python Exercise 2 — Reliable Eval Orchestrator

### Prompt
Build an orchestrator that executes eval tasks with retries, deadlines, and exactly-once result publication.

### Core primitives
- Task queue abstraction
- Retry with jittered exponential backoff
- Idempotency keys
- Dead-letter queue
- Structured tracing (OpenTelemetry IDs)

### Edge cases
- Partial downstream timeout after side effects
- Worker restart mid-task
- Clock skew affecting deadline checks

### Hidden failure modes
- Duplicate publishes after timeout ambiguity
- Retry storms during dependency incident
- Poison-pill tasks causing head-of-line blocking

### Test strategy
- Deterministic simulation with fault injection
- Contract tests for idempotent publish endpoint
- Load tests validating queue depth under incident

### Senior/staff differentiators
- Global retry budget and circuit breakers
- Clear consistency semantics (at-least-once compute, exactly-once publish)
- Incident containment design

---

## 4.3 Google ADK Exercise 1 — Multi-judge ADK Agent

### Prompt
Implement an ADK-based evaluator agent that:
1. Runs deterministic policy checks
2. Calls two model judges
3. Resolves disagreement with escalation logic

### ADK primitives to know (version-dependent naming)
- Agent definition and instruction routing
- Tool wrappers for policy checkers and judge calls
- Session/context state
- Middleware/hooks for logging and guardrails

### Edge cases
- Tool exception inside judge stage
- Judge timeout with partial outputs
- Adversarial prompt trying to override rubric

### Hidden failure modes
- Rubric prompt injection via user content interpolation
- Non-deterministic tool ordering causing irreproducible scores
- Unbounded retries from nested tool calls

### Test strategy
- Tool-level deterministic tests with fixed fixtures
- Adversarial prompt corpus tests
- Golden-trace replay test ensuring identical state transitions

### Senior/staff differentiators
- Strict separation between user content and system rubric channels
- Deterministic execution envelope with trace replay
- Cost-aware fallback path and graceful degradation

---

## 4.4 Google ADK Exercise 2 — Safety-Gated Agentic Tool Runner

### Prompt
Create an ADK agent that can call external tools (search, code-exec mock) but must pass a safety gate before every tool invocation.

### ADK/API focus
- Pre-tool interception hook
- Typed tool schemas and argument validation
- Policy decision object (allow / deny / redact / escalate)

### Edge cases
- Malformed tool JSON from model
- Long-running tool without heartbeat
- Tool output containing prompt-injection payloads

### Hidden failure modes
- TOCTOU gap (policy checks input but transformed payload sent)
- Data exfiltration through tool arguments
- Safety bypass through chain-of-thought leakage in tool inputs

### Test strategy
- Differential tests across policy versions
- Mutation tests on tool payload fields
- Red-team scenarios for exfiltration attempts

### Senior/staff differentiators
- End-to-end threat model and mitigations
- Policy explainability logs for postmortems
- Defense-in-depth: schema + policy + runtime anomaly checks

---

## 4.5 LangGraph Exercise 1 — Stateful Eval Workflow Graph

### Prompt
Build a LangGraph workflow with nodes:
`ingest -> normalize -> score -> aggregate -> publish`,
with checkpointing and resumability.

### LangGraph primitives
- `StateGraph` with typed state
- Node and edge definitions (conditional edges)
- Checkpointer for durable state
- Interrupt/resume controls for human-in-the-loop steps

### Edge cases
- Resume after crash between score and publish
- State schema migration after deployment
- Conditional branch loops

### Hidden failure modes
- Non-idempotent publish node re-executed on resume
- Checkpoint corruption or incompatible serializer version
- “Stuck” graph due to unmet branch condition

### Test strategy
- Replay from checkpoints under fault injection
- Schema migration compatibility tests
- Liveness tests (graph always reaches terminal state or explicit fail state)

### Senior/staff differentiators
- Formal node contracts (input/output invariants)
- Versioned state migration plan
- Operational observability by node-level SLIs

---

## 4.6 LangGraph Exercise 2 — Adaptive Evaluation with Escalation

### Prompt
Create a graph that adaptively routes examples:
- Easy examples: cheap deterministic checks
- Medium: one LLM judge
- Hard/disagreement: second judge + human queue

### LangGraph/API focus
- Confidence-based routing node
- Parallel branch execution + join
- Budget-aware control state (`remaining_cost`, `deadline_ms`)

### Edge cases
- Confidence score unavailable
- Budget exhausted mid-branch
- Human queue unavailable

### Hidden failure modes
- Branch fan-out explosion from malformed routing policy
- Join node deadlock when one branch silently fails
- Hidden starvation of hard examples

### Test strategy
- Monte Carlo simulation for routing policy
- Budget exhaustion scenario tests
- Queue outage drills with fallback behavior assertions

### Senior/staff differentiators
- Joint optimization of quality, latency, cost
- Explicit fairness constraints across slices
- Governance controls to prevent metric gaming

---

### Practical interview note for ADK/LangGraph coding rounds
If API names differ by release, state your assumptions explicitly, pin versions, and focus on **execution semantics** (state, retries, idempotency, guardrails). Senior candidates are graded on systems correctness, not memorizing exact import paths.

---

## 5) High-Stakes 2–4 Week Roadmap (Integrated)

Two variants: **2-week crash plan** and **4-week full plan**.

## A. 2-Week Crash Plan (high intensity)

### Week 1 — Core rigor foundation
1. **Day 1-2:** Concept drills
   - Master traps section; rehearse staff-level responses aloud.
   - Deliverable: 20-question oral bank with structured answers.
2. **Day 3-4:** Architecture reps
   - Solve 4 design traps on whiteboard with trade-off matrices.
   - Deliverable: one-page architecture template for interview use.
3. **Day 5-7:** Math speed training
   - Complete 30 back-of-envelope problems under timer.
   - Deliverable: personal cheat sheet of canonical formulas.

### Week 2 — Execution + mock pressure
1. **Day 8-10:** Coding track
   - Implement at least 1 Python + 1 LangGraph + 1 ADK exercise.
   - Include tests and failure-injection cases.
2. **Day 11-12:** Reliability/safety deep dive
   - Prepare incident scenario responses (rollback, blast-radius control, comms).
3. **Day 13-14:** Full-loop mocks
   - 3 end-to-end mock interviews:
     - System design
     - Coding
     - Research/evaluation methodology
   - Postmortem each with rubric-driven gap closure.

Exit criterion: can defend architecture and metrics under adversarial questioning without hand-waving.

---

## B. 4-Week Full Plan (preferred for staff/stakes)

### Week 1: Conceptual correctness
- Interview trap mastery and literature refresh:
  - Calibration
  - Causal pitfalls in online eval
  - Safety taxonomies
- Output: polished answer bank with failure-case examples.

### Week 2: Design and production judgment
- Daily architecture prompts with explicit non-functional constraints:
  - SLOs, RBAC, reproducibility, disaster recovery
- Output: reusable design skeleton + trade-off catalog.

### Week 3: Implementation + reliability engineering
- Build full mini eval platform slice:
  - Dataset ingestion
  - Multi-judge scoring
  - Escalation queue
  - Dashboard metrics
- Add fault injection, retries, idempotency, and observability.
- Output: runnable artifact + test report.

### Week 4: Interview simulation and refinement
- 5 high-pressure mock sessions
- Strict scoring:
  - Technical depth
  - Clarity
  - Correctness
  - Operational realism
- Iterate weak areas with targeted drills.

Exit criterion: performance is consistent across conceptual, coding, and system design rounds at staff calibration.

---

## Senior/Staff Rubric You Should Self-Enforce

1. **No hand-waving on uncertainty**: always discuss confidence, calibration, and drift.
2. **No single-metric thinking**: show multi-objective optimization with hard safety constraints.
3. **Production realism**: retries, idempotency, observability, rollback, and incident response are mandatory.
4. **Reproducibility mindset**: version every artifact needed for replay.
5. **Adversarial robustness**: explicitly design for abuse, not just happy paths.

If your answer would still work after an oncall incident at 2 AM, it is likely senior/staff-quality.
