# 06 — Methodology and Behavioural (v2)

Three parts: human-evaluation methodology, capability benchmark design, and statistical power with worked numbers, followed by behavioural preparation. All numbers were recomputed in Python with \(z_{0.975} = 1.960\) and \(z_{0.80} = 0.842\).

---

## Part 1 — Human-evaluation methodology

### 1.1 Annotation guidelines

A guideline is a measurement instrument, so version it, test it, and change it like code.

- **Construct definition first.** State what is being measured, for example "task success: the user's stated goal is achieved in the end state with no policy violation." Keep separate constructs (correctness, helpfulness, safety, tone) in separate questions. Composite "overall quality" ratings mix constructs and make disagreement impossible to diagnose.
- **Decision procedure, not adjectives.** Write each item as ordered yes/no checks, for example "Did the agent take any irreversible action without confirmation? If yes → Fail regardless of outcome." Adjective scales such as "mostly helpful" produce annotator-specific thresholds.
- **Anchored scales.** Each scale point has a written definition and 2–3 real examples, including *boundary* examples (a clear 3 vs. a clear 4). Prefer binary or 3-point scales unless finer resolution has been shown to add reliable signal. A 7-point scale with low reliability is worse than a reliable 3-point scale.
- **Explicit "cannot judge" option** (missing context, requires domain expertise), tracked separately. Otherwise annotators guess and add noise.
- **Rationale field** for Fail and edge cases. This feeds adjudication and guideline revision.
- **Pilot loop.** Three to five annotators label 50–100 items. Compute agreement per item. Review the lowest-agreement items: each is either a guideline gap (fix the text), an ambiguous item (drop it or mark it), or an annotator error (retrain). Repeat until agreement plateaus. Publish a changelog for every revision, and keep labels from different versions apart.
- **Blinding.** Hide model identity, position-randomise pairwise items, and don't show judge pre-labels when the labels will be used to validate the judge.

### 1.2 Choosing an inter-annotator agreement statistic

| Situation | Statistic | Notes |
|---|---|---|
| 2 fixed raters, nominal | Cohen's κ | Depends on prevalence (traps doc, Trap 6). Report per-class agreement too. |
| 2 raters, ordinal | Weighted κ (quadratic) | Quadratic-weighted κ ≈ ICC for consistency. |
| >2 raters, each item rated by the same number of raters | Fleiss' κ | Raters need not be the same people. |
| Any number of raters, missing ratings, any scale type | **Krippendorff's α** | The default for production pipelines with variable overlap. Choose a distance metric (nominal, ordinal, interval). |
| Continuous or averaged scores | ICC | Choose the form: ICC(2,1) absolute agreement vs. ICC(3,1) consistency; ICC(·,k) for mean of k raters. |
| Heavily skewed prevalence | Gwet's AC1, PABAK, plus per-class agreement | More stable across prevalence, but can hide minority-class disagreement. |

**Staff rule:** choose the statistic by the *decision the labels support*. If labels are used to measure the recall of a failure detector, the relevant reliability is agreement *on failures*, so report negative-class agreement and the confusion matrix, not a single chance-corrected number.

Rough interpretation bands for κ or α ("≥ 0.8 good, 0.67–0.8 tentative") are conventions, not laws. The actionable comparison is judge-human agreement against human-human agreement on the same items (traps doc, Trap 5).

### 1.3 Adjudication

- **Redundancy policy.** Label a random 10–20% of items twice (to estimate reliability) and 100% of high-stakes items (safety, launch-gate sets).
- **Disagreement routing.** A disagreement goes to a senior adjudicator who sees both labels and rationales and records a final label plus a *disagreement cause*: guideline gap, item ambiguity, annotator error, or legitimate subjectivity.
- **Legitimate subjectivity is signal, not error.** For subjective constructs (tone, preference), keep the label distribution, for example as a soft label or the proportion of raters preferring A. Don't force a gold label. Evaluate the judge against the distribution, for example by comparing its probability with the rater proportion, or by accuracy only on high-consensus items.
- **Don't let adjudication contaminate reliability estimates.** Inter-annotator agreement is computed on *pre-adjudication* independent labels. Adjudicated labels are the reference for judge validation.
- **Close the loop.** Recurring disagreement causes feed guideline revisions, and adjudicated boundary cases become new anchor examples.

### 1.4 Annotator drift

- **Gold items** (5–10%, blind, expert-adjudicated) inserted continuously. Track per-annotator gold accuracy with EWMA or CUSUM control charts.
- **Behavioural telemetry.** Time per item (drops signal speeding), label distribution per annotator (drift toward the majority class), and position preference in pairwise tasks.
- **Anchor-set re-labelling.** Every quarter, re-label a fixed set with the current pool. A change against the historical labels means pool-level drift or a changed interpretation of the guideline.
- **Remediation.** Pause, retrain, recertify with a qualification test, and re-review recent work from flagged annotators. Recalibrate the whole pool after guideline changes.
- See System Design, Scenario 9 for the full pipeline version.

---

## Part 2 — Benchmark design by capability

Design checklist that applies to every capability:
- **Construct validity.** Does success require the capability, or is there a shortcut?
- **Scoring validity.** Is the scorer validated against humans, including valid alternative solutions?
- **Contamination resistance.** Private, procedurally generated, or refreshed items.
- **Headroom.** Current best model well below ceiling, with no saturation.
- **Difficulty stratification.** Report by difficulty tier, not just the aggregate.
- **Reliability.** Report pass^k (traps doc, Trap 11) alongside pass^1.
- **Cost.** Tokens and latency per task reported beside the score.

| Capability | Construct and item design | Scoring | Characteristic shortcut or threat | Staff-level control |
|---|---|---|---|---|
| **Planning** | Multi-step goals with dependencies, resource constraints, and replanning triggers (a tool fails mid-plan, new information invalidates a step). Vary plan length and branching factor. | End-state goal satisfaction plus constraint violations. Plan validity checked by a symbolic verifier where the domain allows (PDDL-style, scheduling constraints). | Memorised plans for canonical domains (Blocksworld). The model succeeds by pattern-matching familiar surface forms. | Obfuscated domains (renamed objects and actions); procedurally generated instances; performance vs. plan length curves; replanning-success rate after injected perturbations. |
| **Reasoning** | Problems with verifiable answers (math, logic, code) plus problems where the reasoning must be faithful (the conclusion depends on the stated steps). | Exact or verifiable answer. Process checks via step verification on a subset. | Answer-only scoring rewards lucky guesses and memorised answers. Multiple-choice allows elimination heuristics. | Numeric or free-form answers; perturbation variants (changed numbers, irrelevant distractor clauses, as in GSM-Symbolic); consistency across paraphrases; contamination checks (traps doc, Trap 14). |
| **Memory** (cross-session, long-horizon) | Facts introduced in session 1, queried in session N. Updates and contradictions ("I moved to Berlin"). Deletion requests ("forget my address"). Interference from similar entities. | Accuracy of recall, correct use of the *latest* value, correct abstention when information was never given or was deleted. | Tests that only probe recall of stable facts miss staleness and privacy failures. | Separate metrics for recall, update-correctness, abstention and forgetting-compliance; vary delay and number of distractor sessions; include a privacy check for deleted data. |
| **Tool use** | Realistic APIs with schemas, errors, pagination, auth, rate limits and ambiguous user requests needing clarification. Include tasks where the correct action is *not* to call a tool. | End-state diff plus policy invariants (τ-bench style), argument validity, unnecessary-call rate. | Path matching against a gold sequence penalises valid alternatives (traps doc, Trap 12). Mocked tools that never error overstate robustness. | Stateful sandbox (System Design, Scenario 8); fault injection (timeouts, 500s, malformed responses); measure recovery rate; clarification-request precision and recall. |
| **Retrieval** (RAG, search agents) | Queries with known relevant documents, including multi-hop, no-answer-in-corpus, and conflicting-source cases. Corpus includes distractors that are lexically similar but wrong. | Separate retrieval (recall@k, nDCG) from generation (answer correctness, citation precision, and faithfulness: every claim supported by a cited passage). | End-to-end answer accuracy can't tell whether retrieval or generation failed. The model may answer from parametric memory and ignore retrieval. | Counterfactual corpora (edited facts that contradict world knowledge) to test grounding vs. parametric memory; no-answer items to measure abstention; attribution checks by NLI or a judge validated against humans. |
| **Long-context** | Information at controlled positions and depths across lengths (8k → 1M). Tasks needing aggregation or multi-hop across distant passages, not just single-needle lookup. | Accuracy as a function of (length, position, number of needles, distractor similarity). | Single-needle retrieval ("needle in a haystack") saturates and doesn't reflect real use. Lexical overlap between question and needle makes it a string search. | Needles with no lexical overlap with the question (paraphrased); multi-needle aggregation and reasoning (RULER-style); report the effective context length at which accuracy drops below a threshold; include realistic agent traces as haystacks. |

**Agent-evaluation composition.** For end-to-end agent benchmarks, add a **capability attribution** layer. Tag each failure with its first-error category (planning, tool call, retrieval, memory, reasoning; traps doc, Trap 13) so an aggregate score becomes an actionable breakdown.

---

## Part 3 — Statistical power and sample-size planning

Power analysis happens *before* collecting labels or running evaluations. It turns "how many do we need?" into a budget line and prevents underpowered launches.

### 3.1 Estimating a single proportion to a margin of error

\[n = \frac{z_{1-\alpha/2}^2\, p(1-p)}{E^2}\]

- Worst case p = 0.5, E = ±3 pp: \(n = 3.8415 \times 0.25 / 0.0009 = 1067.1 \Rightarrow\) **1,068**.
- Judge-human agreement expected around 0.85, target E = ±2 pp: \(n = 3.8415 \times 0.1275/0.0004 = 1224.5 \Rightarrow\) **1,225** labelled items.

### 3.2 Comparing two agents on independent samples

Detect 70% → 73% (δ = 3 pp), two-sided α = 0.05, power 0.80:
\[n_{\text{per arm}} = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2\,[p_1(1-p_1) + p_2(1-p_2)]}{\delta^2} = \frac{7.849 \times (0.2100 + 0.1971)}{0.0009} = 3550.3 \Rightarrow \mathbf{3{,}551}\]
That is 7,102 task runs in total.

### 3.3 Same comparison, paired (both agents on the same tasks)

Only *discordant* tasks carry information (McNemar). Suppose the discordant rate is \(p_d = p_{10} + p_{01} = 0.12\) (e.g. \(p_{10} = 0.075\), \(p_{01} = 0.045\), so δ = 0.03):
\[n = \frac{\left(z_{1-\alpha/2}\sqrt{p_d} + z_{1-\beta}\sqrt{p_d - \delta^2}\right)^2}{\delta^2} = \frac{(1.960 \times 0.3464 + 0.842 \times 0.3451)^2}{0.0009} = 1044.2 \Rightarrow \mathbf{1{,}045} \text{ tasks}\]
Pairing cuts the required tasks from 3,551 per arm to 1,045 shared tasks, 3.4× fewer runs (2,090 vs. 7,102) and 6.8× fewer distinct tasks to author (1,045 vs. 7,102). That is why the default design runs both agents on the same task set. The discordant rate \(p_d\) should be estimated from a pilot or from historical comparisons of similar agents.

### 3.4 Minimum detectable effect under a fixed budget

With 500 paired tasks and \(p_d = 0.12\):
\[\text{MDE} \approx (z_{1-\alpha/2} + z_{1-\beta})\sqrt{p_d/n} = 2.802 \times \sqrt{0.12/500} = 0.0434\]
About **4.3 pp**. If the expected improvement is 2 pp, this evaluation cannot answer the question. Say so before running it, not after.

### 3.5 Repeated runs per task (clustering)

Design effect \(\text{DEFF} = 1 + (m-1)\rho\). With m = 5 runs per task and ICC ρ = 0.3, DEFF = 2.2. To match the effective n of 3,551 independent trials from 3.2, you need \(3{,}551 \times 2.2 = 7{,}812\) trials, which is **1,563 tasks** × 5 runs.

- One run per task: 3,551 tasks and 3,551 runs.
- Five runs per task: 1,563 tasks and 7,815 runs.

Extra runs substitute for tasks at a poor exchange rate when ρ is high. Use multiple runs when task authoring is the bottleneck, or when you need per-task reliability (pass^k).

### 3.6 Rare events and safety bounds

To claim a violation rate below 0.1% at 95% confidence after observing zero violations: \(n \ge \ln(0.05)/\ln(0.999) = 2994.2 \Rightarrow\) **2,995** independent trials. The rule of three gives 3,000. This is valid only for the tested distribution (traps doc, Trap 21).

### 3.7 Planning checklist

1. Define the primary metric, the unit of analysis (task, not trial), and the decision rule, including non-inferiority margins for guardrails.
2. Choose α (lower for irreversible or safety decisions), power, and the smallest effect worth detecting. The effect size comes from business value, not from what the budget allows.
3. Use a paired design where possible. Estimate \(p_d\) or ICC from a pilot.
4. Account for multiplicity (traps doc, Trap 9) and for planned interim looks (Trap 8).
5. If the required n exceeds the budget, reduce variance (pairing, stratification, CUPED-style covariate adjustment using pre-period scores), accept a larger MDE explicitly, or don't run the evaluation.

---

## Part 4 — Behavioural preparation

### 4.1 Research-to-shipped story template

Prepare 2–3 stories. Each should take 3–4 minutes told aloud and survive 15 minutes of follow-up questions.

| Beat | What to say | What the interviewer listens for |
|---|---|---|
| **Context (20 s)** | Product, users, the metric that mattered, and why the status quo was failing, with a number. | Business grounding, not a research-only framing. |
| **Problem framing** | How you turned a vague ask ("evaluation is slow and untrusted") into a measurable problem ("judge-human agreement on tool-use traces is 0.61 α vs. 0.78 human-human; launch gates take 5 days"). | Ability to define the right problem. |
| **Key insight or decision** | The non-obvious technical call and the alternatives you rejected, with reasons. Example: "Switched from path-matching to end-state diffs because 23% of human-valid trajectories were scored as failures." | Technical judgement and trade-offs. |
| **Validation** | How you knew it worked: an offline validation against humans, the online A/B, and the metric that tied to business impact. Include a result that surprised you. | Rigour, including evaluating your own evaluator. |
| **Shipping** | What it took to get into production: stakeholders, infrastructure, rollout plan, guardrails, what you cut. | Execution and influence beyond your own code. |
| **Impact (quantified)** | Before and after on the business metric, adoption (teams, launches gated), and cost. | Scale of impact appropriate to Staff level. |
| **Reflection** | What you'd do differently, and what the organisation now does because of this work (standards, platforms, hires). | Self-awareness and multiplier effect. |

Senior vs. Staff difference: a Senior story ends at "I built and shipped X." A Staff story shows that you **changed how other teams decide**, for example by making the metric the launch standard, setting the evaluation contract, or stopping a flawed launch across organisational lines.

### 4.2 Explaining evaluation results to non-technical stakeholders

**Structure: decision → evidence → confidence → risk → ask.**

1. **Lead with the decision.** "We recommend launching to 10% of traffic, not 100%."
2. **One headline number in business units.** "Resolves about 4 more customer issues per 100," not "pass rate +4.1 pp, κ = 0.72."
3. **Uncertainty as a range with a plain meaning.** "Our best estimate is +4, and we're confident it's between +1 and +7. We're confident it's not worse."
4. **Where it doesn't hold.** "It's worse for refund requests, about 3 in 100 more handed to a human. That's why we recommend 10% with refunds excluded."
5. **How we know the measurement is trustworthy,** in one sentence. "Our automatic grader agrees with expert reviewers as often as two experts agree with each other."
6. **The ask and the next checkpoint.** "Approve 10%. We'll report the live resolution rate in two weeks, with a predefined rollback trigger."

Avoid: jargon without translation, false precision ("73.24%"), hiding bad slices, and presenting the judge score as ground truth.

Practise translating:
- *Confidence interval* → "the range we're confident about."
- *Non-significant* → "we can't tell it apart from no change with this much data."
- *Judge agreement* → "how often our automatic grader matches expert reviewers."

### 4.3 Disagreeing with a PM about a launch

**Scenario prompt.** "Your PM wants to launch Friday. The aggregate evaluation is positive, but you found a regression on a safety-relevant slice and the judge hasn't been validated on the new tool-use traffic. What do you do?"

**Strong answer arc:**
1. **Understand their constraint first.** Ask what's driving Friday: a customer commitment, a competitor, a quarterly goal. The goal is a decision that serves the real constraint, not winning the argument.
2. **Separate facts from judgement.** Present the evidence neutrally: "Aggregate +3 pp, CI [+1, +5]. Refund slice −4 pp, CI [−7, −1], outside our non-inferiority margin. Judge validity on tool-use traffic is unmeasured: 40% of the new traffic, zero human labels."
3. **Quantify the risk in their terms.** "At launch volume, that's roughly N additional wrong refunds per day, with an estimated cost of $X and a policy exposure of Y."
4. **Offer options, not a veto:**
   - (a) Launch Friday with refunds routed to the current agent, plus a staged 10% rollout with automatic rollback triggers.
   - (b) Launch Friday at 5% while 500 tool-use traces are human-labelled over the weekend (the budget is sized: ±3 pp needs about 1,068).
   - (c) Delay one week for a fix.
5. **Agree on a decision rule in advance.** "If the live refund escalation rate exceeds Z in the first 48 hours, we roll back automatically." This turns the disagreement into a shared experiment.
6. **Escalate cleanly if needed.** If the risk is safety or compliance and the PM still wants the full launch, escalate *together*, as a joint write-up of both positions to the decision owner, not around them.
7. **Disagree and commit.** Once the decision owner decides, support execution fully, and set up the monitoring that would catch the downside you predicted.

**Failure modes interviewers screen for:** absolutism ("evaluation says no, full stop"), capitulation ("it's the PM's call"), arguing from authority instead of evidence, and failing to propose a path that meets the business deadline.

### 4.4 Rapid-fire behavioural prompts to prepare

- A time your evaluation metric was wrong. How did you find out, and what did you change?
- A time you killed or delayed your own project based on evaluation results.
- How you set an evaluation standard adopted by teams you didn't manage.
- A disagreement with another scientist about methodology, and how it was resolved (with data, not seniority).
- The most expensive evaluation mistake you've seen, and what process now prevents it.
