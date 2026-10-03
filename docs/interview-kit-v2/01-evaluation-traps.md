# 01 — Evaluation Traps (v2)

Twenty-one traps, grouped by theme. Each has three parts: **Scenario** (how the interviewer frames it), **Mid-level answer** (plausible but incomplete), and **Staff analysis** (what the strong answer covers). All worked numbers were recomputed in Python; the peeking inflation figures were checked by Monte Carlo simulation (200k runs).

Notation: \(p_o\) observed agreement, \(p_e\) chance agreement, \(\alpha\) significance level, \(z_{0.975}=1.960\), \(z_{0.80}=0.842\), ICC intra-class correlation \(\rho\).

---

## A. Judge validity

### Trap 1 — Position bias in pairwise judging

**Scenario.** "Our pairwise judge says the new agent wins 58% of head-to-heads against production. We always put the candidate in slot A. Ship?"

**Mid-level answer.** "58% is well above 50%, and with n = 2,000 the CI excludes 50%. Ship."

**Staff analysis.**
- Because the candidate always sits in slot A, the treatment is confounded with position. Many judges favour the first slot, some the last, and the direction can flip with prompt length. The 8-point margin may be mostly position.
- **Measure it.** Run every pair in both orders. Define the *consistency rate* as \(P(\text{verdict}(A,B) \text{ and verdict}(B,A) \text{ pick the same underlying response})\). Define *position bias* as \(P(\text{slot 1 wins}) - 0.5\) over the order-balanced set.
- **Correct it.** Score each pair twice. Count a win only if the response wins in both orders, and record a split as a tie. Report win rate, tie rate and consistency rate together, never win rate alone.
- **Diagnose.** If consistency is below about 80%, the judge cannot resolve differences at this quality level. Tighten the rubric, switch to pointwise rubric scoring with pairwise tie-breaking, or use a stronger judge.
- **Staff-level addition.** Position bias grows as the two responses get closer in quality: the judge falls back on heuristics when the signal is weak. So bias measured on easy pairs underestimates bias on the pairs that decide the launch. Measure consistency *stratified by human-judged margin*.

### Trap 2 — Verbosity (length) bias

**Scenario.** "After we added 'be thorough' to the system prompt, judge scores rose 6 points. Humans say the answers got worse."

**Mid-level answer.** "Tell the judge in its rubric not to reward length."

**Staff analysis.**
- Rubric instructions reduce length bias but rarely remove it. Treat length as a measured confounder instead.
- **Quantify it.** Fit a regression on paired data:
  \(\operatorname{logit} P(A \succ B) = (\theta_A - \theta_B) + \gamma \cdot \tanh(\Delta\text{len}/\sigma) + \epsilon\).
  A significant \(\gamma > 0\) is length bias. The *length-controlled win rate* is the prediction with \(\gamma\) set to 0, the same idea as AlpacaEval 2's LC metric.
- **Validate against humans.** Build length-matched pairs (length ratio within ±10%) and length-mismatched pairs with human labels. Compare judge-human agreement across the two strata. A gap means the judge tracks length, not quality.
- **Guard the loop.** Any prompt change that raises mean length by more than a set threshold triggers a mandatory human spot check (see Trap 17).
- **Nuance.** Sometimes longer really is better, for example completeness on multi-part questions. The aim is not to penalise length. It is to make sure the judge's sensitivity to length matches the humans' sensitivity.

### Trap 3 — Self-preference bias

**Scenario.** "We use Model X as the judge. We're comparing agents built on X, Y and Z. X-based agents lead the leaderboard."

**Mid-level answer.** "X is the strongest model, so it's probably right."

**Staff analysis.**
- Judges assign higher scores to text with low perplexity *under the judge*, and a model's own outputs have the lowest perplexity. That is a systematic advantage, not noise.
- **Measure it.** On a human-labelled pairwise set, compute the judge-human disagreement rate conditioned on which family the human-preferred response came from:
  \(\text{SPB} = P(\text{judge picks own-family} \mid \text{human picks other}) - P(\text{judge picks other} \mid \text{human picks own-family})\).
  If SPB is significantly above zero, the judge is biased toward its own family.
- **Mitigate.** Use a judge from a family that is not in the comparison. Or use a panel of judges from several families and take the majority or mean, then check that the panel's SPB is near zero. Paraphrasing both responses with a neutral model before judging normalises style, at the cost of losing style as a quality signal.
- **Staff-level addition.** The same mechanism affects *fine-tuned* judges trained on outputs from one family, and agents distilled from the judge model. Keep a lineage record for each judge and agent.

### Trap 4 — "Temperature 0 means the judge is deterministic"

**Scenario.** "We run the judge at temperature 0, so we score each trace once. Re-running last week's eval gave a 1.4-point different aggregate. Is something broken?"

**Mid-level answer.** "Probably a bug in our pipeline. Temperature 0 is greedy decoding, so it's deterministic."

**Staff analysis.**
- Greedy decoding is deterministic only if the logits are bitwise identical. In production they are not:
  - Floating-point addition is non-associative, and GPU kernels change reduction order with batch size and composition. Your request's logits depend on who else is in the batch.
  - Mixture-of-experts routing with capacity limits can send tokens to different experts depending on the batch.
  - Providers change hardware, kernels, quantisation and serving stacks behind the same model name (see System Design, Scenario 6).
  - Once one token differs near a tie, everything after it diverges. That is enough to flip a verdict.
- **Treat the judge as a stochastic measurement instrument.**
  - Measure test-retest reliability: score N traces R times and report per-item flip rate and aggregate SD across runs.
  - For high-stakes verdicts, sample k times and take a majority vote, or use the mean of token-probability scores.
  - Cache verdicts with a full version key so reruns are reproducible by construction rather than by hoping for determinism.
- **Interpretation.** A 1.4-point shift needs to be compared with the measured run-to-run SD. If the SD is 0.8, a 1.4-point shift is within about 2σ of noise, and deltas below that were never resolvable in the first place.

### Trap 5 — The human-label noise ceiling

**Scenario.** "Our judge agrees with human labels 82% of the time. Leadership wants 95%. How do we get there?"

**Mid-level answer.** "Better prompts, a bigger judge model, few-shot examples."

**Staff analysis.**
- Measured agreement is bounded by the noise in the reference labels. With binary latent truth, a judge error rate \(e_j\) and a human error rate \(e_h\), independent errors, observed agreement is
  \[A_{obs} = (1-e_j)(1-e_h) + e_j e_h.\]
  If \(e_h = 0.10\), a *perfect* judge scores \(A_{obs} = 0.90\). A judge that is exactly as good as one human (\(e_j = 0.10\)) scores \(0.81 + 0.01 = 0.82\). So 95% is unreachable against these labels, and 82% may already be human-level.
- **Right benchmark.** Compare judge-human agreement with *human-human* agreement on the same items. If judge-human ≈ human-human, the judge is interchangeable with one annotator.
- **For continuous scores,** use the attenuation formula \(r_{obs} = r_{true}\sqrt{\text{rel}_x \cdot \text{rel}_y}\). To raise the ceiling, average k annotators. Spearman-Brown gives \(\text{rel}_k = \frac{k\,r}{1+(k-1)r}\). With single-rater reliability 0.60, three raters give \(\text{rel}_3 = 1.8/2.2 = 0.818\), so even a perfectly reliable judge can correlate at most \(\sqrt{0.818} = 0.905\) with the 3-rater mean.
- **Correlated errors break the formula.** If judge and humans share a bias, for example both reward confident tone, observed agreement *overstates* validity. Keep an adversarial slice where the known-correct answer goes against the surface heuristics.
- **Staff move.** Reframe the goal from "95% agreement" to "judge-human agreement ≥ human-human agreement on every priority slice, with no slice more than X points below."

---

## B. Statistics

### Trap 6 — The kappa paradox under class imbalance

**Scenario.** "Judge and human agree on 95% of pass/fail verdicts, but Cohen's κ is 0.26. Which do we believe?"

**Mid-level answer.** "κ corrects for chance, so the judge is bad." Or: "95% is great, so κ is misleading. Ignore it."

**Staff analysis.** Neither statistic alone is enough. Use this worked table (n = 1,000):

| | Human pass | Human fail |
|---|---|---|
| **Judge pass** | 940 | 25 |
| **Judge fail** | 25 | 10 |

- \(p_o = 0.95\). Both marginals are 0.965 pass, so \(p_e = 0.965^2 + 0.035^2 = 0.93245\).
- \(\kappa = (0.95 - 0.93245)/(1 - 0.93245) = 0.260\).
- Why κ is low: at 96.5% prevalence, \(p_e\) is huge and κ's denominator \(1 - p_e = 0.0676\) is tiny. This is the Feinstein-Cicchetti paradox: κ depends on prevalence.
- **Decompose into per-class agreement:**
  - Positive agreement \(= 2a/(2a+b+c) = 1880/1930 = 0.974\).
  - Negative agreement \(= 2d/(2d+b+c) = 20/70 = 0.286\).
- **The real finding:** the judge and the human almost never agree on *which items fail*. Of 35 judge-fails and 35 human-fails, only 10 overlap. If the product purpose is catching failures, this judge is nearly useless, and 95% agreement hides that.
- **Prevalence-robust alternatives:** PABAK \(= 2p_o - 1 = 0.90\). Gwet's AC1 uses \(p_e = 2\bar\pi(1-\bar\pi) = 0.0676\), giving AC1 \(= 0.946\). These are more stable across prevalence, but *they also hide the failure-class problem*.
- **Staff answer:**
  - Report the confusion matrix, per-class agreement, and precision and recall on the minority class.
  - Choose the headline metric by decision use. For a failure detector, report recall and precision on failures with CIs.
  - Enrich the evaluation set with failures by stratified sampling, then reweight to production prevalence when reporting population metrics.

### Trap 7 — Simpson's paradox across slices

**Scenario.** "Agent B beats Agent A on easy tasks and on hard tasks, but A wins overall. Dashboard bug?"

**Mid-level answer.** "Must be a bug. If B wins every slice, B wins overall."

**Staff analysis.** It is not a bug. It is a mix effect.

| | A: n | A: pass | B: n | B: pass |
|---|---|---|---|---|
| Easy | 800 | 720 (90%) | 200 | 190 (95%) |
| Hard | 200 | 60 (30%) | 800 | 280 (35%) |
| **Total** | 1000 | **78%** | 1000 | **47%** |

- B wins both slices by 5 points but loses overall by 31, because the router sent B mostly hard tasks. Difficulty is a confounder: it drives both assignment and outcome.
- **Fix: standardise to a common mix.** Using A's mix (0.8 easy, 0.2 hard), B's standardised pass rate is \(0.8 \cdot 0.95 + 0.2 \cdot 0.35 = 0.83\), compared with A's 0.78.
- **Prevent it:**
  - Randomise assignment within strata, or use the *same task set* for both arms (paired design; see methodology doc).
  - Report stratified results with an explicit target mix.
  - Alert when slice mix differs between arms (chi-square on mix).
- **Staff-level addition.** The reverse also happens: aggregate improvement while every priority slice regresses. The launch criterion should be "no priority slice regresses by more than its non-inferiority margin" *and* aggregate improves, not aggregate alone.

### Trap 8 — Early stopping ("peeking")

**Scenario.** "We check the online A/B dashboard every day and stop as soon as p < 0.05. Usually significant within a week."

**Mid-level answer.** "That's fine. p < 0.05 means a 5% false-positive rate."

**Staff analysis.**
- Every look is another chance to cross the threshold under the null. With equally spaced looks at a fixed-sample z-test at nominal α = 0.05, the actual type-I error is about **14% at 5 looks, 19% at 10, and 25% at 20** (Armitage et al.; reproduced by simulation). "Usually significant within a week" is what you would expect even when nothing changed.
- **Fixes:**
  - **Fixed horizon.** Pre-compute n from a power analysis (see methodology doc) and look once.
  - **Group-sequential design.** Pre-register K looks with an alpha-spending function. Classical O'Brien-Fleming boundaries for 5 looks at two-sided α = 0.05 are \(|z| >\) 4.56, 3.23, 2.63, 2.28, 2.04. They are very strict early and close to 1.96 at the end, so little power is lost compared with a fixed-horizon test.
  - **Always-valid inference.** Use mSPRT or confidence sequences, which allow continuous monitoring with guaranteed type-I control. The cost is wider intervals at any fixed n.
- **Safety exception.** Guardrail metrics such as safety violations or error rate *should* be monitored continuously with sequential tests, and you stop for harm. Stopping-for-harm and stopping-for-benefit need different boundaries.

### Trap 9 — Multiple comparisons across slices and metrics

**Scenario.** "The new judge prompt significantly improved agreement on 3 of 20 task categories. We'll write it up as 'targeted improvements'."

**Mid-level answer.** "Each test is at α = 0.05, so each finding is 95% reliable."

**Staff analysis.**
- With 20 independent tests at α = 0.05, the family-wise error rate is \(P(\ge 1 \text{ false positive}) = 1 - 0.95^{20} = 0.64\). Expect 1 false positive under the global null. Three "wins" is only weakly above chance.
- **Choose the error criterion by decision type:**
  - **FWER (Bonferroni or Holm)** when any false claim is costly, such as launch gates or safety. Bonferroni uses \(\alpha/m = 0.0025\). Holm is uniformly more powerful: compare the i-th smallest p-value with \(\alpha/(m-i+1)\).
  - **FDR (Benjamini-Hochberg)** for exploration and triage. Sort p-values and find the largest k with \(p_{(k)} \le kq/m\). Reject hypotheses 1 through k.
- **Better design:**
  - Pre-register one primary metric and a small set of guardrails.
  - Treat slice analysis as hypothesis generation, then confirm on a fresh sample.
  - Use hierarchical or partial-pooling models across slices so small slices shrink toward the global effect instead of producing noisy outliers.

### Trap 10 — Clustered, non-independent samples

**Scenario.** "We ran 200 tasks × 10 seeds = 2,000 trials, pass rate 70%. 95% CI is ±2.0 points, so a 3-point improvement is significant."

**Mid-level answer.** "Yes. \(\text{SE} = \sqrt{0.7 \cdot 0.3/2000} = 0.0102\), so the CI is ±2.0 pp."

**Staff analysis.**
- The 10 runs of one task are strongly correlated: task difficulty dominates. The independent unit is the task, not the trial. The design effect is
  \[\text{DEFF} = 1 + (m-1)\rho.\]
  With m = 10 and ICC ρ = 0.3, DEFF = 3.7. The effective sample size is \(2000/3.7 = 541\), and the CI inflates by \(\sqrt{3.7} = 1.92\), from ±2.0 pp to **±3.9 pp**. The 3-point "win" is not significant.
- The same problem arises with multiple turns per conversation, several traces per user or tenant, many judge calls per trace, and near-duplicate tasks from one template.
- **Correct inference:**
  - **Cluster bootstrap.** Resample *tasks* with replacement, keeping all runs of each task together.
  - Cluster-robust (sandwich) standard errors.
  - Mixed-effects logistic regression with a task random intercept.
  - When comparing agents, compute the paired per-task difference \(\bar d_t\) and bootstrap over tasks. Pairing removes the task-difficulty variance and is usually the largest single gain in power.
- **Staff-level addition.** When ρ is high, more tasks help far more than more seeds. Seeds beyond about 3–5 per task mostly estimate per-task variance, which is useful for pass^k (Trap 11) but not for the mean.

---

## C. Trajectory and agent evaluation

### Trap 11 — pass@k vs pass^k

**Scenario.** "Our customer-support agent has pass@8 = 99.9%. Product wants to market it as 'reliable'."

**Mid-level answer.** "99.9% over 8 attempts is very reliable."

**Staff analysis.**
- pass@k is the probability that *at least one* of k attempts succeeds: \(1-(1-p)^k\). That measures capability when a verifier can pick the successful attempt.
- pass^k (τ-bench) is the probability that *all* k attempts succeed: \(p^k\). That measures reliability, which is what a user who retries a task on different days experiences.
- With p = 0.8 per task: pass@8 = \(1 - 0.2^8 = 0.9999974\), while pass^8 = \(0.8^8 = 0.168\). Same agent, opposite conclusions.
- **Unbiased estimators** from n samples with c successes per task (n ≥ k):
  - \(\widehat{\text{pass@}k} = 1 - \binom{n-c}{k}/\binom{n}{k}\)
  - \(\widehat{\text{pass}^k} = \binom{c}{k}/\binom{n}{k}\)
  - Example: n = 10, c = 7, k = 3 gives 0.992 and 0.292.
- **Aggregation trap (Jensen).** Compute pass^k *per task*, then average. Do not compute \((\text{mean } p)^k\). Two tasks with p = 1.0 and p = 0.6 have mean p = 0.8, but mean pass^8 = \((1 + 0.6^8)/2 = 0.508\), not 0.168. The distribution of per-task reliability matters: consistently failing tasks are a different product problem from flaky tasks.
- **Staff answer.** Report pass^1, pass^k and the per-task success histogram. Use pass@k only where a verifier really does select among samples at inference time.

### Trap 12 — Trajectory matching that penalises valid alternative paths

**Scenario.** "We score agent trajectories by edit distance to a gold tool-call sequence. The new agent's score dropped, but users like it more."

**Mid-level answer.** "Edit distance is too strict. Use fuzzy matching or let a judge compare with the gold trajectory."

**Staff analysis.**
- Exact-sequence matching assumes one correct path. Real tasks allow valid orderings of commutative steps (look up the user, then the order, or the other way round), different but equivalent tools (search then fetch, versus a direct API call), extra harmless verification steps, and shortcuts. Matching measures *similarity to the annotator's path*, not correctness.
- A judge that compares against a gold path inherits the same anchoring bias.
- **Score against outcome and constraints, not the path:**
  1. **End-state check.** Diff the environment state after execution with the goal state (DB rows, files, tickets). This is τ-bench-style state comparison.
  2. **Constraint and invariant checks.** Required preconditions as a *partial order* (a DAG), for example "authenticate before any write" or "confirm with user before refund." Forbidden actions (policy violations, destructive calls) are hard failures.
  3. **Efficiency as a secondary metric.** Steps, tokens, cost and latency relative to a reference, with tolerance, reported separately from success.
  4. **Process judge only for what state can't capture,** such as communication quality or policy explanation, and with a rubric that explicitly allows alternative paths.
- **Validate the scorer.** Have humans annotate a set of *valid* alternative trajectories and measure the scorer's false-fail rate on them. Track it as a first-class metric of the evaluator.

### Trap 13 — Cascading errors in multi-step agents

**Scenario.** "Each step of our 20-step agent is 98% accurate on unit tests. Why is end-to-end success only 60%?"

**Mid-level answer.** "Errors compound: \(0.98^{20} = 0.67\). Improve each step."

**Staff analysis.**
- Independence gives \(0.98^{20} = 0.668\), and \(0.99^{50} = 0.605\). But the independence model is wrong in both directions:
  - **Worse than independent.** Errors propagate. A wrong entity resolved at step 3 contaminates all later context, so later steps are conditionally worse, and per-step unit tests on *clean* inputs overestimate in-trajectory accuracy.
  - **Better than independent.** Agents can self-correct: retry on tool error, re-verify.
  - The observed 60% compared with 66.8% under independence says propagation outweighs recovery.
- **Measure step accuracy conditioned on context:**
  - **First-error localisation.** For each failed trajectory, find the earliest step where the trajectory became unrecoverable (human or judge annotation). Build a histogram of first-error step type. That is your fix priority, not per-step unit accuracy.
  - **Counterfactual replay.** Checkpoint the state at step t, substitute the gold action, and replay forward (needs the sandbox in System Design, Scenario 8). The change in success is the causal contribution of step t.
  - **Recovery rate.** \(P(\text{success} \mid \text{error at step } t)\). Invest in detection and recovery (verifiers, checkpoints) where recovery is cheap, and in step accuracy where errors are unrecoverable.

---

## D. Data and leakage

### Trap 14 — Benchmark contamination

**Scenario.** "The new base model jumped 15 points on our public coding benchmark but only 2 points on internal tasks."

**Mid-level answer.** "The internal tasks are harder. The public benchmark gain is real capability."

**Staff analysis.** A gap like that between public and private sets is a contamination signal until shown otherwise.
- **Detection:**
  - n-gram or embedding overlap between benchmark items and any available training corpus.
  - Canary strings (BIG-bench style) embedded in the benchmark.
  - Min-k% token-probability membership inference. The model assigns anomalously high likelihood to the verbatim test items.
  - **Perturbation gap.** Rephrase questions, rename variables, change numbers. Real capability survives; memorisation drops.
  - **Temporal split.** Compare performance on items created before and after the model's training cutoff (LiveCodeBench-style).
- **Prevention:**
  - Private held-out sets that never leave the eval service. Only aggregate scores are released.
  - Procedurally generated or parameterised tasks with fresh instances per evaluation.
  - A rolling benchmark refresh with retired items.
  - Access logging on the eval set.
- **Agent-specific contamination.** Web-browsing agents can find the benchmark answers online during the task. Sandbox with allow-listed domains and log retrievals (see System Design, Scenario 8).

### Trap 15 — Time-travel leakage

**Scenario.** "We trained a trace-quality classifier on 6 months of logs with a random 80/20 split. Offline AUC is 0.94; in production it's 0.78."

**Mid-level answer.** "Overfitting. Add regularisation."

**Staff analysis.** Two leakage channels:
1. **Random split across time.** Near-duplicate traces from the same week, user, incident or prompt template land in both train and test. The model learns period-specific artefacts such as a bug that briefly caused a failure mode.
2. **Features computed after the fact.** For example "user escalated," "ticket reopened" or "trace length," which are only known after the outcome and sometimes leak the label directly. At scoring time they don't exist or have different distributions.

**Fixes:**
- Use a time-based split: train on \([t_0, t_1)\), test on \([t_1 + \text{embargo}, t_2)\), with an embargo at least as long as the longest label-maturation window.
- Group-aware splits by user, tenant or task template.
- A feature availability audit: every feature is tagged with the timestamp at which it becomes available, and the training pipeline uses point-in-time joins.
- Backtest with rolling-origin evaluation to measure degradation as a function of the train-test gap.

### Trap 16 — Covariate shift between eval set and production

**Scenario.** "Our judge was validated on a human-labelled set curated last quarter: 91% agreement. Traffic is now 40% from a new tool-use product. Is the 91% still valid?"

**Mid-level answer.** "The judge is general-purpose, so it should transfer."

**Staff analysis.**
- Under covariate shift, \(p(x)\) changes while \(p(y \mid x)\) is assumed stable. Agreement measured under \(p_{\text{source}}\) is not agreement under \(p_{\text{target}}\).
- **Reweight.** Train a domain classifier to distinguish source from target items. Then
  \[w(x) = \frac{P(\text{target} \mid x)}{P(\text{source} \mid x)} \cdot \frac{n_s}{n_t}\]
  and estimate target agreement as \(\sum_i w_i \mathbb{1}[\text{agree}_i]/\sum_i w_i\).
- **Check effective sample size:** \(\text{ESS} = (\sum w_i)^2/\sum w_i^2\). Example: 800 items with w = 0.5 and 200 with w = 3.0 give ESS = 500. With heavy weights the estimate is noisy. If the new product has no support in the source set (w undefined), no reweighting can fix it. You need new labels.
- **Operational answer:**
  - Monitor the input distribution with embedding drift, the domain-classifier AUC, and the share of traffic in each slice.
  - Trigger targeted human labelling of the new slice when the domain classifier's AUC is above about 0.7, or when a slice exceeds X% of traffic.
  - Report judge validity per slice with the date it was measured.
- **Distinguish label shift and concept drift.** Under label shift, \(p(y)\) changes (e.g. the failure rate rises). Under concept drift, \(p(y \mid x)\) changes (e.g. the policy changes what counts as a violation). Concept drift invalidates the labels themselves.

---

## E. Metric and business alignment

### Trap 17 — Optimising prompts against the judge (Goodhart)

**Scenario.** "We ran automated prompt optimisation for the agent against our judge. Judge score went from 7.1 to 8.6 over 30 iterations. Launch?"

**Mid-level answer.** "Huge gain. Maybe check a few examples first."

**Staff analysis.**
- An optimiser searching against a fixed, imperfect judge finds the judge's blind spots: verbosity, confident tone, rubric keywords, repeating the rubric's language, and self-assessment statements ("I have verified…"). The judge score goes up, and the divergence from true quality grows with optimisation pressure.
- **Detect it:**
  - Track the **judge-human gap along the optimisation trajectory.** Human-label a fixed sample at iterations 0, 10, 20 and 30. If judge score rises while human score plateaus or falls, you are overfitting the judge.
  - Use a **held-out judge** from a different family and rubric phrasing, never used in optimisation. Divergence between the optimisation judge and the held-out judge is a cheap Goodhart alarm.
  - Watch **distribution shift in outputs**: length, hedging-phrase frequency, rubric-term frequency.
- **Mitigate:**
  - Separate optimisation and evaluation judges.
  - Regularise toward the base prompt, and cap the number of optimisation steps.
  - Keep a final human evaluation gate.
  - Periodically adversarially harden the judge with the high-scoring, human-rejected outputs.

### Trap 18 — Offline metric misaligned with the business metric

**Scenario.** "Offline judge-scored helpfulness improved 4 points on each of the last three launches. Online resolution rate didn't move. The VP asks whether the eval platform is worth funding."

**Mid-level answer.** "Online metrics are noisy. Give it time." Or: "Add more rubric dimensions."

**Staff analysis.**
- An offline metric is useful only if it is **directionally predictive** of the online metric the business cares about. That is an empirical property to measure, not assume.
- **Metric validation across launches:**
  - For each past launch i, record \((\Delta_i^{\text{offline}}, \Delta_i^{\text{online}})\).
  - Measure sign agreement, rank correlation, and the slope of a regression of online on offline deltas, with CIs.
  - With few launches, add deliberately degraded "anti-launches" (known-worse variants in small online tests) to widen the range.
- **Common causes of misalignment:**
  - The rubric rewards properties users don't value, such as thoroughness when they want speed.
  - The offline set doesn't match production traffic (Trap 16).
  - The online metric is bottlenecked elsewhere, for example resolution capped by missing tool permissions that no response-quality change can fix.
  - The offline metric saturates in the region where launches operate.
- **Staff move:**
  - Build a **metric hierarchy**. North-star online metric (resolution, retention, cost per resolved task) → proxy metrics validated to predict it → diagnostic metrics for debugging.
  - Retire proxies whose predictive validity falls below a threshold.
  - Show the VP the validation analysis rather than asserting the metric's value.

### Trap 19 — Feedback loops

**Scenario.** "We only send traces the judge flags as failures to human review, then use those labels to re-calibrate the judge. The judge's precision looks better every month."

**Mid-level answer.** "Great. Active learning is working."

**Staff analysis.**
- **Selection bias.** Humans never see judge-passes, so judge false negatives are never observed. Precision can be measured; recall cannot. Recalibrating only on flagged items drifts the judge toward whatever it already flags.
- **Performative loop.** If the agent is trained or filtered with the judge, the future data distribution is shaped by the judge, which then looks increasingly accurate on data it created.
- **Fix: a random audit stream with known inclusion probabilities.** Send every item to review with probability \(\pi_i\) (e.g. 1% of judge-passes and 20% of judge-fails). Estimate population quantities with Horvitz-Thompson:
  \[\hat{Y} = \frac{1}{N}\sum_{i \in S} \frac{y_i}{\pi_i}.\]
  This recovers unbiased recall, precision and base rate. Uncertainty sampling can stay for model improvement, but *measurement* must come from the probability sample.
- **Staff-level addition.** Log the judge version that made each routing decision. That lets you reconstruct inclusion probabilities after the fact when policies change.

---

## F. Safety

### Trap 20 — Base rates: a "99% accurate" safety judge

**Scenario.** "Our safety classifier has 99% sensitivity and 99% specificity. Policy violations occur in 0.1% of agent traces. Should flagged traces auto-suspend the tenant?"

**Mid-level answer.** "99% accurate, so yes."

**Staff analysis.**
- \[\text{PPV} = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.01 \times 0.999} = \frac{0.00099}{0.01098} = 0.090.\]
  91% of flags are false positives. Auto-suspension would mostly punish innocent tenants.
- **Design by cost:**
  - Choose the operating threshold from explicit costs, \(C_{FP}\) (wrongly suspending a customer) and \(C_{FN}\) (missed harm), and the prevalence. Don't use the default threshold of 0.5.
  - Use a **tiered response.** A high-recall first stage (cheap classifier) → a high-precision second stage (strong judge or ensemble) → human review for irreversible actions. Two conditionally independent stages multiply the likelihood ratios. Real stages are correlated, so measure the stacked PPV empirically.
  - Measure over-refusal alongside violation rate. A safety judge tuned only for recall pushes the agent toward refusing benign requests, a harm that doesn't appear on the safety dashboard.
- **Prevalence varies by tenant and product.** The PPV must be computed per segment, and the threshold may need to differ.

### Trap 21 — "Zero violations observed" on a red-team set

**Scenario.** "We ran 1,000 adversarial prompts against the agent: zero policy violations. Can we claim the violation rate is below 0.1%?"

**Mid-level answer.** "Zero out of 1,000 means the rate is essentially zero."

**Staff analysis.**
- With 0 events in n trials, the one-sided 95% upper bound is \(1 - 0.05^{1/n} \approx 3/n\) (rule of three). For n = 1,000 that is **0.30%**, not below 0.1%. To claim below 0.1% at 95% confidence with zero observed, you need \(n \ge \ln 0.05/\ln 0.999 = 2{,}995\) independent trials.
- **But the bound covers only the tested distribution.**
  - A static red-team set measures robustness to *those* attacks. Adaptive attackers search the space, so the attack success rate against a fixed set is a lower bound on real-world risk.
  - Prompts from the same template or attack family are clustered (Trap 10), which reduces effective n.
- **Staff answer:**
  - Report the upper bound by attack family.
  - Add adaptive red-teaming: automated attackers, iterative jailbreak search, and human red teams with a budget.
  - Track **attack success rate vs. attacker budget** curves.
  - Include agent-specific vectors: indirect prompt injection via tool outputs and retrieved documents, and privilege escalation through tool chaining.
  - Treat safety claims as "no violations found under this threat model and budget," never as absolute.
