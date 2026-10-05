# 07 — Aman PDF Quick-Revision Cheatsheet (Staff/Senior Applied Scientist / ML Engineer)

Built from: **Aman's AI Journal Interview Questions PDF**.  
Goal: high-density review in **10–15 minutes** before interviews.

---

## 0) 2-minute review strategy

1. Read sections **1, 2, 5** first (highest interview hit rate).
2. Then skim **3 and 4** for depth probes.
3. Use section **8** for formula recall and section **9** for rapid-fire.
4. If you get stuck in interview: answer with **definition -> intuition -> equation -> trade-off -> failure mode**.

---

## 1) Core ML + DL fundamentals (must-answer cleanly)

### Bias-variance tradeoff
- **Bias**: underfit, high training/test error.
- **Variance**: overfit, low train error, high test error.
- Control with model capacity, regularization, data size, augmentation.
- Staff-level answer includes **which knob** you would turn first and why.

### L1 vs L2 regularization
- L1 penalty: \(\lambda \sum_i |w_i|\) -> sparse weights, feature selection.
- L2 penalty: \(\lambda \sum_i w_i^2\) -> smooth shrinkage, stability.
- Use L2 as default for dense signals; L1 when sparsity/selection matters.

### Why random initialization?
- Same initialization for all neurons in a layer -> identical gradients -> neurons remain symmetric -> no diversity learned.
- Random init breaks symmetry and allows specialized feature learning.

### Why MSE is poor for classification
- MSE with sigmoid can yield slow gradients in saturated regions.
- Cross-entropy gives stronger, better-shaped gradients for probabilistic classification.

### Batch / stochastic / mini-batch GD
- Batch GD: stable but slow per update.
- SGD: noisy but can escape sharp minima/saddles.
- Mini-batch: practical default; best throughput and convergence balance.

---

## 2) Transformers / LLMs / GenAI (very common senior probes)

### Self-attention in one line
- Computes context-aware token representations by weighted aggregation of all token values.

### Why Q, K, V?
- \(QK^\top\) computes relevance scores (who should attend to whom).
- \(V\) holds payload to aggregate.
- Separating \(K\) and \(V\) decouples **matching** from **content**.

### Attention formula
- \(\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\)
- \(\sqrt{d_k}\) scaling prevents large dot products from saturating softmax.

### Transformer drawbacks
- \(O(n^2)\) attention cost (memory/latency bottleneck on long context).
- Data-hungry; can hallucinate; inference expensive.
- Mitigations: sparse/linear attention, retrieval, chunking, KV-cache optimizations.

### Temperature / top-k / top-p
- Temperature controls sharpness of token distribution.
- top-k truncates to k most likely tokens.
- top-p (nucleus) truncates to smallest set with cumulative prob >= p.
- Higher values => more diversity, less determinism.

### Why same prompt can produce different outputs
- Sampling randomness + temperature/top-k/top-p + seed differences + backend nondeterminism.
- At near-zero temperature and fixed seed, outputs are more reproducible.

### DDPM vs DDIM
- DDPM: stochastic reverse process; more diverse, slower sampling.
- DDIM: deterministic/implicit trajectory; faster sampling, often less diversity.

### GAN mode collapse
- Generator maps many latent points to limited outputs (low diversity).
- Mitigate with improved objectives (WGAN-GP), minibatch discrimination, better D/G balance.

### RLHF at high level
- SFT -> preference data -> reward model -> PPO/DPO-style optimization.
- Key risk: reward hacking / alignment tax / preference noise.

---

## 3) Training pathologies + debugging checklist

### Overfitting mitigation stack (in order)
1. Data quality + leakage check.
2. Train/val split sanity (no overlap, proper time split if needed).
3. Regularization (weight decay, dropout, label smoothing).
4. Early stopping + learning-rate schedule.
5. Augmentation / more data.
6. Reduce model capacity or freeze lower layers in fine-tuning.

### Vanishing gradients
- Deep nets with saturating activations or poor initialization can lose gradient signal.
- Fixes: ReLU-family activations, residual connections, normalization, careful initialization.

### Batch norm: why it helps
- Stabilizes optimization landscape and gradient flow.
- Enables larger learning rates and faster convergence.

### Validation loss up while accuracy up
- Possible when confidence calibration worsens: more correct labels but overconfident wrong predictions inflate log loss.

### If test performance underperforms
- Check: leakage, distribution shift, label noise, class imbalance, metric mismatch, insufficient hyperparameter search, data preprocessing mismatch.

---

## 4) Classical ML, stats, probability (still asked frequently)

### Pearson vs Spearman
- Pearson: linear correlation on raw values.
- Spearman: rank correlation (monotonic relationships, robust to outliers/non-normality).

### CLT use case
- Means/sums of many independent samples approximate normal; allows confidence intervals/tests even if raw data non-normal (with caveats).

### Cross-validation
- Estimates generalization robustly on small/medium datasets.
- Use **stratified** for imbalance, **grouped/time-series CV** when IID assumption breaks.

### Random forest vs gradient boosting
- RF: bagging, parallel trees, variance reduction, robust baseline.
- GBM/XGBoost: sequential residual fitting, lower bias, often better accuracy, more tuning-sensitive.

### K-means convergence nuance
- Objective (within-cluster SSE) monotonically decreases; converges to a **local** optimum.
- Not guaranteed unique/global optimum; sensitive to initialization.

### Imbalanced classes
- Use class weights, focal loss, threshold tuning, resampling, PR-AUC/F1/recall@precision targets.
- Avoid reporting accuracy alone.

### Long-tail data
- Majority classes dominate gradients.
- Use reweighting, resampling, decoupled head training, focal/class-balanced losses.

---

## 5) Experimentation and decision quality (staff differentiator)

### A/B testing pitfalls
- Sample ratio mismatch, peeking, multiple comparisons, novelty effects, interference between users.
- Staff answer includes pre-registered metrics + guardrails + stopping rule.

### P-value in plain language
- Probability of observing data this extreme (or more) **assuming null is true**.
- Not the probability that null is true.

### Metric selection
- Match metric to business objective and failure costs.
- Track primary metric + guardrails (latency, safety, fairness, cost).

### Selection bias
- Data collection process distorts representativeness.
- Fix with better sampling design, randomization, stratification, and post-stratification checks.

---

## 6) Applied ML engineer/system view (what senior loops test)

### On-device deployment
- Optimize with quantization, pruning, distillation, operator fusion, batching policy.
- Track latency p95/p99, memory footprint, thermal/power constraints.

### Distribution shift
- Types: covariate shift, label shift, concept drift.
- Build drift monitors + retraining triggers + rollback plan.

### Uncertainty for regression
- Aleatoric (data noise) vs epistemic (model uncertainty).
- Methods: quantile regression, ensembling/MC dropout, conformal prediction.

### Hallucination mitigation
- Retrieval grounding, constrained decoding, tool verification, abstain option, self-consistency with verifier.

---

## 7) SQL/Python coding mini-cheats

### SQL order of execution (logical)
`FROM -> JOIN -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY -> LIMIT`

### Common SQL gotchas
- `WHERE` filters rows pre-aggregation; `HAVING` filters groups post-aggregation.
- `BETWEEN` is inclusive; `IN` matches set membership.
- Prefer explicit joins over comma joins.

### Python quick reminders
- `list`: mutable; `tuple`: immutable/hashable (if elements hashable).
- For set operations/intersection at scale, convert to `set` first.
- For duplicates: hash map / `collections.Counter` / sort+scan.

---

## 8) Formula flashcards (memorize)

- **Cross-entropy (binary)**: \(-[y\log p + (1-y)\log(1-p)]\)
- **L2-regularized loss**: \(L = L_{\text{data}} + \lambda\sum_i w_i^2\)
- **L1-regularized loss**: \(L = L_{\text{data}} + \lambda\sum_i |w_i|\)
- **Softmax**: \(\sigma(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}\)
- **Bayes rule**: \(P(A|B)=\frac{P(B|A)P(A)}{P(B)}\)
- **Precision**: \(TP/(TP+FP)\)
- **Recall**: \(TP/(TP+FN)\)
- **F1**: \(2PR/(P+R)\)
- **Attention**: \(\text{softmax}(QK^\top/\sqrt{d_k})V\)
- **K-means objective**: \(\sum_k\sum_{x_i\in C_k}\|x_i-\mu_k\|^2\)

---

## 9) Rapid-fire (speak each in <=20 seconds)

1. Why random initialization? -> symmetry breaking.
2. Why mini-batch over full batch? -> throughput + stable enough gradients.
3. Pearson vs Spearman? -> linear vs rank-monotonic.
4. Why CE over MSE for classification? -> better gradients/probabilistic fit.
5. What is overfitting? -> memorization; poor generalization.
6. How to detect leakage? -> implausibly high CV, feature availability audit, time-based holdout.
7. Random forest vs boosting? -> variance reduction vs bias reduction.
8. Why Q/K/V? -> relevance scoring separated from value aggregation.
9. Why transformer expensive? -> quadratic attention.
10. Why outputs vary for same prompt? -> sampling/randomness/backend nondeterminism.
11. What is mode collapse? -> generator loses diversity.
12. What is focal loss doing? -> downweights easy examples, focuses hard/minority samples.
13. K-means guaranteed global optimum? -> no, local optimum.
14. What is label smoothing? -> reduce overconfidence, improve calibration.
15. What is CLT and why useful? -> sampling mean approx normal for inference.
16. What is selection bias? -> non-representative sampling; invalid conclusions.
17. What is batch norm benefit? -> stabilizes and speeds training.
18. What is concept drift? -> \(P(y|x)\) changes over time.
19. How to choose eval metric under imbalance? -> PR-AUC/recall@precision + business-cost thresholds.
20. Staff-level escalation trigger? -> when metric gain conflicts with safety/cost/latency guardrails.

---

## 10) 30-second staff answer template (use for any hard question)

1. **Define** the concept precisely.
2. Give **one intuition/example**.
3. Add **one equation/metric**.
4. State **trade-offs** and failure modes.
5. End with **how you'd validate** in production.

This format avoids rambling and signals senior depth quickly.
