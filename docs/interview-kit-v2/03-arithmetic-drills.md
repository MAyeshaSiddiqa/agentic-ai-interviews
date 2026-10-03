# 03 — Arithmetic Drills: Hardware, Serving, Cost and Statistics

Fifteen back-of-envelope drills for an AI-agent evaluation platform: LLM-as-a-judge over large volumes of agent traces, served through a mix of self-hosted GPUs and production LLM APIs. Every drill has three parts: **Problem**, **Formula / logic**, and **Math → Final**. Every number in this file was recomputed by script. If you get a different answer, the error is yours or one of your assumptions differs; find out which.

Interviewers use these drills to see whether you can find the **binding constraint** (memory, bandwidth, compute, rate limit, or statistical power). A number with no statement of what binds it is only half an answer.

---

## Conventions and approximations (state these out loud in an interview)

| Quantity | Value used | Note |
|---|---|---|
| Storage and memory units | 1 KB = 10³ B, 1 MB = 10⁶ B, 1 GB = 10⁹ B, 1 TB = 10¹² B | Decimal throughout. GiB differs from GB by about 7.4%, which is well inside back-of-envelope error, but say which one you use. |
| Bytes per parameter | fp16/bf16 = 2, fp8/int8 = 1, int4 = 0.5 | Excludes quantization scales unless stated. |
| Reference GPU | NVIDIA H100 SXM: 80 GB HBM3, **3.35 TB/s** memory bandwidth, **989 TFLOP/s** dense BF16 | The 1,979 TFLOP/s headline figure assumes 2:4 sparsity. Do not use it for dense models. |
| Forward-pass FLOPs | ≈ **2 · params · tokens** | Ignores the attention-score term. Drill 5 shows that term is about 5% at 6k context for a 70B model. |
| Reference 70B model | 80 layers, d_model 8192, 64 query heads, **8 KV heads (GQA)**, head_dim 128 | Llama-3-70B-shaped. |
| Reference 8B model | 32 layers, 8 KV heads, head_dim 128 | Llama-3-8B-shaped. |
| GPU price | $2.50 per GPU-hour | Hypothetical blended cloud rate. |
| API prices | "Premium judge": $3 / 1M input, $15 / 1M output. "Budget judge": $0.50 / 1M input, $1.50 / 1M output | Hypothetical list prices. Never quote a real vendor's price from memory in an interview; parameterize. |
| Day | 86,400 s = 1,440 min | |
| Normal quantiles | z₀.₉₇₅ = 1.960, z₀.₈₀ = 0.8416 | |

**Running workload** (reused across drills): **2,000,000 traces/day**. Each judge call is **6,000 input tokens** (2,000-token static rubric and system prefix plus a 4,000-token trace excerpt) and **300 output tokens**.

---

## Drill 1 — VRAM for weights at fp16, int8, int4

**Problem.** You want to self-host a 70B-parameter judge on 80 GB H100s. How much memory do the weights alone take at fp16, int8, and int4? What is the minimum GPU count for each, before any KV cache?

**Formula.** `weight_bytes = params × bytes_per_param`. Group-wise int4 (for example, group size 128 with one fp16 scale per group) adds `16 / 128 = 0.125` bits per weight.

**Math.**
1. fp16: `70×10⁹ × 2 = 140×10⁹ B` = **140 GB**, so at least 2 GPUs (140 > 80).
2. int8: `70×10⁹ × 1` = **70 GB**. It fits on 1 GPU, leaving 10 GB for everything else.
3. int4 (raw): `70×10⁹ × 0.5` = **35 GB**.
4. int4 with scales: `70×10⁹ × (4 + 0.125) / 8 = 36.09 GB` ≈ **36.1 GB**, so 1 GPU with about 44 GB free.

**Final.** 140 / 70 / 35 GB (≈36.1 GB with scales). The minimum is 2 / 1 / 1 GPUs, but "fits" is not the same as "serves". At int8 on one GPU, 10 GB of headroom holds only about 30k tokens of KV cache (Drill 2), roughly five 6k-token judge calls. The practical fp16 deployment is TP=4 or TP=8, sized for KV cache rather than weights. Quantizing the judge also changes its verdicts, so re-measure agreement with human labels before you quantize a judge in production.

---

## Drill 2 — KV-cache size per token and per batch

**Problem.** For the 70B model in fp16, compute the KV cache per token, per 8,192-token sequence, and for a batch of 32 such sequences. On a TP=4 node (4 × 80 GB), how many concurrent 8k sequences fit after the fp16 weights? What would the per-token figure be without GQA (64 KV heads)?

**Formula.** `KV_bytes_per_token = layers × 2 (K and V) × kv_heads × head_dim × bytes`.

**Math.**
1. Per token: `80 × 2 × 8 × 128 × 2 = 327,680 B` ≈ **0.328 MB/token**.
2. Per 8,192-token sequence: `327,680 × 8,192 = 2,684,354,560 B` ≈ **2.68 GB**.
3. Batch of 32: `32 × 2.684 GB` ≈ **85.9 GB**, which is more than one H100 holds.
4. TP=4 node: `4 × 80 − 140 = 180 GB` free, and `180 / 2.684` = **67 sequences** at the ceiling. That ignores activations, CUDA graphs and allocator fragmentation; budget about 10% less, so about 60.
5. Without GQA (64 KV heads): `80 × 2 × 64 × 128 × 2 = 2,621,440 B` ≈ **2.62 MB/token**, 8× larger, so only about 8 sequences would fit.

**Final.** **0.328 MB/token, 2.68 GB per 8k sequence, 85.9 GB per batch of 32.** At long context, concurrency is bounded by KV cache, not weights. The levers are GQA/MQA (architecture), an fp8 KV cache (halves it), paged attention (removes fragmentation), and prefix sharing (Drill 9).

---

## Drill 3 — Memory-bandwidth ceiling on decode tokens/s

**Problem.** (a) An 8B judge in fp16 runs on one H100 at batch size 1. What is the upper bound on decode tokens/s? (b) Repeat for the 70B fp16 model on TP=4. (c) For the 8B model at batch 32 with 4,096 tokens of context per sequence, what are the aggregate and per-sequence decode rates?

**Logic.** Each decode step must stream every weight byte from HBM, plus the KV cache for every sequence in the batch. At small batch, the arithmetic is negligible next to that memory traffic, so:
`step_time ≥ (weight_bytes + Σ KV_bytes) / bandwidth`, and `tokens/s per sequence ≤ 1 / step_time`.

**Math.**
1. (a) `3.35×10¹² / 16×10⁹` = **209 tokens/s** ceiling. Real kernels reach about 60–80% of peak bandwidth, so expect about 130–170.
2. (b) Aggregate bandwidth is `4 × 3.35 = 13.4 TB/s`, so `13.4×10¹² / 140×10⁹` = **95.7 tokens/s** per sequence, before tensor-parallel all-reduce overhead.
3. (c) The 8B model's KV cache per token is `32 × 2 × 8 × 128 × 2 = 131,072 B`. The batch's KV is `32 × 4,096 × 131,072` = **17.18 GB**.
   - Bytes per step: `16 + 17.18 = 33.18 GB`, so the step takes `33.18×10⁹ / 3.35×10¹²` = **9.90 ms**.
   - That is 101 steps/s, so **≈101 tokens/s per sequence** and `32 × 101` = **≈3,231 tokens/s aggregate**.
   - Compute check: `2 × 8×10⁹ × 32 / 989×10¹² = 0.52 ms` ≪ 9.90 ms, so the step is memory-bound.

**Final.** 209 tok/s (8B, batch 1), 95.7 tok/s (70B, TP=4), and ≈3,231 tok/s aggregate at batch 32 with 4k context. Batching amortizes the weight reads, but at this context length the KV reads (17.18 GB) already exceed the weight reads (16 GB), so throughput grows sub-linearly with batch size.

---

## Drill 4 — The roofline: when does decode become compute-bound?

**Problem.** At what batch size does the 8B fp16 decode step on an H100 stop being memory-bound? Answer first ignoring KV reads, then with 4,096 tokens of context per sequence.

**Formula.** The ridge point is `peak_FLOPs / bandwidth`. A step's arithmetic intensity is `FLOPs / bytes = (2 · P · B) / (2P + B · KV_seq)`, where P is the parameter count and 2P is the fp16 weight bytes.

**Math.**
1. Ridge point: `989×10¹² / 3.35×10¹²` = **295 FLOPs/byte**.
2. Ignoring KV: the intensity is `2PB / 2P = B`, so the step becomes compute-bound at **B ≳ 295**.
3. With KV at 4k context: `KV_seq = 4,096 × 131,072 = 0.537 GB`.
   - Intensity at B = 32, 128, 512, 1,024: **15.4, 24.2, 28.2, 29.0** FLOPs/byte.
   - As B → ∞: `2P / KV_seq = 16×10⁹ / 0.537×10⁹` = **29.8** FLOPs/byte.

**Final.** Without KV, decode flips to compute-bound around batch 295. With 4k-token contexts it **never** does: intensity is capped at about 30 FLOPs/byte, roughly 10× below the ridge. Long-context judge decode is permanently bandwidth-bound. More FLOPs will not help; the levers are fewer KV bytes (GQA, fp8 KV, shorter judge outputs) and prefix sharing.

---

## Drill 5 — Prefill FLOPs vs GPU peak compute and utilization

**Problem.** A judge call puts a 6,000-token prompt through the 70B model. Compute the prefill FLOPs and the time-to-first-token (TTFT) on 1 GPU and on TP=4, at 40% model FLOPs utilization (MFU). How big is the attention term that the 2·P·T rule ignores?

**Formula.** `FLOPs ≈ 2 · P · T`. `time = FLOPs / (N_gpu × peak × MFU)`. Causal attention adds about `4 · layers · d_model · (T/2)` FLOPs per token (QKᵀ plus AV, at an average context of T/2).

**Math.**
1. FLOPs: `2 × 70×10⁹ × 6,000` = **8.4×10¹⁴**.
2. Effective throughput per GPU: `989×10¹² × 0.40` = **395.6 TFLOP/s**.
3. On 1 GPU (compute only, ignoring that fp16 weights don't fit on one GPU): `8.4×10¹⁴ / 3.956×10¹⁴` = **2.12 s**.
4. On TP=4: `2.12 / 4` = **0.53 s**, ignoring communication.
5. Attention term: `4 × 80 × 8,192 × 3,000` = 7.86×10⁹ FLOPs per token, against `2 × 70×10⁹` = 1.4×10¹¹. That is **5.6%**, so the rule is safe at 6k. At 100k context the attention term becomes comparable to the parameter term.

**Final.** **8.4×10¹⁴ FLOPs; TTFT ≈ 2.1 s on one GPU, ≈ 0.53 s on TP=4, at 40% MFU.** Prefill is compute-bound and decode is bandwidth-bound (Drills 3–4). That asymmetry is why judge workloads (long input, short output) are dominated by prefill, and why prefix caching pays off.

---

## Drill 6 — GPU fleet for N traces/day, and self-hosted vs API break-even

**Problem.** Judge the running workload (2M traces/day, 6,000 input and 300 output tokens each) with the 70B fp16 model on 8×H100 nodes (TP=8).
(a) How many GPUs do you need?
(b) What does the fleet cost per day?
(c) At what daily volume does self-hosting beat the budget API ($0.50 / $1.50 per 1M) and the premium API ($3 / $15)? Include 2 platform/ML engineers at a fully loaded $400k/year each as the fixed operating cost of self-hosting.

**Logic.** Size prefill by compute and decode by bandwidth, add the GPU-time, then divide by a target utilization so there is headroom for peaks and failures.

**Math.**
1. **Prefill.**
   - Input tokens: `2×10⁶ × 6,000 = 1.2×10¹⁰` per day.
   - FLOPs: `2 × 70×10⁹ × 1.2×10¹⁰` = **1.68×10²¹**.
   - One GPU-day at 40% MFU: `989×10¹² × 0.4 × 86,400` = **3.418×10¹⁹** FLOPs.
   - GPU-days needed: `1.68×10²¹ / 3.418×10¹⁹` = **49.2 GPUs**.
2. **Decode.**
   - Output tokens: `2×10⁶ × 300 = 6×10⁸` per day, which is **6,944 tokens/s** on average.
   - Node bandwidth at 60% efficiency: `8 × 3.35 × 0.6` = **16.08 TB/s**.
   - At batch 64 and ≈6,150 tokens of context per sequence, KV per sequence is `6,150 × 327,680 = 2.015 GB`, so **129.0 GB** for the batch.
   - Bytes per step: `140 + 129 = 269 GB`, so a step takes **16.7 ms**. That is 59.8 steps/s, or `× 64` = **3,826 tokens/s per node**.
   - Nodes: `6,944 / 3,826 = 1.815` nodes = **14.5 GPUs**.
3. **Total.**
   - GPU-time: `49.2 + 14.5 = 63.7` GPUs at 100% duty.
   - At a 70% utilization target: `63.7 / 0.7 = 91.0`. Round up to whole nodes: **12 nodes = 96 GPUs**.
4. **Cost.**
   - GPUs: `96 × 24 × $2.50` = **$5,760/day**, which is **$0.00288/trace**.
   - Ops staff: `2 × $400k / 365` = **$2,192/day**.
   - Total: **$7,952/day**.
5. **API cost per trace.**
   - Budget: `6,000 × 0.5×10⁻⁶ + 300 × 1.5×10⁻⁶` = **$0.00345**, which is **$6,900/day** at 2M traces.
   - Premium: `6,000 × 3×10⁻⁶ + 300 × 15×10⁻⁶` = **$0.0225**, which is **$45,000/day**.
6. **Break-even.** Treat GPU cost as scaling linearly with volume, in node-sized steps. Then `N* = fixed_ops / (api_per_trace − gpu_per_trace)`.
   - Budget: `2,192 / (0.00345 − 0.00288)` = **≈3.85M traces/day**.
   - Premium: `2,192 / (0.0225 − 0.00288)` = **≈112k traces/day**.

**Final.** **96 H100s, ≈$5.8k/day in GPUs, ≈$8.0k/day with ops staff.** At 2M traces/day, self-hosting loses to the budget API ($6.9k/day, break-even ≈3.85M/day) and beats the premium API by about 5.7× (break-even ≈112k/day). The comparison is only valid if the self-hosted 70B judge **matches the API judge's agreement with human labels**. If it doesn't, you are comparing the cost of two different measurement instruments. Also account for the fact that the self-hosted fleet's capacity is fixed: it has to cover peaks, which an API does not charge you for.

---

## Drill 7 — Throughput under provider rate limits (RPM / TPM)

**Problem.** Your API key allows 4,000 requests/min (RPM) and 2,000,000 tokens/min (TPM), with input and output counted together. Each call is 6,300 tokens. What is the maximum sustained throughput, and which limit binds? How much quota do you need for 2M traces/day? What changes if the provider counts `max_tokens = 1,024` against TPM at admission, rather than the 300 tokens actually generated?

**Formula.** `max_RPM = min(RPM_limit, TPM_limit / tokens_per_request)`.

**Math.**
1. The TPM bound is `2,000,000 / 6,300` = **317.5 RPM**, far below 4,000, so **TPM binds**.
2. That is `317.5 / 60` = **5.29 requests/s**, or `317.5 × 1,440` = **457,143 traces/day**.
3. Required rate for 2M/day: `2×10⁶ / 1,440` = **1,389 RPM**, and `× 6,300` = **8.75M TPM**. That is **4.375×** the limit, so you need at least 5 keys' worth of quota (or a higher tier) at a flat arrival rate.
4. With a 2× peak-to-average ratio you need **17.5M TPM**, unless a queue smooths the arrivals.
5. If `max_tokens` is reserved at admission, each call counts as `6,000 + 1,024 = 7,024` tokens, so `2×10⁶ / 7,024` = **284.7 RPM**. That is a 10% capacity loss from a parameter you never use. Set `max_tokens` close to the real output length.

**Final.** **≈317 RPM (≈457k traces/day) per key, and TPM binds.** 2M/day needs 8.75M TPM on average. Admission control must be token-aware (a token bucket keyed on estimated tokens), not request-aware; a limiter that only counts requests will send you straight into 429 storms.

---

## Drill 8 — Little's law for concurrency

**Problem.** Mean end-to-end judge-call latency, including queueing, is 12 s. How many requests must be in flight to (a) saturate one key at 5.29 req/s, and (b) sustain 2M traces/day? (c) Redo (b) with 5% of calls retried once.

**Formula.** `L = λ · W`: in-flight requests = arrival rate × mean time in system.

**Math.**
1. (a) `5.29 × 12` = **63.5, so 64** concurrent requests.
2. (b) The rate is `2×10⁶ / 86,400` = **23.15 req/s**, and `× 12` = **277.8, so 278**.
3. (c) Retries raise the rate to `23.15 × 1.05 = 24.31` req/s, and `× 12` = **291.7, so 292**.

**Final.** **64 per key; 278 for the fleet (292 with 5% retries).** Use the **mean** latency W, not p99. But size the connection pool and worker semaphore with headroom above L, because W itself rises when the provider slows down. If the pool is fixed at L, rising latency directly reduces throughput (`λ = L / W`). That is the mechanism behind "the provider got slower and our backlog exploded."

---

## Drill 9 — Prompt-cache and dedup hit-rate savings

**Problem.** On the premium API ($3 / $15), 2,000 of the 6,000 input tokens are a static rubric prefix. Cached reads bill at 0.10× the input price, and cache writes (on a miss) bill at 1.25×. The prefix hit rate is 90%. Separately, 15% of traces are exact duplicates of already-judged traces under the key `hash(normalized_trace, judge_version, rubric_version)`, and those can be skipped entirely. What are the new daily cost and the savings, starting from $45,000/day?

**Formula.** `effective_prefix_tokens = prefix × (h × 0.10 + (1 − h) × 1.25)`. Then `daily = N × (1 − dedup) × cost_per_call`.

**Math.**
1. Effective prefix tokens: `2,000 × (0.9 × 0.10 + 0.1 × 1.25) = 2,000 × 0.215` = **430**.
2. Effective input tokens: `430 + 4,000` = **4,430**, versus 6,000 before. That is **26.2% less input cost**.
3. Cost per call: `4,430 × 3×10⁻⁶ + 300 × 15×10⁻⁶ = 0.01329 + 0.0045` = **$0.01779** (was $0.0225).
4. Daily cost: `2×10⁶ × 0.85 × 0.01779` = **$30,243**, which is **32.8% below $45,000**.

**Final.** **≈$30.2k/day, saving ≈$14.8k/day (32.8%).** Things that break this in practice:
- The cached prefix must be **byte-identical and come first**, so any per-trace field placed ahead of the rubric (a timestamp or trace ID) drives the hit rate to 0.
- Cache entries expire (TTL), so low-QPS rubrics rarely hit.
- The dedup key must include the judge and rubric versions. Otherwise a rubric change silently serves stale verdicts.

---

## Drill 10 — Batch API discount vs latency

**Problem.** The provider's batch API is 50% cheaper and completes within a 24 h window. Starting from the $30,243/day of Drill 9, what does it cost if you batch everything, versus keeping the 20% of traffic that gates releases on the synchronous API? Production serves 12M requests/day, and a regression is only caught when judge results come back. How many production requests are exposed if detection takes up to 24 h, versus a 5-minute online gate?

**Formula.** `blended = C × (f_sync + (1 − f_sync) × (1 − discount))`. `exposure = daily_requests × detection_delay / 1,440 min`.

**Math.**
1. All batch: `30,243 × 0.5` = **$15,122/day**.
2. Blended with 20% synchronous: `30,243 × (0.2 + 0.8 × 0.5) = 30,243 × 0.6` = **$18,146/day**, a saving of **$12,097/day** (40%).
3. Exposure at a 24 h worst-case delay: **12M requests**, a full day of traffic.
4. Exposure at a 5-minute gate: `12×10⁶ × 5 / 1,440` = **≈41,667 requests**.

**Final.** **Batch the 80% of traffic that feeds offline analysis and keep release gates synchronous. That costs $18.1k/day, versus $15.1k for all-batch.** The extra $3k/day buys a 288× smaller blast radius (12M / 41,667). Check whether the cache and batch discounts stack at your provider before you multiply them. Also note that "within 24 h" is an SLA ceiling, not a median, so plan for the ceiling.

---

## Drill 11 — Corrected v1 Drill 6: null SE, then a clustered-sample design effect

**Problem.** Model A wins 5,300 of 10,000 pairwise comparisons against B (53%, ties excluded). Is A better than B, i.e. is the win rate above 50%? The 10,000 comparisons came from **500 prompts × 20 sampled responses each**. Test under within-prompt intra-class correlation ρ = 0.1 and ρ = 0.3, and under a 200-prompt × 50-response design with ρ = 0.3.

**What v1 got wrong.**
1. **It used the wrong SE for the test.** A hypothesis test against p₀ = 0.5 uses the SE **under the null**, `√(p₀(1−p₀)/n)`. v1 used `√(p̂(1−p̂)/n)`, which belongs to a Wald confidence interval, not a null test. At p̂ = 0.53 the two agree to about 0.2% (0.005000 vs 0.004991), so the verdict was right by luck. At p̂ = 0.70 they differ by about 8%.
2. **It treated clustered samples as independent.** Twenty responses to the same prompt share that prompt's difficulty, so their outcomes are correlated. The effective sample size is `n_eff = n / DEFF`, where `DEFF = 1 + (m − 1)ρ` and m is the cluster size. For unequal cluster sizes, use `1 + ((1 + CV²) · m̄ − 1)ρ`, where CV is the coefficient of variation of the cluster sizes.

**Math.**
1. **Naive (independent) test.**
   - `SE₀ = √(0.25 / 10,000)` = **0.00500**.
   - `z = 0.03 / 0.005` = **6.00**, giving a two-sided p ≈ **2.0×10⁻⁹**.
2. **m = 20, ρ = 0.1.**
   - `DEFF = 1 + 19 × 0.1` = **2.9**, so `n_eff = 3,448`.
   - `SE₀ = √(0.25 / 3,448)` = **0.00851**, so `z = 3.52` and p ≈ **4.3×10⁻⁴**. Still significant.
3. **m = 20, ρ = 0.3.**
   - `DEFF = 6.7`, so `n_eff = 1,493`.
   - `SE₀` = **0.01294**, so `z = 2.32` and p ≈ **0.020**. Marginal.
4. **m = 50, ρ = 0.3 (200 prompts).**
   - `DEFF = 15.7`, so `n_eff = 637`.
   - `SE₀` = **0.01981**, so `z = 1.51` and p ≈ **0.13**. **Not significant.**
5. **95% CI for the win rate (Wald, using p̂ and n_eff).**
   - Naive: 53.0% ± 0.98 pp.
   - m = 20, ρ = 0.1: ± **1.67 pp**.
   - m = 20, ρ = 0.3: ± **2.53 pp**.
   - m = 50, ρ = 0.3: ± **3.88 pp**, so the interval **[49.1%, 56.9%] includes 50%**.

**Final.** The same "5,300 / 10,000" is either overwhelming evidence (z = 6.0) or no evidence (z = 1.5), depending on the sampling design and the ICC. The count of comparisons alone doesn't determine the answer. The correct procedure is:
- Test with the null SE.
- Estimate ρ from the data (or use a cluster bootstrap / cluster-robust SE, resampling **prompts**, not responses).
- Report n_eff.
- For decision-making, prefer a per-prompt estimator: compute a win rate per prompt, then average across prompts.

Also settle the handling of ties and position bias before looking at results: randomize A/B order per comparison and count ties as ½ or exclude them, as preregistered.

---

## Drill 12 — Sample size to detect a 2-point change in pass rate

**Problem.** The baseline agent pass rate is 80%. How many tasks do you need to detect a drop to 78% (α = 0.05 two-sided, power 80%)?
(a) Two independent task sets.
(b) The same tasks run under both versions (paired), with a discordance rate of ψ = 10%.
(c) The paired design, where each task contributes 5 traces with within-task ICC 0.2.

**Formulas.**
- Two-proportion test: `n = [z_{α/2}·√(2p̄q̄) + z_β·√(p₁q₁ + p₂q₂)]² / δ²`.
- Paired (McNemar): `n = [z_{α/2}·√ψ + z_β·√(ψ − δ²)]² / δ²`, where ψ = p₁₀ + p₀₁ is the fraction of tasks where the two versions disagree.

**Math.**
1. (a) Inputs: `p̄ = 0.79`, so `√(2 × 0.79 × 0.21)` = **0.57602**. `√(0.16 + 0.1716)` = **0.57585**.
   - Numerator term: `1.960 × 0.57602 + 0.8416 × 0.57585 = 1.1290 + 0.4846 = 1.6136`, and squared = **2.6038**.
   - `n = 2.6038 / 0.02² = 2.6038 / 0.0004` = **6,510 per arm (13,020 total)**.
2. (b) Inputs: `√0.10 = 0.31623`, and `√(0.10 − 0.0004)` = **0.31559**.
   - Numerator term: `1.960 × 0.31623 + 0.8416 × 0.31559 = 0.6198 + 0.2656 = 0.8854`, and squared = **0.78395**.
   - `n = 0.78395 / 0.0004` = **1,960 tasks**.
3. (c) `DEFF = 1 + (5 − 1) × 0.2` = **1.8**, so `1,960 × 1.8` = **≈3,528 trace-pairs**, i.e. about 706 tasks × 5 traces. This treats the ICC as applying to discordance, an approximation. Confirm by simulation.

**Final.** **6,510 per arm unpaired; 1,960 paired**, a 6.6× reduction in total runs. Always evaluate candidate and baseline on the **same** tasks with the **same** judge version. Also remember that judge noise inflates ψ: a noisier judge raises the paired n directly, so judge reliability is a sample-size cost, not a separate concern. If you check results repeatedly while data accumulates, use a sequential or alpha-spending design. Otherwise the nominal α is fiction.

---

## Drill 13 — pass^k reliability decay

**Problem.** An agent solves a task on a single trial with probability 0.9.
(a) What are pass@8 ("at least one of 8 trials succeeds") and pass^k ("all k trials succeed") for k = 1, 2, 4, 8?
(b) Same mean success, but heterogeneous: half the tasks have p = 1.0 and half p = 0.8. What is pass^8?
(c) What per-trial success rate is needed for pass^8 ≥ 0.9?
(d) Estimate pass^4 from n = 10 trials with c = 9 successes.

**Formulas.**
- Independent trials: `pass@k = 1 − (1 − p)^k` and `pass^k = p^k`.
- Across tasks: `pass^k = E_task[p_i^k]`.
- Unbiased estimator from n trials with c successes: `pass^k ≈ C(c, k) / C(n, k)`.

**Math.**
1. (a) pass@8 = `1 − 0.1⁸` = **0.99999999**. pass^k = **0.900, 0.810, 0.656, 0.430** for k = 1, 2, 4, 8.
2. (b) `0.5 × 1 + 0.5 × 0.8⁸ = 0.5 + 0.5 × 0.1678` = **0.584**, versus 0.430 if every task has p = 0.9. By Jensen's inequality, `E[p^k] ≥ (E[p])^k`.
3. (c) `p ≥ 0.9^{1/8}` = **0.98692**, so the per-trial failure rate must be ≤ 1.31%.
4. (d) `C(9, 4) / C(10, 4) = 126 / 210` = **0.600**. The naive plug-in `0.9⁴ = 0.656` overstates it.

**Final.** At the same 90% single-trial rate, pass@8 is essentially 100% while pass^8 is 43%. For production agents, pass^k is the reliability metric: a user who retries or hits the same workflow repeatedly experiences pass^k. Estimate p **per task**. The pooled p^k is wrong whenever difficulty varies across tasks, and that bias can go either way in comparisons between agents whose difficulty profiles differ.

---

## Drill 14 — Storage for a trace archive

**Problem.** You ingest 2M traces/day. Each trace averages 40 KB of raw JSON (messages, tool calls, spans) plus 2 KB of judge outputs. zstd compresses this JSON about 5×. Retention is 395 days (13 months). Object storage costs $0.023 per GB-month. You also embed each trace as 10 chunks × 1,024-dimensional fp32 vectors for retrieval and clustering. Size the raw archive, the compressed steady state and its cost, and the embeddings.

**Formula.** `daily = N × bytes_per_trace`. `steady_state = daily / compression × retention_days`. Embeddings: `N × chunks × dim × 4 B`.

**Math.**
1. Raw per day: `2×10⁶ × 42×10³` = **84 GB/day**, and `× 365` = **30.7 TB/year**.
2. Compressed: `84 / 5` = **16.8 GB/day**. At steady state, `16.8 × 395` = **6,636 GB ≈ 6.6 TB**.
3. Storage cost: `6,636 × $0.023` = **≈$153/month**.
4. Embeddings: `2×10⁶ × 10 × 1,024 × 4` = **81.9 GB/day**, about the same as the raw traces. They barely compress. Over 395 days that is **32.4 TB**, or about **48.5 TB** once an HNSW index adds its typical ~1.5× overhead.

**Final.** **The compressed archive is ≈6.6 TB and ≈$153/month, which rounds to nothing. The embedding index is ≈48.5 TB, which does not.** The indexed embeddings are about 7× the size of the compressed archive (48.5 / 6.6), and they usually live on RAM or SSD, not object storage, so they dominate storage cost. The levers are fewer chunks, a smaller dimension, fp16/int8 or product quantization (PQ), and embedding only a sample or only failed traces. The other trap is PII: the raw archive needs retention enforcement and deletion-by-user-ID. Deletion propagates to derived artifacts like embeddings and judge outputs only if you design it to.

---

## Drill 15 — Tail latency when fanning out to multiple judges

**Problem.** Each judge call independently has a p99 of 8 s.
(a) You fan out to 5 judges and wait for all of them. What fraction of requests take longer than 8 s? What per-judge percentile would make 8 s the aggregate p99?
(b) Same question when judging each of 100 trace steps in parallel.
(c) With 5 judges and a 3-of-5 quorum, what fraction miss 8 s?

**Formula.** `P(all n finish by t) = F(t)^n`. For a quorum of k of n with per-call on-time probability p: `P(miss) = P(at least n − k + 1 late) = Σ_{j=n−k+1}^{n} C(n, j)(1 − p)^j p^{n−j}`.

**Math.**
1. (a) `0.99⁵ = 0.9510`, so **4.9%** exceed 8 s. The aggregate p99 is not 8 s, it is somewhere near each judge's p99.8, because each judge needs `F(t) = 0.99^{1/5}` = **0.99799**.
2. (b) `0.99¹⁰⁰ = 0.366`, so **63.4%** of requests exceed 8 s. To get a 99% aggregate, each call needs `0.99^{1/100}` = **0.99990** (its p99.99). Even a per-call p99.9 gives only `0.999¹⁰⁰` = **90.5%**.
3. (c) With 3-of-5, a request misses 8 s only if at least 3 judges are late:
   - `C(5,3)(0.01)³(0.99)² + C(5,4)(0.01)⁴(0.99) + (0.01)⁵`
   - `= 9.801×10⁻⁶ + 4.95×10⁻⁸ + 1×10⁻¹⁰`
   - = **9.85×10⁻⁶**, about **1 in 100,000**.

**Final.** Waiting for all judges turns a per-call p99 into a 4.9% (n = 5) or 63.4% (n = 100) tail-exceedance rate. The fixes are:
- **Quorum or first-k aggregation**: 3-of-5 cuts misses about 5,000×. It must be declared in the scoring rule, because it changes which judges' votes count.
- **Hedged requests** after the p95.
- **Per-call deadlines with an explicit "abstain" verdict.**
- **Batching steps into fewer judge calls.**

The independence assumption is optimistic: a provider brownout correlates the tails, so measure the joint tail. Don't multiply marginals and call it an SLO.

---

## Self-check sheet (answers only)

| # | Answer |
|---|---|
| 1 | 140 / 70 / 35 GB (36.1 GB with int4 scales) |
| 2 | 0.328 MB/token; 2.68 GB per 8k sequence; 85.9 GB per batch of 32; 67 sequences on TP=4 |
| 3 | 209 tok/s (8B, batch 1); 95.7 tok/s (70B, TP=4); ≈3,231 tok/s aggregate at batch 32 |
| 4 | Ridge 295 FLOPs/B; KV-capped intensity ≈29.8, so never compute-bound at 4k context |
| 5 | 8.4×10¹⁴ FLOPs; 2.12 s on 1 GPU, 0.53 s on TP=4; attention ≈5.6% |
| 6 | 96 H100s; $5,760/day GPUs, $7,952 with ops; break-even ≈3.85M/day (budget API), ≈112k/day (premium) |
| 7 | TPM binds: 317.5 RPM, ≈457k/day per key; 8.75M TPM needed for 2M/day |
| 8 | 64 per key; 278 fleet; 292 with retries |
| 9 | $30,243/day (−32.8%) |
| 10 | $18,146/day blended; exposure 12M vs ≈41.7k requests |
| 11 | z = 6.0 naive; 3.52, 2.32, 1.51 with design effects 2.9, 6.7, 15.7 |
| 12 | 6,510 per arm; 1,960 paired; ≈3,528 trace-pairs with DEFF 1.8 |
| 13 | pass^8 = 0.430 (iid) vs 0.584 (heterogeneous); need p ≥ 0.9869; estimator 0.600 |
| 14 | 84 GB/day raw; 6.6 TB compressed, ≈$153/month; embeddings ≈48.5 TB |
| 15 | 4.9% (n = 5); 63.4% (n = 100); quorum 9.85×10⁻⁶ |
