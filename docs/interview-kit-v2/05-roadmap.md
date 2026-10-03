# 05 — Preparation Roadmap: 2-, 3- and 4-Week Tracks

This plan turns the kit into a schedule with numbers attached. Every week ends in an **exit check** with pass/fail thresholds. If you miss a threshold, repeat the failed component before advancing. Moving on with an unclosed gap is how people go into staff loops with a polished story and a hole in their statistics.

Kit files used here:

| File | Used for |
|---|---|
| [01-evaluation-traps.md](01-evaluation-traps.md) | Oral trap drills (methodology, judges, metrics) |
| [02-system-design-traps.md](02-system-design-traps.md) | Whiteboard design reps |
| [03-arithmetic-drills.md](03-arithmetic-drills.md) | Timed back-of-envelope sprints |
| [04-coding-exercises.md](04-coding-exercises.md) | Python, then Google ADK, then LangGraph |
| [code/](code/) | Reference implementations and tests to run your solutions against |
| [06-methodology-and-behavioral.md](06-methodology-and-behavioral.md) | Research methodology deep dives and behavioral stories |

---

## 1. Choosing a track

| Track | Daily load | Use when |
|---|---|---|
| **2-week** | 3.5–4 h weekdays, 5 h weekend days | The interview is booked and you already ship eval or ML infrastructure. It compresses coding across frameworks and has no slack for repeating a failed week. |
| **3-week** | 2.5–3 h weekdays, 4 h weekend days | The default. One framework per week, with a buffer for gap-closing. |
| **4-week** | 2–2.5 h weekdays, 4 h weekend days | A staff loop, a switch from a non-eval background, or weak statistics or distributed-systems fundamentals. Week 4 is full simulation. |

Pick the track by your weakest area, not your average. If Drill 11 or 12 in [03-arithmetic-drills.md](03-arithmetic-drills.md) takes you more than 10 minutes cold, take the 3- or 4-week track.

---

## 2. Daily drill format (the core block)

Every day of every track uses this block. Heavier days add a second main segment.

| Segment | Time | Content | Output |
|---|---|---|---|
| **A. Arithmetic sprint** | 15 min | 3 drills from [03](03-arithmetic-drills.md), cold, with a timer. Write the assumptions line first. | Time and correctness per drill in the error log |
| **B. Trap round** | 20 min | 2 traps from [01](01-evaluation-traps.md) or [02](02-system-design-traps.md), answered **aloud** and recorded, 3 min each. Then compare against the kit's answer. | Rubric score (section 6) per answer |
| **C. Main segment** | 60–90 min | That day's system design, coding, or methodology item (per the schedule) | Artifact: design doc, passing tests, or a written answer |
| **D. Adversarial review** | 15 min | Re-read the day's output as a hostile interviewer. Name 3 follow-up questions you could not answer crisply. | 3 entries in the error log |
| **E. Error-log close** | 10 min | Classify each miss as *concept*, *number*, *omission* or *communication*, and schedule a retry in 2 days and again in 7 days. | Updated retry queue |

**Error log.** One row per miss: `date | source (file#item) | miss type | what I said | what is correct | retry dates`. The next day's segments A and B draw from the retry queue first.

**Rules.**
- Segment A is closed-book and uses no calculator beyond pen and paper. State units and approximations before computing.
- Segment B is spoken, not written. Written answers hide filler and missing structure.
- Segment C coding runs against tests. "It looks right" doesn't count.

---

## 3. 2-Week Track

Coding compresses to Python (days 1–4), then Google ADK (days 5–7), then LangGraph (days 8–10).

| Day | Arithmetic (A) | Traps (B) | Main segment (C) |
|---|---|---|---|
| 1 | Drills 1–3 | 01: first 2 traps | Python exercise 1 from [04](04-coding-exercises.md), core logic plus tests |
| 2 | Drills 4–6 | 01: next 2 | Python exercise 1: edge cases and failure injection; compare with [code/](code/) |
| 3 | Drills 7–9 | 02: first 2 | System design: judge pipeline at 2M traces/day (reuse the Drill 6–10 numbers) |
| 4 | Drills 10–12 | 01: next 2 | Python exercise 2 from [04](04-coding-exercises.md) |
| 5 | Drills 13–15 | 02: next 2 | ADK exercise 1: agent, tools, session state |
| 6 | Retry queue | 06: 2 methodology prompts | ADK exercise 1 hardening, then **Mock #1 (design, 45 min)** |
| 7 | Mixed 5, timed | 01 + 02 random | ADK exercise 2; **Week 1 exit check** |
| 8 | Retry queue | 02: next 2 | LangGraph exercise 1: state schema, conditional edges, checkpointing |
| 9 | Mixed 3 | 06: behavioral stories (write 4) | LangGraph exercise 1: retries, interrupts, resume from checkpoint |
| 10 | Mixed 3 | 01 random | LangGraph exercise 2; **Mock #2 (coding, 60 min)** |
| 11 | Drills 11–12 cold | 06: methodology | System design: online eval and canary gating, with a stats defense |
| 12 | Mixed 5, timed | Random 4 | **Mock #3 (methodology / research, 45 min)** |
| 13 | Retry queue | Random 4 | **Mock #4 (full loop: design + coding + behavioral, 2.5 h)** |
| 14 | 15-drill sweep | Weakest 4 | Gap closing, then the **Week 2 exit check**. Rest the evening before the interview. |

---

## 4. 3-Week Track (default)

Each week has one coding framework: **Week 1 Python, Week 2 Google ADK, Week 3 LangGraph**. Every weekday runs the core block. Weekends run two main segments.

### Week 1 — Foundations: statistics, judges, Python

| Day | A | B | C |
|---|---|---|---|
| Mon | Drills 1–3 | 01 ×2 | Python exercise 1 |
| Tue | Drills 4–5 | 01 ×2 | Python exercise 1: tests and failure injection |
| Wed | Drills 11–12 | 01 ×2 | Methodology: judge validation, ICC and design effects ([06](06-methodology-and-behavioral.md)) |
| Thu | Drill 13 + retry | 02 ×2 | Python exercise 2 |
| Fri | Drills 6–7 | 01 ×2 | System design #1 ([02](02-system-design-traps.md)) |
| Sat | Mixed 5 | Retry | Python exercise 2 hardening, then **Mock #1 (methodology, 45 min)** |
| Sun | Retry | — | **Week 1 exit check**, then plan Week 2 from the error log |

### Week 2 — Systems: serving, cost, reliability, ADK

| Day | A | B | C |
|---|---|---|---|
| Mon | Drills 8–9 | 02 ×2 | ADK exercise 1 |
| Tue | Drill 10 + retry | 02 ×2 | ADK exercise 1: tool errors, state, determinism in tests |
| Wed | Drills 14–15 | 01 ×2 | System design #2: self-hosted vs API judge fleet (defend with Drill 6) |
| Thu | Mixed 3 | 02 ×2 | ADK exercise 2 |
| Fri | Mixed 3 | 06: behavioral (draft 4 stories) | System design #3: trace ingestion and archive (Drill 14) |
| Sat | Mixed 5, timed | Retry | **Mock #2 (system design, 60 min)**, then **Mock #3 (ADK coding, 60 min)** |
| Sun | Retry | — | **Week 2 exit check** |

### Week 3 — Integration: LangGraph and full loops

| Day | A | B | C |
|---|---|---|---|
| Mon | Mixed 3 | 01 ×2 | LangGraph exercise 1 |
| Tue | Mixed 3 | 02 ×2 | LangGraph exercise 1: checkpointing, resume, interrupts |
| Wed | Drills 11–12 cold | 06 methodology | LangGraph exercise 2 |
| Thu | Retry | Random 4 | **Mock #4 (LangGraph coding, 60 min)** |
| Fri | Mixed 5 | 06 behavioral (finalize 6 stories) | **Mock #5 (full loop, 2.5–3 h)** |
| Sat | 15-drill sweep | Weakest 4 | Gap closing on the two lowest rubric dimensions |
| Sun | Light | — | **Week 3 exit check**, then rest |

---

## 5. 4-Week Track

Weeks 1–3 are the 3-week track, with two additions:
- An extra 30-minute segment each weekday for reading and re-deriving: each week, derive from scratch the formulas behind 2 drills and 1 methodology topic from [06](06-methodology-and-behavioral.md).
- Weekend main segments get a second design problem.

### Week 4 — Simulation and calibration

| Day | Content |
|---|---|
| Mon | **Mock #6: full loop** (design 60 + coding 60 + methodology 45 + behavioral 30), with a different mock interviewer if possible |
| Tue | Postmortem. Rebuild the 2 weakest answers from scratch. Arithmetic retry queue. |
| Wed | **Mock #7: "staff escalation" design.** The interviewer injects a mid-interview constraint change: 10× volume, a halved budget, or the provider's rate limit cut in half. |
| Thu | Coding speed round: one exercise per framework from [04](04-coding-exercises.md), 40 min each, tests from [code/](code/) |
| Fri | **Mock #8: methodology adversarial.** The interviewer attacks every statistical claim: SE, clustering, power, judge bias. |
| Sat | **Mock #9: full loop**, under exact interview-day conditions |
| Sun | **Week 4 exit check**, then taper |

---

## 6. Exit criteria (measurable)

Thresholds apply to every track, mapped to its weeks. The 2-week track checks the Week 1 criteria on day 7 and the combined Week 2 + Week 3 criteria on day 14. Criteria marked "cold" mean no notes and no re-reading beforehand.

### Week 1 exit
- **Arithmetic:** Drills 1–5 and 11–13 cold, **≥ 7/8 correct within 5%**, median **≤ 4 min** per drill. On Drill 11 you state the null SE *and* the design effect unprompted.
- **Traps:** 6 random traps from [01](01-evaluation-traps.md), aloud, 3 min each, with **mean rubric ≥ 2.5 and no answer below 2**.
- **Coding (Python):** 2 exercises from [04](04-coding-exercises.md) pass their tests in [code/](code/) (or your equivalent), and each includes **≥ 2 failure-injection tests**. The second exercise finishes in **≤ 60 min**.
- **Methodology:** In 5 minutes, explain why 5,300/10,000 can be non-significant, with the numbers.

### Week 2 exit
- **Arithmetic:** all 15 drills cold, **≥ 13/15 within 5%**, median **≤ 3.5 min**. For each drill, you name the binding constraint without being asked.
- **System design:** 2 problems from [02](02-system-design-traps.md), 45 min each, scoring **≥ 3 on "Quantification" and "Failure handling"**. Each design includes capacity math, a degradation path and a rollback path.
- **Coding (ADK):** 2 ADK exercises working with tests, where tool failure, timeout and malformed tool output are all handled and tested.
- **Mock:** design mock score **≥ 2.75 mean, no dimension below 2**.

### Week 3 exit
- **Arithmetic:** 5 random drills with changed inputs (for example 405B instead of 70B, or a 3M-TPM limit), **5/5 within 5%**, **≤ 3 min** each.
- **Coding (LangGraph):** 2 exercises with typed state, a conditional route, a retry/escalation branch and checkpoint resume, each demonstrated by a test that kills and resumes the run.
- **Behavioral:** 6 STAR stories, each ≤ 2.5 min, each with a **quantified outcome** and a **decision you would make differently**.
- **Full-loop mock:** **mean ≥ 3.0, no dimension below 2.5.**

### Week 4 exit (4-week track only)
- **3 consecutive full-loop mocks at mean ≥ 3.25** with no dimension below 3.
- **Zero repeated error-log misses** in the final 5 days. Every miss is new, which means old gaps are closed.
- In the constraint-change mock, you re-derive the capacity and cost numbers **within 5 minutes** of the change.

---

## 7. Self-scoring rubric

Score each answer or mock on 6 dimensions from 0 to 4. Record it immediately, not from memory later. The staff bar is a mean of **≥ 3.0 with no dimension below 2.5**. The senior bar is a mean of **≥ 2.5 with no dimension below 2**.

| Dimension | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **Correctness** | Wrong core claim | Right idea, wrong mechanics or numbers | Correct with minor slips | Correct and precise | Correct, and catches the interviewer's implicit error |
| **Quantification** | No numbers | Numbers without units or assumptions | Numbers with assumptions stated | Numbers plus the binding constraint identified | Sensitivity: says which input flips the decision, and at what value |
| **Uncertainty and statistics** | Point estimates only | Mentions a CI or noise vaguely | Correct SE or CI for iid data | Handles clustering, pairing and multiple comparisons | Designs the experiment (power, sequential testing, judge noise) before collecting data |
| **Failure handling** | Happy path only | Lists failures without mitigations | Mitigates the obvious ones (retry, timeout) | Idempotency, backpressure, degradation, rollback | Tests for it (failure injection, replay) and names the residual risk |
| **Trade-off reasoning** | One option | Options without criteria | Criteria, but no decision | A decision with explicit criteria and the cost of being wrong | A decision plus the trigger that would reverse it |
| **Communication** | Unstructured | Structured, but rambling or over time | Clear, on time | Leads with the answer, then the support | Adapts depth to the interviewer's probes and checks alignment |

**Drill scoring (segment A).** A drill is correct if the answer is within 5% *and* the assumptions line is present. A correct number with no assumptions counts as half. Track the median time per drill each week; it should fall about 25% per week.

**Mock protocol.**
- Record every mock.
- Score it from the recording within 24 h.
- Convert the lowest two dimensions into the next 3 days' segment B and C content.
- If a mock interviewer is available, have them score blind with the same rubric. A gap of more than 0.5 between their score and yours on any dimension means your self-assessment is miscalibrated there, and the same measurement problem as an uncalibrated judge applies to you.

---

## 8. Failure modes of preparation itself

- **Reading instead of producing.** If a day ends without a scored artifact, it didn't count.
- **Drilling the same numbers.** By Week 2, change the inputs to the [03](03-arithmetic-drills.md) drills. Memorized answers are not fluency.
- **Skipping the kill-and-resume test** in LangGraph and ADK work. Durability is the staff signal in those rounds.
- **Mocking only your strengths.** Schedule mocks by lowest rubric dimension, not by preference.
