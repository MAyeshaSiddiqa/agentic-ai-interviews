# Staff/Senior Agent-Evaluation Interview Kit — v2

A preparation kit for **Staff/Senior Applied Scientist and Research Scientist** roles on AI-agent evaluation platforms: LLM-as-a-judge pipelines over large volumes of agent traces, production LLM APIs, and the cost, latency and statistical trade-offs that come with them. The standard is simple: every claim can be defended with a number, a failure mode, or a test.

v2 supersedes [v1](../staff-senior-agent-eval-interview-kit.md). Among other changes, it adds hardware and serving arithmetic and corrects v1's Drill 6. v1 used the observed-proportion SE for a test against 50% and treated clustered samples as independent. The corrected drill is Drill 11 in [03-arithmetic-drills.md](03-arithmetic-drills.md).

## Contents

| File | Description |
|---|---|
| [01-evaluation-traps.md](01-evaluation-traps.md) | Conceptual traps in evaluation methodology and LLM-as-a-judge, each with the interviewer's scenario, a plausible mid-level answer and the staff-level analysis. |
| [02-system-design-traps.md](02-system-design-traps.md) | System design scenarios for eval platforms, each with the hidden bottleneck and a fault-tolerant, quantified solution. |
| [03-arithmetic-drills.md](03-arithmetic-drills.md) | 15 back-of-envelope drills covering VRAM, KV cache, the bandwidth and compute roofline, fleet cost, API break-even, rate limits, Little's law, caching, statistics (null SE, design effect, power, pass^k), storage and fan-out tail latency. Each has fully worked math. |
| [04-coding-exercises.md](04-coding-exercises.md) | Coding exercises in Python, Google ADK and LangGraph, focused on reliability, evaluation infrastructure and testability. |
| [05-roadmap.md](05-roadmap.md) | 2-, 3- and 4-week preparation tracks with a daily drill format, weekly schedules, measurable exit criteria and a self-scoring rubric. |
| [06-methodology-and-behavioral.md](06-methodology-and-behavioral.md) | Research methodology deep dives and behavioral preparation for staff-level loops. |
| [code/](code/) | Reference implementations and tests for the coding exercises. |

## How to use this kit

1. **Pick a track** in [05-roadmap.md](05-roadmap.md), choosing by your weakest area. Take a cold baseline first: 5 drills from [03](03-arithmetic-drills.md) and 3 traps from [01](01-evaluation-traps.md), scored with the rubric in section 7 of the roadmap.
2. **Run the daily block** from the roadmap: an arithmetic sprint, spoken trap answers, one main segment (design, coding or methodology), an adversarial review, and an error-log update.
3. **Produce, don't read.** Answer traps aloud before you read the kit's answer. Code against the tests in [code/](code/) before you look at the reference solutions.
4. **Gate on the exit criteria.** Don't advance a week until its thresholds pass. Repeat the failed component instead.
5. **Change the inputs.** From Week 2 on, re-run the drills with different model sizes, volumes, prices and limits, so you are practicing the method rather than recalling the answers.

## Conventions

- Decimal units (1 GB = 10⁹ B). Hardware reference: H100 SXM (80 GB, 3.35 TB/s, 989 TFLOP/s dense BF16).
- Prices in the kit are **hypothetical** and stated as parameters. In an interview, parameterize and don't quote vendor prices from memory.
- Framework APIs (Google ADK, LangGraph) change between versions. Check the primitives against the version you are using.
