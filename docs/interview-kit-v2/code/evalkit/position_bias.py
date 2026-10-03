"""Exercise PY-6: probe a pairwise judge for position bias by swapping order.

Each pair is judged twice, (A, B) and (B, A). Verdicts are mapped back to the
canonical frame before comparison. The probe reports the swap-consistency
rate, P(judge picks the first slot) with a Wilson interval and an exact
binomial test against 0.5, and a debiased verdict per pair (consistent
verdict, otherwise tie).
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

Slot = Literal["first", "second", "tie"]
Canon = Literal["A", "B", "tie"]
PairJudge = Callable[[str, str, str], Awaitable[Slot]]  # (question, first, second) -> slot


@dataclass(frozen=True)
class Pair:
    pair_id: str
    question: str
    a: str
    b: str


@dataclass(frozen=True)
class PairProbe:
    pair_id: str
    ab: Canon  # verdict with A shown first, in canonical frame
    ba: Canon  # verdict with B shown first, in canonical frame
    consistent: bool
    debiased: Canon


@dataclass(frozen=True)
class BiasReport:
    n_pairs: int
    consistency: float
    first_slot_rate: float  # over all non-tie slot decisions, both orders
    first_slot_ci: tuple[float, float]
    p_value: float  # two-sided exact binomial vs 0.5
    n_decisions: int
    probes: tuple[PairProbe, ...]


def _to_canon(slot: Slot, first_is_a: bool) -> Canon:
    if slot == "tie":
        return "tie"
    if slot not in ("first", "second"):
        raise ValueError(f"judge returned unparseable slot {slot!r}")
    return "A" if (slot == "first") == first_is_a else "B"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    return min(1.0, sum(x for x in pmf if x <= obs * (1 + 1e-7)))


async def probe_position_bias(judge: PairJudge, pairs: Sequence[Pair]) -> BiasReport:
    """Both orders must be real, independent calls. If the judge sits behind a
    cache, the order has to be part of the cache key, or the swapped call
    returns the unswapped verdict and the probe reports perfect consistency."""
    probes: list[PairProbe] = []
    first_picks = decisions = 0
    for p in pairs:
        s_ab = await judge(p.question, p.a, p.b)
        s_ba = await judge(p.question, p.b, p.a)
        for s in (s_ab, s_ba):
            if s != "tie":
                decisions += 1
                first_picks += s == "first"
        ab, ba = _to_canon(s_ab, True), _to_canon(s_ba, False)
        consistent = ab == ba
        probes.append(PairProbe(p.pair_id, ab, ba, consistent, ab if consistent else "tie"))
    n = len(probes)
    return BiasReport(
        n_pairs=n,
        consistency=sum(x.consistent for x in probes) / n if n else math.nan,
        first_slot_rate=first_picks / decisions if decisions else math.nan,
        first_slot_ci=wilson(first_picks, decisions),
        p_value=binom_two_sided(first_picks, decisions),
        n_decisions=decisions,
        probes=tuple(probes),
    )
