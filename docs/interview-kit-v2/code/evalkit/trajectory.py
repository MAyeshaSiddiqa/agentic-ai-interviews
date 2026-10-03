"""Exercise PY-4: trajectory matching for agent tool calls.

Modes:
  exact      actual == expected, step for step
  in_order   expected is a subsequence of actual (extra calls allowed)
  any_order  every expected step matched to a distinct actual step

A reference may list several acceptable trajectories (alternative valid
paths), and a single step may accept several calls (AnyOf). Forbidden tools
fail the match in every mode, closing the in_order/any_order loophole where an
extra destructive call still "matches".
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Mode = Literal["exact", "in_order", "any_order"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)


ANY = object()  # wildcard argument value


@dataclass(frozen=True)
class ArgPolicy:
    ignore: frozenset[str] = frozenset()  # e.g. request ids, timestamps
    float_tol: float = 1e-9
    casefold_strings: bool = False
    allow_extra_args: bool = False


@dataclass(frozen=True)
class Step:
    """Expected step: one of several acceptable calls."""

    options: tuple[ToolCall, ...]

    @staticmethod
    def of(*calls: ToolCall) -> "Step":
        if not calls:
            raise ValueError("Step needs at least one option")
        return Step(tuple(calls))


def AnyOf(*calls: ToolCall) -> Step:
    return Step.of(*calls)


Expected = Sequence[ToolCall | Step]


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    score: float  # fraction of expected steps matched, best alternative
    mode: Mode
    alternative: int | None  # index of best-matching reference path
    missing: tuple[int, ...] = ()
    forbidden_hits: tuple[int, ...] = ()
    first_divergence: int | None = None


def _values_equal(exp: Any, act: Any, pol: ArgPolicy) -> bool:
    if exp is ANY:
        return True
    if callable(exp) and not isinstance(exp, type):
        return bool(exp(act))
    if isinstance(exp, bool) or isinstance(act, bool):
        return exp is act if isinstance(exp, bool) and isinstance(act, bool) else False
    if isinstance(exp, (int, float)) and isinstance(act, (int, float)):
        if isinstance(exp, float) and math.isnan(exp):
            return isinstance(act, float) and math.isnan(act)
        return math.isclose(exp, act, rel_tol=0.0, abs_tol=pol.float_tol) or exp == act
    if isinstance(exp, str) and isinstance(act, str) and pol.casefold_strings:
        return exp.casefold() == act.casefold()
    if isinstance(exp, Mapping) and isinstance(act, Mapping):
        return _args_match(exp, act, pol)
    if isinstance(exp, (list, tuple)) and isinstance(act, (list, tuple)):
        return len(exp) == len(act) and all(_values_equal(e, a, pol) for e, a in zip(exp, act))
    return exp == act


def _args_match(exp: Mapping, act: Mapping, pol: ArgPolicy) -> bool:
    ek = set(exp) - pol.ignore
    ak = set(act) - pol.ignore
    if not ek <= ak or (not pol.allow_extra_args and ak - ek):
        return False
    return all(_values_equal(exp[k], act[k], pol) for k in ek)


def call_matches(step: ToolCall | Step, act: ToolCall, pol: ArgPolicy) -> bool:
    opts = step.options if isinstance(step, Step) else (step,)
    return any(o.name == act.name and _args_match(o.args, act.args, pol) for o in opts)


def _exact(exp: Expected, act: Sequence[ToolCall], pol: ArgPolicy):
    div = None
    hits = 0
    for i in range(max(len(exp), len(act))):
        ok = i < len(exp) and i < len(act) and call_matches(exp[i], act[i], pol)
        hits += ok
        if not ok and div is None:
            div = i
    missing = tuple(i for i in range(len(exp)) if i >= len(act) or not call_matches(exp[i], act[i], pol))
    return div is None, (hits / len(exp) if exp else float(not act)), missing, div


def _in_order(exp: Expected, act: Sequence[ToolCall], pol: ArgPolicy):
    # LCS-style DP gives partial credit; full match iff DP value == len(exp).
    n, m = len(exp), len(act)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + call_matches(exp[i - 1], act[j - 1], pol))
    best = dp[n][m]
    matched_idx, i, j = set(), n, m
    while i > 0 and j > 0:
        if call_matches(exp[i - 1], act[j - 1], pol) and dp[i][j] == dp[i - 1][j - 1] + 1:
            matched_idx.add(i - 1)
            i, j = i - 1, j - 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    missing = tuple(k for k in range(n) if k not in matched_idx)
    return best == n, (best / n if n else 1.0), missing, (missing[0] if missing else None)


def _any_order(exp: Expected, act: Sequence[ToolCall], pol: ArgPolicy):
    # Maximum bipartite matching (Kuhn). Greedy assignment is wrong once steps
    # carry wildcards/predicates: exp [search(q=ANY), search(q="x")] vs
    # act [search(q="x"), search(q="y")] is a full match greedy misses.
    adj = [[j for j, a in enumerate(act) if call_matches(e, a, pol)] for e in exp]
    owner = [-1] * len(act)

    def augment(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1 or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    size = sum(augment(i, set()) for i in range(len(exp)))
    matched = {i for i in owner if i != -1}
    missing = tuple(k for k in range(len(exp)) if k not in matched)
    n = len(exp)
    return size == n, (size / n if n else 1.0), missing, (missing[0] if missing else None)


_MODES: dict[str, Callable] = {"exact": _exact, "in_order": _in_order, "any_order": _any_order}


def match_trajectory(
    actual: Sequence[ToolCall],
    references: Sequence[Expected],
    *,
    mode: Mode = "in_order",
    arg_policy: ArgPolicy = ArgPolicy(),
    forbidden: Callable[[ToolCall], bool] | frozenset[str] = frozenset(),
) -> MatchResult:
    """Best match of `actual` against any of the acceptable `references`.

    An empty reference means "no tool call should happen": under in_order and
    any_order it would otherwise match everything vacuously, so it is treated
    as exact.
    """
    if not references:
        raise ValueError("at least one reference trajectory is required")
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}")
    is_forbidden = forbidden if callable(forbidden) else (lambda c: c.name in forbidden)
    hits = tuple(i for i, c in enumerate(actual) if is_forbidden(c))

    best: MatchResult | None = None
    for k, ref in enumerate(references):
        fn = _exact if not ref else _MODES[mode]
        ok, score, missing, div = fn(ref, actual, arg_policy)
        ok = ok and not hits
        res = MatchResult(ok, score if not hits else 0.0, mode, k, missing, hits, div)
        if best is None or (res.matched, res.score) > (best.matched, best.score):
            best = res
    assert best is not None
    return best
