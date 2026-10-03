"""Exercise PY-1: chance-corrected agreement between an LLM judge and humans.

cohen_kappa: two raters, paired labels, optional linear/quadratic weights.
krippendorff_alpha: any number of raters, missing data, and
nominal/ordinal/interval/ratio distance metrics.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from itertools import permutations
from typing import Literal

Level = Literal["nominal", "ordinal", "interval", "ratio"]
Weights = Literal["linear", "quadratic"] | None


def cohen_kappa(
    rater_a: Sequence[Hashable | None],
    rater_b: Sequence[Hashable | None],
    *,
    labels: Sequence[Hashable] | None = None,
    weights: Weights = None,
) -> float:
    """Cohen's kappa over items both raters labelled.

    Pairs with a missing label (None) are dropped instead of treated as a
    category. Returns NaN when expected disagreement is zero (one rater is
    constant on the same single label as the other), because kappa is
    undefined there. Reporting 0 or 1 in that case would be wrong.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same items (equal length)")
    pairs = [(a, b) for a, b in zip(rater_a, rater_b) if a is not None and b is not None]
    if not pairs:
        raise ValueError("no items labelled by both raters")

    if labels is None:
        cats = sorted({x for p in pairs for x in p}, key=_sort_key)
    else:
        cats = list(labels)
        unknown = {x for p in pairs for x in p} - set(cats)
        if unknown:
            raise ValueError(f"labels outside the declared label set: {unknown}")
    # Weighted kappa uses the positional order of `cats`; for string labels
    # that order is only meaningful if the caller supplied it.
    if weights is not None and labels is None and not all(isinstance(c, (int, float)) for c in cats):
        raise ValueError("weighted kappa on non-numeric labels requires explicit ordered `labels`")

    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    n = len(pairs)
    obs = [[0.0] * k for _ in range(k)]
    for a, b in pairs:
        obs[idx[a]][idx[b]] += 1.0 / n
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if weights is None:
            return 0.0 if i == j else 1.0
        d = abs(i - j) / max(k - 1, 1)
        return d if weights == "linear" else d * d

    d_obs = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    d_exp = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k))
    if d_exp == 0.0:
        return math.nan
    return 1.0 - d_obs / d_exp


def krippendorff_alpha(
    units: Sequence[Sequence[float | Hashable | None]],
    *,
    level: Level = "nominal",
) -> float:
    """Krippendorff's alpha.

    `units` is a list of units (items); each unit lists the values assigned by
    each rater, with None for "rater did not label this unit". Units with fewer
    than two values are unpairable and are excluded, as the coefficient
    requires. Returns NaN when there is no variation in the pairable values.
    """
    pairable = [[v for v in u if v is not None] for u in units]
    pairable = [u for u in pairable if len(u) >= 2]
    if not pairable:
        raise ValueError("no unit has two or more ratings")

    # Coincidence matrix: each ordered pair of values within a unit
    # contributes 1/(m_u - 1).
    coinc: Counter[tuple[Hashable, Hashable]] = Counter()
    for u in pairable:
        m = len(u)
        for i, j in permutations(range(m), 2):
            coinc[(u[i], u[j])] += 1.0 / (m - 1)

    values = sorted({v for u in pairable for v in u}, key=_sort_key)
    n_c = {c: sum(coinc[(c, k)] for k in values) for c in values}
    n = sum(n_c.values())
    if len(values) < 2:
        return math.nan

    delta2 = _distance(level, values, n_c)
    d_o = sum(coinc[(c, k)] * delta2(c, k) for c in values for k in values)
    d_e = sum(n_c[c] * n_c[k] * delta2(c, k) for c in values for k in values) / (n - 1)
    if d_e == 0.0:
        return math.nan
    return 1.0 - d_o / d_e


def _distance(level: Level, values: list, n_c: dict):
    if level == "nominal":
        return lambda c, k: 0.0 if c == k else 1.0
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        raise ValueError(f"level={level!r} requires numeric values")
    if level == "interval":
        return lambda c, k: float(c - k) ** 2
    if level == "ratio":
        if any(v < 0 for v in values):
            raise ValueError("ratio level requires non-negative values")
        return lambda c, k: 0.0 if c == k else ((c - k) / (c + k)) ** 2
    if level == "ordinal":
        # Distance is the mass of values between c and k in the pooled
        # marginal, so it depends on the observed distribution, not the
        # numeric gaps.
        rank = {v: i for i, v in enumerate(values)}
        cum = [0.0]
        for v in values:
            cum.append(cum[-1] + n_c[v])

        def d(c, k):
            if c == k:
                return 0.0
            lo, hi = sorted((rank[c], rank[k]))
            between = cum[hi + 1] - cum[lo]
            return (between - (n_c[values[lo]] + n_c[values[hi]]) / 2.0) ** 2

        return d
    raise ValueError(f"unknown level {level!r}")


def _sort_key(x):
    return (0, x) if isinstance(x, (int, float)) and not isinstance(x, bool) else (1, str(x))
