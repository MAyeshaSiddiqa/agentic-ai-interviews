"""Exercise PY-3: Bradley-Terry ranking from pairwise judge verdicts.

P(i beats j) = p_i / (p_i + p_j). Fitted by Newton-Raphson in log-strength.
A weak prior (virtual games against a phantom item of strength 1) keeps
undefeated or winless items finite and fixes the scale.
"""

from __future__ import annotations

import math

import numpy as np
from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Comparison:
    a: Hashable
    b: Hashable
    outcome: float  # 1.0 = a wins, 0.0 = b wins, 0.5 = tie


@dataclass(frozen=True)
class BTResult:
    log_strength: dict[Hashable, float]  # centred: mean over items = 0
    iterations: int
    converged: bool

    def ranking(self) -> list[Hashable]:
        return sorted(self.log_strength, key=lambda k: (-self.log_strength[k], str(k)))

    def win_prob(self, i: Hashable, j: Hashable) -> float:
        return 1.0 / (1.0 + math.exp(self.log_strength[j] - self.log_strength[i]))


class DisconnectedComparisonGraph(ValueError):
    pass


def fit_bradley_terry(
    comparisons: Iterable[Comparison],
    *,
    prior_games: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-10,
    allow_disconnected: bool = False,
) -> BTResult:
    comps = list(comparisons)
    if not comps:
        raise ValueError("no comparisons")
    wins: dict[Hashable, float] = defaultdict(float)
    games: dict[tuple[Hashable, Hashable], float] = defaultdict(float)
    items: set[Hashable] = set()
    for c in comps:
        if c.a == c.b:
            raise ValueError(f"self-comparison for {c.a!r}")
        if c.outcome not in (0.0, 0.5, 1.0):
            raise ValueError(f"outcome must be 0, 0.5 or 1, got {c.outcome}")
        items.update((c.a, c.b))
        # A tie counts as half a win each way. The principled alternative is
        # Davidson's tie model; discarding ties biases toward decisive pairs.
        wins[c.a] += c.outcome
        wins[c.b] += 1.0 - c.outcome
        key = (c.a, c.b) if str(c.a) <= str(c.b) else (c.b, c.a)
        games[key] += 1.0

    _check_connected(items, games, allow_disconnected)
    if prior_games <= 0 and any(wins[i] == 0 or wins[i] == _n_games(i, games) for i in items):
        raise ValueError("an item is undefeated or winless: MLE diverges; use prior_games > 0")

    # Newton-Raphson on the penalised log-likelihood in theta = log p. Hunter's
    # MM updates are simpler but converge linearly, and very slowly along the
    # common-scale direction when the prior is weak relative to the data.
    order = sorted(items, key=str)
    ix = {k: n for n, k in enumerate(order)}
    pa = np.array([ix[a] for a, _ in games])
    pb = np.array([ix[b] for _, b in games])
    n_ab = np.array(list(games.values()))
    w = np.array([wins[k] for k in order]) + prior_games / 2.0
    m = len(order)

    def loglik(th):
        # sum over games of log P(winner) = sum_i w_i th_i - sum_pairs n log(e^a + e^b);
        # the phantom (theta = 0) contributes prior_games games per item.
        ll = float(w @ th - (n_ab * np.logaddexp(th[pa], th[pb])).sum())
        return ll - float(prior_games * np.logaddexp(0.0, th).sum())

    th = np.zeros(m)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        s = 1.0 / (1.0 + np.exp(-(th[pa] - th[pb])))  # P(a beats b)
        sp = 1.0 / (1.0 + np.exp(-th))  # P(item beats phantom)
        grad = w - prior_games * sp
        np.subtract.at(grad, pa, n_ab * s)
        np.subtract.at(grad, pb, n_ab * (1.0 - s))
        hess = np.zeros((m, m))
        v = n_ab * s * (1.0 - s)
        np.add.at(hess, (pa, pa), -v)
        np.add.at(hess, (pb, pb), -v)
        np.add.at(hess, (pa, pb), v)
        np.add.at(hess, (pb, pa), v)
        hess[np.diag_indices(m)] -= prior_games * sp * (1.0 - sp)
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        base, t = loglik(th), 1.0
        while loglik(th - t * step) < base - 1e-12 and t > 1e-8:
            t /= 2.0
        th = th - t * step
        if prior_games <= 0:
            th -= th.mean()
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break

    th = th - th.mean()
    return BTResult({k: float(th[ix[k]]) for k in order}, it, converged)


def _n_games(i, games) -> float:
    return sum(n for (a, b), n in games.items() if i in (a, b))


def _check_connected(items, games, allow: bool) -> None:
    adj: dict[Hashable, set] = defaultdict(set)
    for a, b in games:
        adj[a].add(b)
        adj[b].add(a)
    start = next(iter(items))
    seen, stack = {start}, [start]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if len(seen) != len(items) and not allow:
        # Across components, relative strengths are set only by the prior,
        # i.e. by nothing observed. Refuse to rank them silently.
        raise DisconnectedComparisonGraph(
            f"comparison graph has >1 component ({len(seen)} of {len(items)} items reachable)"
        )
