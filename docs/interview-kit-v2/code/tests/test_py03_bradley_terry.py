import math
import random

import pytest

from evalkit.bradley_terry import (
    Comparison,
    DisconnectedComparisonGraph,
    fit_bradley_terry,
)


def _simulate(true_log, n_per_pair, rng, tie_rate=0.0):
    items = list(true_log)
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            p = 1 / (1 + math.exp(true_log[b] - true_log[a]))
            for _ in range(n_per_pair):
                if rng.random() < tie_rate:
                    out.append(Comparison(a, b, 0.5))
                else:
                    out.append(Comparison(a, b, 1.0 if rng.random() < p else 0.0))
    return out


def test_recovers_ranking_and_strengths():
    rng = random.Random(0)
    true = {"m1": 1.5, "m2": 0.7, "m3": 0.0, "m4": -0.6, "m5": -1.6}
    res = fit_bradley_terry(_simulate(true, 200, rng), prior_games=0.5)
    assert res.converged
    assert res.ranking() == ["m1", "m2", "m3", "m4", "m5"]
    for k, v in true.items():
        assert res.log_strength[k] == pytest.approx(v, abs=0.2)


def test_orientation_invariance():
    comps = [Comparison("a", "b", 1.0), Comparison("b", "c", 1.0), Comparison("a", "c", 0.0), Comparison("a", "b", 1.0)]
    flipped = [Comparison(c.b, c.a, 1.0 - c.outcome) for c in comps]
    r1, r2 = fit_bradley_terry(comps), fit_bradley_terry(flipped)
    for k in r1.log_strength:
        assert r1.log_strength[k] == pytest.approx(r2.log_strength[k], abs=1e-8)


def test_undefeated_item_is_finite_with_prior_and_rejected_without():
    comps = [Comparison("champ", "x", 1.0)] * 5 + [Comparison("x", "y", 1.0), Comparison("y", "x", 1.0)]
    res = fit_bradley_terry(comps, prior_games=1.0)
    assert math.isfinite(res.log_strength["champ"])
    assert res.ranking()[0] == "champ"
    with pytest.raises(ValueError):
        fit_bradley_terry(comps, prior_games=0.0)


def test_disconnected_graph_refused():
    comps = [Comparison("a", "b", 1.0), Comparison("b", "a", 0.0), Comparison("c", "d", 1.0)]
    with pytest.raises(DisconnectedComparisonGraph):
        fit_bradley_terry(comps)
    fit_bradley_terry(comps, allow_disconnected=True)


def test_ties_pull_strengths_together():
    decisive = [Comparison("a", "b", 1.0)] * 6 + [Comparison("a", "b", 0.0)] * 2
    with_ties = decisive + [Comparison("a", "b", 0.5)] * 8
    gap = lambda r: r.log_strength["a"] - r.log_strength["b"]
    assert 0 < gap(fit_bradley_terry(with_ties)) < gap(fit_bradley_terry(decisive))


def test_win_prob_and_centering():
    rng = random.Random(1)
    res = fit_bradley_terry(_simulate({"a": 1.0, "b": 0.0, "c": -1.0}, 100, rng))
    assert sum(res.log_strength.values()) == pytest.approx(0.0, abs=1e-9)
    assert res.win_prob("a", "c") > 0.8
    assert res.win_prob("a", "c") + res.win_prob("c", "a") == pytest.approx(1.0)


def test_input_validation():
    with pytest.raises(ValueError):
        fit_bradley_terry([])
    with pytest.raises(ValueError):
        fit_bradley_terry([Comparison("a", "a", 1.0)])
    with pytest.raises(ValueError):
        fit_bradley_terry([Comparison("a", "b", 0.7)])
