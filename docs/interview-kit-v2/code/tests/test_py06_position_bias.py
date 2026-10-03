import random

import pytest

from evalkit.position_bias import Pair, binom_two_sided, probe_position_bias, wilson


def _pairs(n, rng):
    """Answer quality is encoded as 'q=<float>'; half the pairs are close calls."""
    out = []
    for i in range(n):
        qa = rng.random()
        qb = qa + rng.choice([-1, 1]) * (0.02 if i % 2 else 0.5)
        out.append(Pair(f"p{i}", "question", f"q={qa:.3f}", f"q={qb:.3f}"))
    return out


def _q(s):
    return float(s.split("=")[1])


async def fair_judge(question, first, second):
    d = _q(first) - _q(second)
    return "tie" if abs(d) < 1e-9 else ("first" if d > 0 else "second")


def make_biased(rng, p_first_on_close=0.9):
    async def judge(question, first, second):
        d = _q(first) - _q(second)
        if abs(d) < 0.1:
            return "first" if rng.random() < p_first_on_close else "second"
        return "first" if d > 0 else "second"

    return judge


async def test_fair_judge_is_consistent_and_unbiased():
    rep = await probe_position_bias(fair_judge, _pairs(200, random.Random(0)))
    assert rep.consistency == 1.0
    assert rep.first_slot_rate == pytest.approx(0.5)
    assert rep.p_value == pytest.approx(1.0)


async def test_biased_judge_detected():
    rep = await probe_position_bias(make_biased(random.Random(1)), _pairs(200, random.Random(0)))
    assert rep.consistency < 0.7
    assert rep.first_slot_rate > 0.6
    assert rep.first_slot_ci[0] > 0.5
    assert rep.p_value < 1e-6


async def test_debiased_verdict_keeps_clear_wins_and_ties_flips():
    rep = await probe_position_bias(make_biased(random.Random(2)), _pairs(100, random.Random(3)))
    pairs = {p.pair_id: p for p in _pairs(100, random.Random(3))}
    for pr in rep.probes:
        p = pairs[pr.pair_id]
        gap = _q(p.a) - _q(p.b)
        if abs(gap) > 0.1:
            assert pr.debiased == ("A" if gap > 0 else "B")
        if not pr.consistent:
            assert pr.debiased == "tie"


async def test_single_order_probe_would_confound_bias_with_quality():
    """If the better answer always sits in slot A, a naive 'P(first wins)' on
    one order looks like bias from a perfectly fair judge."""
    rng = random.Random(4)
    pairs = [Pair(f"p{i}", "q", f"q={0.9:.3f}", f"q={rng.random() * 0.5:.3f}") for i in range(50)]
    naive = sum([await fair_judge(p.question, p.a, p.b) == "first" for p in pairs]) / len(pairs)
    assert naive == 1.0
    rep = await probe_position_bias(fair_judge, pairs)
    assert rep.first_slot_rate == pytest.approx(0.5)


async def test_cached_judge_without_order_in_key_hides_bias():
    biased = make_biased(random.Random(5), p_first_on_close=1.0)
    cache = {}

    async def badly_cached(question, first, second):
        # "Cache the winner for this pair": order is dropped from the key, so
        # the swapped call never reaches the model.
        key = (question, frozenset((first, second)))
        if key not in cache:
            slot = await biased(question, first, second)
            cache[key] = None if slot == "tie" else (first if slot == "first" else second)
        winner = cache[key]
        return "tie" if winner is None else ("first" if winner == first else "second")

    ps = _pairs(100, random.Random(6))
    honest = await probe_position_bias(biased, ps)
    fooled = await probe_position_bias(badly_cached, ps)
    assert honest.consistency < 0.7 and honest.p_value < 1e-6
    assert fooled.consistency == 1.0 and fooled.first_slot_rate == pytest.approx(0.5)


async def test_unparseable_verdict_raises():
    async def junk(q, a, b):
        return "Answer 1"

    with pytest.raises(ValueError):
        await probe_position_bias(junk, [Pair("x", "q", "a", "b")])


def test_stats_helpers():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    assert binom_two_sided(5, 10) == pytest.approx(1.0)
    assert binom_two_sided(10, 10) == pytest.approx(2 / 1024)
