import math
import random

import pytest

from evalkit.agreement import cohen_kappa, krippendorff_alpha

# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", worked
# example: 4 coders x 12 units with missing values.
_A = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
_B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
_C = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
_D = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
UNITS = [list(u) for u in zip(_A, _B, _C, _D)]


@pytest.mark.parametrize(
    "level,expected",
    [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849), ("ratio", 0.797)],
)
def test_alpha_matches_published_example(level, expected):
    assert krippendorff_alpha(UNITS, level=level) == pytest.approx(expected, abs=5e-4)


def test_alpha_excludes_unpairable_units():
    # Unit 12 has one value; adding more single-rated units must not change alpha.
    base = krippendorff_alpha(UNITS)
    assert krippendorff_alpha(UNITS + [[None, 7, None, None]] * 5) == pytest.approx(base)


def test_alpha_no_variation_is_nan():
    assert math.isnan(krippendorff_alpha([[1, 1], [1, 1, 1]]))


def test_alpha_ordinal_rejects_strings():
    with pytest.raises(ValueError):
        krippendorff_alpha([["a", "b"], ["b", "b"]], level="interval")


def test_kappa_known_value():
    # 2x2 table [[20, 5], [10, 15]]: po = .70, pe = .50, kappa = .40
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_prevalence_paradox():
    # 96% raw agreement, kappa below 0.5: the skewed-prevalence trap.
    a = ["pass"] * 95 + ["fail"] * 5
    b = ["pass"] * 93 + ["fail"] * 2 + ["pass"] * 2 + ["fail"] * 3
    raw = sum(x == y for x, y in zip(a, b)) / len(a)
    assert raw == pytest.approx(0.96)
    assert cohen_kappa(a, b) < 0.6


def test_kappa_constant_rater_is_nan_not_zero_or_one():
    assert math.isnan(cohen_kappa(["pass"] * 10, ["pass"] * 10))


def test_kappa_drops_missing_pairs_and_checks_length():
    assert cohen_kappa([1, 2, None, 1], [1, 2, 2, None]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1])


def test_weighted_kappa_orders_and_penalises_distance():
    a = [1, 2, 3, 4, 5] * 10
    near = [min(5, x + 1) for x in a]
    far = [6 - x for x in a]
    assert cohen_kappa(a, near, weights="quadratic") > cohen_kappa(a, far, weights="quadratic")
    with pytest.raises(ValueError):
        cohen_kappa(["lo", "hi"], ["hi", "lo"], weights="linear")


def test_kappa_matches_sklearn_when_available():
    skm = pytest.importorskip("sklearn.metrics")
    rng = random.Random(7)
    a = [rng.randint(1, 5) for _ in range(300)]
    b = [x if rng.random() < 0.6 else rng.randint(1, 5) for x in a]
    for w in (None, "linear", "quadratic"):
        assert cohen_kappa(a, b, weights=w) == pytest.approx(skm.cohen_kappa_score(a, b, weights=w))


def test_two_rater_alpha_close_to_kappa_for_large_n():
    rng = random.Random(3)
    a = [rng.choice("xyz") for _ in range(2000)]
    b = [x if rng.random() < 0.7 else rng.choice("xyz") for x in a]
    alpha = krippendorff_alpha([[x, y] for x, y in zip(a, b)])
    assert alpha == pytest.approx(cohen_kappa(a, b), abs=0.01)
