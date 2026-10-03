import numpy as np
import pytest

from evalkit.bootstrap import (
    clustered_bootstrap_ci,
    iid_bootstrap_ci,
    paired_clustered_bootstrap_diff,
)


def _clustered_data(rng, g=40, per=25, sigma_cluster=0.25, mu=0.6):
    """Pass/fail scores whose pass rate varies strongly by task (cluster)."""
    vals, cl = [], []
    for c in range(g):
        p = float(np.clip(mu + rng.normal(0, sigma_cluster), 0.01, 0.99))
        vals += list((rng.random(per) < p).astype(float))
        cl += [f"task-{c}"] * per
    return np.array(vals), cl


def test_estimate_is_row_mean_and_interval_brackets_it():
    rng = np.random.default_rng(0)
    v, c = _clustered_data(rng)
    ci = clustered_bootstrap_ci(v, c, n_boot=1000)
    assert ci.estimate == pytest.approx(v.mean())
    assert ci.low < ci.estimate < ci.high
    assert (ci.n_clusters, ci.n_rows) == (40, 1000)


def test_iid_interval_is_too_narrow_under_clustering():
    rng = np.random.default_rng(1)
    v, c = _clustered_data(rng)
    cl = clustered_bootstrap_ci(v, c, n_boot=2000)
    iid = iid_bootstrap_ci(v, n_boot=2000)
    assert (cl.high - cl.low) > 2.0 * (iid.high - iid.low)


def test_coverage_cluster_vs_iid():
    """Monte Carlo: the cluster interval covers the true mean near nominally,
    the iid interval badly under-covers."""
    rng = np.random.default_rng(2)
    mu, reps = 0.6, 120
    cov_cl = cov_iid = 0
    for r in range(reps):
        v, c = _clustered_data(rng, g=30, per=20, mu=mu)
        a = clustered_bootstrap_ci(v, c, n_boot=400, seed=r)
        b = iid_bootstrap_ci(v, n_boot=400, seed=r)
        # True mean of the clipped cluster rates is ~mu; use a tolerance-free check.
        cov_cl += a.low <= mu <= a.high
        cov_iid += b.low <= mu <= b.high
    assert cov_cl / reps >= 0.85
    assert cov_iid / reps <= 0.65


def test_row_weighting_differs_from_cluster_mean():
    # One big bad cluster and many small good ones.
    v = [0.0] * 100 + [1.0] * 10
    c = ["big"] * 100 + [f"s{i}" for i in range(10)]
    ci = clustered_bootstrap_ci(v, c, n_boot=200)
    assert ci.estimate == pytest.approx(10 / 110)


def test_paired_diff_is_tighter_than_unpaired():
    rng = np.random.default_rng(4)
    difficulty = rng.normal(0, 1, 50)
    clusters = [f"t{i}" for i in range(50) for _ in range(4)]
    base = np.repeat(difficulty, 4) + rng.normal(0, 0.2, 200)
    a, b = base + 0.1, base
    paired = paired_clustered_bootstrap_diff(a, b, clusters, n_boot=1000)
    ca = clustered_bootstrap_ci(a, clusters, n_boot=1000, seed=1)
    cb = clustered_bootstrap_ci(b, clusters, n_boot=1000, seed=2)
    unpaired_width = np.hypot(ca.high - ca.low, cb.high - cb.low)
    assert paired.low > 0.0
    assert (paired.high - paired.low) < 0.25 * unpaired_width


def test_guards():
    with pytest.raises(ValueError):
        clustered_bootstrap_ci([1.0, 0.0], ["one", "one"])
    with pytest.raises(ValueError):
        clustered_bootstrap_ci([1.0, float("nan")], ["a", "b"])
    with pytest.raises(ValueError):
        clustered_bootstrap_ci([1.0], ["a", "b"])


def test_cluster_ids_are_not_stringified():
    # 1 and "1" are different clusters.
    ci = clustered_bootstrap_ci([0.0, 1.0], [1, "1"], n_boot=10)
    assert ci.n_clusters == 2


def test_seed_reproducibility():
    v, c = [0, 1, 1, 0, 1, 1], ["a", "a", "b", "b", "c", "c"]
    assert clustered_bootstrap_ci(v, c, seed=5) == clustered_bootstrap_ci(v, c, seed=5)
