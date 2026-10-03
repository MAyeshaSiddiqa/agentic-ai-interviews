"""Exercise PY-2: clustered bootstrap confidence intervals.

Rows (turns, judged steps) inside one cluster (trace, task, user) are
correlated. Resampling rows iid treats them as independent and produces
intervals that are too narrow. Resample whole clusters instead.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    low: float
    high: float
    n_clusters: int
    n_rows: int
    n_boot: int


def _cluster_sums(values: Sequence[float], clusters: Sequence[Hashable]):
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("no data")
    if not np.all(np.isfinite(v)):
        raise ValueError("values contain NaN/inf; decide on a policy before bootstrapping")
    ids: dict[Hashable, int] = {}
    inv = np.fromiter((ids.setdefault(c, len(ids)) for c in clusters), dtype=np.int64, count=len(clusters))
    g = len(ids)
    sums = np.bincount(inv, weights=v, minlength=g)
    counts = np.bincount(inv, minlength=g).astype(float)
    return sums, counts


def clustered_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[Hashable],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
    min_clusters: int = 2,
) -> BootstrapCI:
    """Percentile CI for the row-level mean, resampling clusters.

    The statistic is the ratio of sums (total / rows), so large clusters keep
    their row weight. Averaging per-cluster means would answer a different
    question (the mean over clusters), and the two can diverge when cluster
    size correlates with quality.
    """
    sums, counts = _cluster_sums(values, clusters)
    g = sums.size
    if g < min_clusters:
        raise ValueError(f"need >= {min_clusters} clusters, got {g}: CI is not estimable")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, g, size=(n_boot, g))
    stats = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    a = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(stats, [a, 1.0 - a])
    return BootstrapCI(
        estimate=float(sums.sum() / counts.sum()),
        low=float(lo),
        high=float(hi),
        n_clusters=int(g),
        n_rows=int(counts.sum()),
        n_boot=n_boot,
    )


def paired_clustered_bootstrap_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    clusters: Sequence[Hashable],
    **kwargs,
) -> BootstrapCI:
    """CI for mean(a) - mean(b) when both systems were scored on the same rows.

    Bootstrapping the per-row difference keeps the pairing. Two independent
    bootstraps would discard the shared item difficulty and inflate the width.
    """
    if len(values_a) != len(values_b):
        raise ValueError("paired comparison needs equal-length score vectors")
    diff = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    return clustered_bootstrap_ci(diff, clusters, **kwargs)


def iid_bootstrap_ci(values: Sequence[float], **kwargs) -> BootstrapCI:
    """The naive baseline: every row is its own cluster."""
    return clustered_bootstrap_ci(values, list(range(len(values))), **kwargs)
