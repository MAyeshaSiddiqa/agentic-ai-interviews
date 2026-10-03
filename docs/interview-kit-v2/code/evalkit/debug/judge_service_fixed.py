"""DEBUG-1 (fixed): judge scoring service."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    version: str
    rubric: str
    temperature: float = 0.0

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()[:16]


def cache_key(item: dict, cfg: JudgeConfig) -> str:
    # Canonical JSON: plain concatenation lets ("ab", "c") collide with ("a", "bc").
    body = json.dumps({"prompt": item["prompt"], "response": item["response"], "judge": cfg.fingerprint},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass
class Metrics:
    counters: Counter = field(default_factory=Counter)

    def inc(self, name: str, **labels) -> None:
        self.counters[(name, tuple(sorted(labels.items())))] += 1


def _retry(fn: Callable, *, attempts: int, retry_on: tuple[type[BaseException], ...], metrics: Metrics, op: str):
    for a in range(1, attempts + 1):
        try:
            return fn()
        except retry_on:
            metrics.inc("retry", op=op, attempt=a)
            if a == attempts:
                raise


def score_items(items, judge, writer, cfg: JudgeConfig, cache: dict, metrics: Metrics, max_retries=3):
    results = {}
    for item in items:
        k = cache_key(item, cfg)
        if k in cache:
            verdict = cache[k]
            metrics.inc("cache_hit", judge=cfg.fingerprint)
        else:
            metrics.inc("cache_miss", judge=cfg.fingerprint)
            # Retry the judge call and the write independently: a write timeout
            # must never trigger a second (billed, possibly different) judgement.
            verdict = _retry(lambda: judge(item, cfg.version), attempts=max_retries,
                             retry_on=(TimeoutError,), metrics=metrics, op="judge")
            cache[k] = verdict
        row = {"item_id": item["item_id"], "score": verdict["score"], "judge_fingerprint": cfg.fingerprint,
               "cache_key": k}
        # Upsert on (item_id, judge_fingerprint): replay-safe after an ambiguous timeout.
        _retry(lambda: writer.upsert((item["item_id"], cfg.fingerprint), row), attempts=max_retries,
               retry_on=(TimeoutError,), metrics=metrics, op="write")
        results[item["item_id"]] = verdict
    return results


def _bucket(group_id: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{group_id}".encode()).hexdigest()
    return int(h[:15], 16) / 16**15


def split_for_calibration(rows, *, group_key="trace_id", holdout_frac=0.3, salt="calib-v1"):
    """Group split by trace: every row of a trace lands on the same side, and
    the assignment is a pure function of the id, stable as data grows."""
    calib, holdout = [], []
    for r in rows:
        (holdout if _bucket(str(r[group_key]), salt) < holdout_frac else calib).append(r)
    overlap = {r[group_key] for r in calib} & {r[group_key] for r in holdout}
    assert not overlap, f"group leakage: {sorted(overlap)[:5]}"
    return calib, holdout
