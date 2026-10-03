"""DEBUG-1 (as shipped): judge scoring service. Contains planted bugs.

Symptom report: "After the judge-v2 rollout, agreement with humans did not
move at all; the scores table has more rows than items; the calibration
report shows kappa 0.91 but production disagreement looks much worse."
"""

import hashlib
import random

_CACHE: dict[str, dict] = {}


def cache_key(item: dict) -> str:
    return hashlib.md5((item["prompt"] + item["response"]).encode()).hexdigest()


def score_items(items, judge, writer, judge_version, max_retries=3):
    for item in items:
        k = cache_key(item)
        if k in _CACHE:
            verdict = _CACHE[k]
        else:
            verdict = None
            for attempt in range(max_retries):
                try:
                    verdict = judge(item, judge_version)
                    writer.insert({"item_id": item["item_id"], "score": verdict["score"],
                                   "judge_version": judge_version})
                    break
                except TimeoutError:
                    continue
            _CACHE[k] = verdict


def split_for_calibration(rows, holdout_frac=0.3, seed=0):
    random.seed(seed)
    random.shuffle(rows)
    cut = int(len(rows) * (1 - holdout_frac))
    return rows[:cut], rows[cut:]
