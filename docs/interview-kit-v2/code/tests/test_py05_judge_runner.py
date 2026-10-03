import asyncio
import random

import pytest

from evalkit.judge_runner import (
    Budget,
    FatalJudgeError,
    JudgeRequest,
    JudgeRunner,
    RetryableJudgeError,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, d: float) -> None:
        self.sleeps.append(d)
        self.t += d
        await asyncio.sleep(0)


def req(i, payload=None, version="v1"):
    return JudgeRequest(f"item-{i}", payload or {"trace": f"t{i}"}, "judge-model", version, "rubric-3")


def make_runner(judge, clock, *, budget=100.0, **kw):
    return JudgeRunner(
        judge,
        store={},
        budget=Budget(budget),
        estimate_cost=lambda r: 1.0,
        clock=clock.now,
        sleep=clock.sleep,
        rng=random.Random(0),
        **kw,
    )


async def test_idempotent_across_runs_and_within_batch():
    calls = []

    async def judge(r):
        calls.append(r.item_id)
        await asyncio.sleep(0)
        return {"score": 4}, 1.0

    clock = FakeClock()
    runner = make_runner(judge, clock)
    batch = [req(1), req(2), JudgeRequest("dup-of-1", {"trace": "t1"}, "judge-model", "v1", "rubric-3")]
    out = await runner.run(batch)
    assert sorted(calls) == ["item-1", "item-2"]
    assert [o.status for o in out].count("cached") == 1
    out2 = await runner.run(batch)
    assert all(o.status == "cached" for o in out2) and len(calls) == 2
    assert runner.budget.spent == 2.0


async def test_judge_version_is_part_of_key():
    assert req(1, version="v1").idempotency_key() != req(1, version="v2").idempotency_key()
    assert req(1).idempotency_key() == JudgeRequest("other-id", {"trace": "t1"}, "judge-model", "v1", "rubric-3").idempotency_key()


async def test_retry_then_success_with_bounded_jitter():
    attempts = {"n": 0}

    async def flaky(r):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RetryableJudgeError("429")
        return {"score": 5}, 1.0

    clock = FakeClock()
    runner = make_runner(flaky, clock, base_delay=1.0, max_delay=10.0)
    [o] = await runner.run([req(1)])
    assert o.status == "ok" and o.attempts == 3
    assert len(o.backoffs) == 2
    assert 0 <= o.backoffs[0] <= 1.0 and 0 <= o.backoffs[1] <= 2.0
    assert o.cost == 3.0 and runner.budget.spent == 3.0  # failed attempts are billed


async def test_retry_after_is_a_floor():
    async def limited(r):
        raise RetryableJudgeError("429", retry_after=7.5)

    clock = FakeClock()
    runner = make_runner(limited, clock, max_attempts=2, base_delay=0.1)
    [o] = await runner.run([req(1)])
    assert o.status == "failed" and o.backoffs == [7.5]


async def test_fatal_and_unexpected_errors_are_not_retried():
    async def bad(r):
        if r.item_id == "item-1":
            raise FatalJudgeError("400 invalid schema")
        raise KeyError("parser bug")

    clock = FakeClock()
    runner = make_runner(bad, clock)
    o1, o2 = await runner.run([req(1), req(2)])
    assert (o1.status, o1.attempts) == ("failed", 1) and "fatal" in o1.error
    assert (o2.status, o2.attempts) == ("failed", 1) and "unexpected" in o2.error
    assert not runner.store


async def test_budget_is_never_exceeded_under_concurrency():
    async def judge(r):
        await asyncio.sleep(0)
        return {"score": 3}, 1.0

    clock = FakeClock()
    runner = make_runner(judge, clock, budget=5.0, max_concurrency=10, rate_per_sec=1000, burst=100)
    out = await runner.run([req(i) for i in range(12)])
    statuses = [o.status for o in out]
    assert statuses.count("ok") == 5 and statuses.count("skipped_budget") == 7
    assert runner.budget.spent <= 5.0 and runner.budget.reserved == 0.0


async def test_concurrency_cap():
    live = {"now": 0, "max": 0}

    async def judge(r):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        for _ in range(3):
            await asyncio.sleep(0)
        live["now"] -= 1
        return 1, 0.0

    clock = FakeClock()
    runner = make_runner(judge, clock, max_concurrency=3, rate_per_sec=1e6, burst=1000)
    await runner.run([req(i) for i in range(20)])
    assert live["max"] == 3


async def test_rate_limit_spreads_calls_in_time():
    stamps = []
    clock = FakeClock()

    async def judge(r):
        stamps.append(clock.now())
        return 1, 0.0

    runner = make_runner(judge, clock, rate_per_sec=2.0, burst=2, max_concurrency=50)
    await runner.run([req(i) for i in range(10)])
    stamps.sort()
    assert stamps[1] == pytest.approx(0.0)  # burst
    assert stamps[-1] == pytest.approx(4.0)  # (10 - 2) / 2 per sec
    for k in range(2, 10):
        assert stamps[k] - stamps[k - 1] >= 0.5 - 1e-9
