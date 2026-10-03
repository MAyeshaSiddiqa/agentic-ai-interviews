"""Exercise PY-5: rate-limited async judge runner.

Guarantees:
  * at most `max_concurrency` judge calls in flight
  * a token-bucket rate limit (calls/sec with a burst)
  * retries only on retryable errors, exponential backoff with full jitter,
    Retry-After honoured as a lower bound
  * a hard spend budget: cost is reserved before each attempt and settled
    after, so concurrent attempts cannot jointly overshoot
  * idempotency: a request key covers everything that changes the verdict;
    completed keys are served from the store, concurrent duplicates share one
    in-flight call
Clock, sleep and RNG are injectable so tests are deterministic and instant.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Literal


class RetryableJudgeError(Exception):
    def __init__(self, msg: str = "", retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class FatalJudgeError(Exception):
    """Bad request, content policy, schema violation: retrying cannot help."""


@dataclass(frozen=True)
class JudgeRequest:
    item_id: str
    payload: dict[str, Any]
    judge_model: str
    judge_version: str
    rubric_version: str

    def idempotency_key(self) -> str:
        # item_id is deliberately excluded: two items with identical content
        # get the same verdict. Every field that can change the verdict is in.
        body = json.dumps(
            {
                "payload": self.payload,
                "judge_model": self.judge_model,
                "judge_version": self.judge_version,
                "rubric_version": self.rubric_version,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(body.encode()).hexdigest()


Status = Literal["ok", "cached", "failed", "skipped_budget"]


@dataclass
class JudgeOutcome:
    item_id: str
    key: str
    status: Status
    verdict: Any = None
    attempts: int = 0
    cost: float = 0.0
    error: str | None = None
    backoffs: list[float] = field(default_factory=list)


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, limit: float):
        self.limit = limit
        self.spent = 0.0
        self.reserved = 0.0

    def reserve(self, amount: float) -> None:
        if self.spent + self.reserved + amount > self.limit + 1e-12:
            raise BudgetExceeded(f"spent={self.spent:.4f} reserved={self.reserved:.4f} ask={amount:.4f}")
        self.reserved += amount

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved -= reserved
        self.spent += actual


class TokenBucket:
    def __init__(self, rate: float, burst: int, clock: Callable[[], float], sleep: Callable[[float], Awaitable[None]]):
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be > 0 and burst >= 1")
        self.rate, self.burst = rate, float(burst)
        self.tokens = float(burst)
        self.clock, self.sleep = clock, sleep
        self.t = clock()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        # Holding the lock while sleeping serialises waiters in FIFO order,
        # so a burst of callers cannot all observe the same refilled token.
        async with self.lock:
            while True:
                now = self.clock()
                self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                await self.sleep((1.0 - self.tokens) / self.rate)


JudgeFn = Callable[[JudgeRequest], Awaitable[tuple[Any, float]]]  # -> (verdict, actual_cost)


class JudgeRunner:
    def __init__(
        self,
        judge: JudgeFn,
        *,
        store: MutableMapping[str, Any],
        budget: Budget,
        estimate_cost: Callable[[JudgeRequest], float],
        max_concurrency: int = 8,
        rate_per_sec: float = 10.0,
        burst: int = 10,
        max_attempts: int = 4,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ):
        self.judge = judge
        self.store = store
        self.budget = budget
        self.estimate_cost = estimate_cost
        self.sem = asyncio.Semaphore(max_concurrency)
        self.bucket = TokenBucket(rate_per_sec, burst, clock, sleep)
        self.max_attempts = max_attempts
        self.base_delay, self.max_delay = base_delay, max_delay
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.inflight: dict[str, asyncio.Future] = {}

    def backoff(self, attempt: int, retry_after: float | None) -> float:
        cap = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        delay = self.rng.uniform(0.0, cap)  # full jitter decorrelates clients
        return max(delay, retry_after or 0.0)

    async def run(self, requests: Iterable[JudgeRequest]) -> list[JudgeOutcome]:
        return list(await asyncio.gather(*(self.run_one(r) for r in requests)))

    async def run_one(self, req: JudgeRequest) -> JudgeOutcome:
        key = req.idempotency_key()
        if key in self.store:
            return JudgeOutcome(req.item_id, key, "cached", verdict=self.store[key])
        if key in self.inflight:
            shared: JudgeOutcome = await asyncio.shield(self.inflight[key])
            return JudgeOutcome(req.item_id, key, "cached" if shared.status == "ok" else shared.status,
                                verdict=shared.verdict, error=shared.error)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.inflight[key] = fut
        try:
            out = await self._execute(req, key)
            fut.set_result(out)
            return out
        except BaseException as e:
            fut.set_exception(e)
            fut.exception()  # mark retrieved when no duplicate awaits it
            raise
        finally:
            del self.inflight[key]

    async def _execute(self, req: JudgeRequest, key: str) -> JudgeOutcome:
        out = JudgeOutcome(req.item_id, key, "failed")
        for attempt in range(1, self.max_attempts + 1):
            est = self.estimate_cost(req)
            try:
                self.budget.reserve(est)
            except BudgetExceeded as e:
                out.status, out.error = "skipped_budget", str(e)
                return out
            actual = est  # a failed/timed-out call is still billed: assume the estimate
            retry_after = None
            try:
                await self.bucket.acquire()
                async with self.sem:
                    out.attempts = attempt
                    verdict, actual = await self.judge(req)
                self.store[key] = verdict  # write-once under a content key: replay-safe
                out.status, out.verdict = "ok", verdict
                return out
            except RetryableJudgeError as e:
                out.error = f"retryable: {e}"
                retry_after = e.retry_after
            except FatalJudgeError as e:
                out.status, out.error = "failed", f"fatal: {e}"
                return out
            except Exception as e:  # unknown errors are not retried: fail closed, keep the batch alive
                out.status, out.error = "failed", f"unexpected: {type(e).__name__}: {e}"
                return out
            finally:
                self.budget.settle(est, actual)
                out.cost += actual
            if attempt < self.max_attempts:
                d = self.backoff(attempt, retry_after)
                out.backoffs.append(d)
                await self.sleep(d)
        out.status = "failed"
        return out
