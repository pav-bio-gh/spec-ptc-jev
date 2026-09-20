"""The judge queue and the judge worker.

Every verdict request, from any thread, goes onto ONE queue. ONE worker (an asyncio loop on
its own daemon thread) takes requests off it and runs them concurrently. Callers get a
`concurrent.futures.Future[Decision]` back immediately and never block to submit.

    stream thread ──(statement closed, literal call)──┐
                                                       ├──> judge queue ──> judge worker ──> verdict futures
    look-ahead thread ──(reached a call)───────────────┘

Guarantees:
  - every future settles. Error, timeout, cancellation and shutdown all settle it as a REFUSAL:
    no verdict, no early execution.
  - a request's deadline starts when it is SUBMITTED, so waiting in the queue counts.
  - nothing but judging runs on the loop. Callers must keep their done-callbacks tiny.

Sync judges (`decide` only) run on a bounded pool of `max_concurrency` threads. A timeout cannot
stop a sync call that is already running, so a stuck sync judge holds its thread until it
returns, and later requests time out as refusals. Give sync judges their own network deadline,
or implement `adecide` (as `JevJudge` does).
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from spec_ptc_jev.judge import Decision, Judge

DEFAULT_TIMEOUT_S = 2.0
DEFAULT_MAX_CONCURRENCY = 64


def _refusal(reason: str) -> Decision:
    return Decision(False, 0.0, 1.0, reason=reason)


@dataclass
class _Request:
    judge: Judge
    tool: str
    description: str
    policy: str
    inputs: Any
    deadline: float
    future: Future = field(default_factory=Future)

    def settle(self, decision: Decision) -> None:
        if not self.future.done():
            try:
                self.future.set_result(decision)
            except Exception:  # lost a race with another settle: already has a result
                pass


class JudgeWorker:
    def __init__(
        self,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        if not timeout_s > 0:
            raise ValueError("timeout_s must be positive")
        if (
            not isinstance(max_concurrency, int)
            or isinstance(max_concurrency, bool)
            or max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        self.timeout_s = timeout_s
        self.max_concurrency = max_concurrency
        self._state = threading.Lock()
        self._closed = False
        self._outstanding: set[int] = set()
        self._requests: dict[int, _Request] = {}
        self._sync_pool = ThreadPoolExecutor(
            max_workers=max_concurrency, thread_name_prefix="spec-ptc-jev-syncjudge"
        )
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="spec-ptc-jev-judge", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    # ------------------------------------------------------------------ any thread
    def submit(
        self, judge: Judge, *, tool: str, description: str, policy: str, inputs: Any
    ) -> Future:
        """Queue one verdict request. Never blocks; the future resolves to a `Decision`."""
        req = _Request(
            judge, tool, description, policy, inputs, deadline=time.monotonic() + self.timeout_s
        )
        with self._state:
            if self._closed:
                req.settle(_refusal("judge worker is closed"))
                return req.future
            self._requests[id(req)] = req
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, req)
        except RuntimeError:  # loop closed between the check and the call
            self._finish(req, _refusal("judge worker is closed"))
        return req.future

    def close(self, timeout: float = 5.0) -> None:
        """Refuse new work, settle everything outstanding as refused, stop and join the loop."""
        with self._state:
            if self._closed:
                return
            self._closed = True
            pending = list(self._requests.values())
            self._requests.clear()
        for req in pending:
            req.settle(_refusal("judge worker closed before a verdict"))

        def stop() -> None:
            for task in list(self._tasks):
                task.cancel()
            self._loop.stop()

        try:
            self._loop.call_soon_threadsafe(stop)
        except RuntimeError:
            pass
        self._thread.join(timeout)
        self._sync_pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ worker thread
    def _run(self) -> None:
        try:
            asyncio.set_event_loop(self._loop)
            self._queue: asyncio.Queue[_Request] = asyncio.Queue()
            self._slots = asyncio.Semaphore(self.max_concurrency)
            self._tasks: set[asyncio.Task] = set()
            self._drainer = self._loop.create_task(self._drain())
        except BaseException as e:
            self._startup_error = e
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()
        # stopped by close(): let every cancelled task unwind (each settles its future), then
        # close the loop here, on its own thread
        leftovers = [self._drainer, *self._tasks]
        for task in leftovers:
            task.cancel()
        self._loop.run_until_complete(asyncio.gather(*leftovers, return_exceptions=True))
        self._loop.close()

    async def _drain(self) -> None:
        # One task per request, started at once: a request waiting for a slot is still on its
        # own deadline, so a full worker refuses late requests instead of stalling them.
        while True:
            req = await self._queue.get()
            task = self._loop.create_task(self._judge(req))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def _finish(self, req: _Request, decision: Decision) -> None:
        with self._state:
            self._requests.pop(id(req), None)
        req.settle(decision)

    async def _judge(self, req: _Request) -> None:
        decision = _refusal("judge worker stopped before a verdict")
        try:
            remaining = req.deadline - time.monotonic()
            if remaining <= 0:
                decision = _refusal(f"judge timeout after {self.timeout_s}s (queued)")
            else:
                decision = await asyncio.wait_for(self._guarded(req), remaining)
        except TimeoutError:
            decision = _refusal(f"judge timeout after {self.timeout_s}s")
        except asyncio.CancelledError:
            decision = _refusal(
                "judgment cancelled"
            )  # settle first, then let cancellation finish
            self._finish(req, decision)
            raise
        except Exception as e:  # fail closed
            decision = _refusal(f"judge error: {type(e).__name__}")
        finally:
            self._finish(req, decision)  # idempotent: every exit path settles the future

    async def _guarded(self, req: _Request) -> Decision:
        async with self._slots:
            return await self._ask(req)

    async def _ask(self, req: _Request) -> Decision:
        kwargs = dict(
            tool=req.tool, description=req.description, policy=req.policy, inputs=req.inputs
        )
        adecide = getattr(req.judge, "adecide", None)
        if adecide is not None:
            return await adecide(**kwargs)
        return await self._loop.run_in_executor(
            self._sync_pool, lambda: req.judge.decide(**kwargs)
        )


_default: JudgeWorker | None = None
_default_lock = threading.Lock()


def default_worker() -> JudgeWorker:
    global _default
    with _default_lock:
        if _default is None or _default._closed:
            _default = JudgeWorker()
        return _default
