"""The judge queue and worker: every request settles, and anything but a verdict is a refusal."""

import asyncio
import threading
import time

import pytest

from spec_ptc_jev import Decision
from spec_ptc_jev.worker import JudgeWorker

ALLOW = Decision(True, 1.0, 0.0)


class Always:
    def decide(self, **_):
        return ALLOW


def ask(worker, judge):
    return worker.submit(judge, tool="t", description="", policy="p", inputs={})


def test_rejects_bad_concurrency():
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            JudgeWorker(max_concurrency=bad)


def test_close_settles_pending_and_later_requests_as_refusals():
    hold = threading.Event()

    class Stuck:
        def decide(self, **_):
            hold.wait(5)
            return ALLOW

    worker = JudgeWorker(timeout_s=30)
    pending = ask(worker, Stuck())
    time.sleep(0.05)
    worker.close()
    hold.set()
    assert pending.result(2).allowed is False
    late = ask(worker, Always())
    assert late.result(2).allowed is False and "closed" in late.result().reason


def test_a_cancelled_judgment_is_a_refusal_and_the_worker_keeps_working():
    class Cancels:
        async def adecide(self, **_):
            raise asyncio.CancelledError

        def decide(self, **_):
            return ALLOW

    worker = JudgeWorker()
    try:
        d = ask(worker, Cancels()).result(2)
        assert d.allowed is False and "cancel" in d.reason
        assert ask(worker, Always()).result(2).allowed
    finally:
        worker.close()


def test_a_judge_error_is_a_refusal():
    class Broken:
        def decide(self, **_):
            raise RuntimeError("down")

    worker = JudgeWorker()
    try:
        d = ask(worker, Broken()).result(2)
        assert d.allowed is False and "RuntimeError" in d.reason
    finally:
        worker.close()


def test_queued_requests_time_out_on_their_own_deadline():
    hold = threading.Event()

    class Stuck:
        def decide(self, **_):
            hold.wait(3)
            return ALLOW

    worker = JudgeWorker(timeout_s=0.2, max_concurrency=1)
    try:
        t0 = time.perf_counter()
        futs = [ask(worker, Stuck()) for _ in range(4)]
        assert [f.result(2).allowed for f in futs] == [False] * 4
        assert (
            time.perf_counter() - t0 < 1.0
        )  # all four refused near 0.2 s, not one after another
    finally:
        hold.set()
        worker.close()


def test_requests_run_concurrently():
    class Slow:
        def decide(self, **_):
            time.sleep(0.3)
            return ALLOW

    worker = JudgeWorker()
    try:
        t0 = time.perf_counter()
        futs = [ask(worker, Slow()) for _ in range(5)]
        assert all(f.result(3).allowed for f in futs)
        assert time.perf_counter() - t0 < 0.8
    finally:
        worker.close()
