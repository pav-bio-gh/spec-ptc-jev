"""The SDK surface: it never hangs, it is a plain loop unless you opt in, and async just works."""

import asyncio
import threading
import time

import pytest

import spec_ptc_jev.repl as R
from spec_ptc_jev import Decision, Run, SpecRepl


class Yes:
    def __init__(self):
        self.asked = 0

    def decide(self, **_):
        self.asked += 1
        return Decision(True, 1.0, 0.0)


def chunks(code, tok_per_s=400, fail_after=None):
    text = f"```repl\n{code}\n```\n"
    for i in range(0, len(text), 4):
        if fail_after is not None and i > fail_after:
            raise ConnectionError("stream dropped")
        time.sleep(4 / tok_per_s / 4)
        yield text[i : i + 4]


def within(seconds, fn):
    """Run fn on a thread; fail the test (rather than hang the suite) if it does not return."""
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:
            box["error"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(seconds)
    assert not th.is_alive(), f"did not return within {seconds}s"
    if "error" in box:
        raise box["error"]
    return box["value"]


def reader(repl, log=None, latency=0.0):
    @repl.tool(early_when="reads only")
    def kv(op: str, key: str = "") -> str:
        """Key-value store."""
        if log is not None:
            log.append((time.perf_counter(), op, key))
        time.sleep(latency)
        return f"{op}:{key}"

    return kv


# --------------------------------------------------------------------------- it never hangs
def test_a_bug_in_planning_falls_back_to_a_normal_run(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("planner bug")

    monkeypatch.setattr(R, "plan_calls", boom)
    events = []
    repl = SpecRepl(judge=Yes(), on_event=lambda kind, **d: events.append(kind))
    reader(repl)
    turn = within(5, lambda: repl.run_turn(chunks("a = kv('get', 'k')\nfinal_answer(a)")))
    assert turn.error is None and turn.answer == "get:k"
    assert "planning_off" in events


def test_a_bug_while_reading_the_code_ends_the_turn_with_an_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("splitter bug")

    monkeypatch.setattr(R, "closed_statements", boom)
    repl = SpecRepl(judge=Yes())
    reader(repl)
    turn = within(5, lambda: repl.run_turn(chunks("a = kv('get', 'k')\nfinal_answer(a)")))
    assert turn.answer is None and "internal error" in turn.error


def test_a_dropped_model_stream_raises_and_leaves_no_threads_behind():
    repl = SpecRepl(judge=Yes())
    log = []
    reader(repl, log)

    @repl.tool
    def send(msg: str) -> str:
        """Send a message."""
        log.append((time.perf_counter(), "send", msg))
        return "sent"

    code = "a = kv('get', 'k')\nsend(a)\n" + "\n".join(f"x{i} = {i}" for i in range(200))
    before = threading.active_count()
    with pytest.raises(ConnectionError):
        within(5, lambda: repl.run_turn(chunks(code, fail_after=200)))
    time.sleep(0.2)
    assert (
        threading.active_count() <= before + 3
    )  # pools may keep idle workers; no stuck turn threads
    assert [op for _, op, _ in log] == [
        "get"
    ]  # the send was waiting for the full block: it never ran
    assert (
        within(5, lambda: repl.run_turn(chunks("final_answer('still usable')"))).answer
        == "still usable"
    )


def test_one_turn_at_a_time():
    repl = SpecRepl()
    gate = threading.Event()

    @repl.tool
    def slow() -> str:
        """Wait."""
        gate.wait(5)
        return "done"

    th = threading.Thread(target=lambda: repl.run_turn(chunks("slow()")), daemon=True)
    th.start()
    time.sleep(0.3)
    with pytest.raises(RuntimeError, match="one SpecRepl per conversation"):
        repl.run_turn(chunks("x = 1"))
    gate.set()
    th.join(5)


# --------------------------------------------------------------------------- a plain loop unless you opt in
def test_with_no_marked_tools_it_is_exactly_generate_then_run():
    judge, log, events = Yes(), [], []
    repl = SpecRepl(judge=judge, on_event=lambda kind, **d: events.append((kind, d)))

    @repl.tool
    def fetch(key: str) -> str:
        """Read a record."""
        log.append(time.perf_counter())
        return key

    ran = []
    repl.ns["note"] = ran.append
    code = (
        "note('line 1 ran')\nv = fetch('a')\n"
        + "\n".join(f"p{i} = {i}" for i in range(60))
        + "\nfinal_answer(v)"
    )
    turn = repl.run_turn(chunks(code))
    commit = repl._t0 + next(d["t"] for k, d in events if k == "commit")
    assert turn.answer == "a" and judge.asked == 0
    assert log[0] >= commit  # nothing, not even plain Python, ran before the block was complete

    ran.clear()
    turn = repl.run_turn(chunks("note('line 1 ran')\nthis is not python ((("))
    assert "SyntaxError" in turn.error and ran == []  # a broken block runs nothing at all


def test_speculate_false_is_the_same_plain_loop_even_with_marked_tools():
    judge, log, events = Yes(), [], []
    repl = SpecRepl(
        judge=judge, speculate=False, on_event=lambda kind, **d: events.append((kind, d))
    )
    reader(repl, log)
    turn = repl.run_turn(
        chunks(
            "a = kv('get', 'k')\n"
            + "\n".join(f"p{i} = {i}" for i in range(60))
            + "\nfinal_answer(a)"
        )
    )
    commit = repl._t0 + next(d["t"] for k, d in events if k == "commit")
    assert turn.answer == "get:k" and judge.asked == 0 and log[0][0] >= commit


# --------------------------------------------------------------------------- async, on YOUR loop
def test_arun_turn_runs_async_tools_on_the_callers_loop_and_does_not_block_it():
    async def main():
        app_loop = asyncio.get_running_loop()
        app_lock = asyncio.Lock()  # bound to the app's loop, like an HTTP session or a DB pool
        seen, ticks = set(), []
        repl = SpecRepl(judge=Yes(), max_parallel=2)

        @repl.tool(early_when="reads only")
        async def fetch(key: str) -> str:
            """Read a record."""
            seen.add(asyncio.get_running_loop() is app_loop)
            async with app_lock:
                pass
            await asyncio.sleep(0.3)
            return key.upper()

        async def astream():
            text = "```repl\nout = []\nfor k in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:\n    out.append(fetch(k))\nfinal_answer(''.join(out))\n```\n"
            for i in range(0, len(text), 6):
                await asyncio.sleep(0.001)
                yield text[i : i + 6]

        async def heartbeat():
            while True:
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.02)

        hb = asyncio.create_task(heartbeat())
        t0 = time.perf_counter()
        turn = await repl.arun_turn(astream())
        wall = time.perf_counter() - t0
        hb.cancel()
        repl.close()
        return turn, wall, seen, ticks

    turn, wall, seen, ticks = asyncio.run(main())
    assert turn.error is None and turn.answer == "ABCDEFGH"
    assert seen == {True}  # every async tool ran on the app's own loop
    assert wall < 1.2  # eight 0.3 s reads side by side, on two worker threads
    gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
    assert max(gaps) < 0.25  # the app's loop kept running the whole time


def test_arun_works_with_an_async_chat_and_a_sync_one():
    async def main():
        out = []
        for make_async in (True, False):
            repl = SpecRepl(judge=Yes())

            @repl.tool(early=True)
            async def double(x: int) -> int:
                """Double a number."""
                await asyncio.sleep(0.01)
                return x * 2

            replies = iter(
                [
                    "```repl\nprint(double(4))\n```",
                    "```repl\nfinal_answer(str(double(21)))\n```",
                ]
            )

            def chat_sync(messages, replies=replies):
                yield next(replies)

            async def chat_async(messages, replies=replies):
                yield next(replies)

            result = await repl.arun(chat_async if make_async else chat_sync, "double things")
            out.append(result)
        return out

    for result in asyncio.run(main()):
        assert isinstance(result, Run) and result.answer == "42" and len(result.turns) == 2
        assert "8" in result.turns[0].stdout


def test_the_sync_entry_points_refuse_to_block_a_running_loop():
    async def main():
        repl = SpecRepl()
        with pytest.raises(RuntimeError, match="arun_turn"):
            repl.run_turn(iter(["```repl\nx = 1\n```"]))
        with pytest.raises(RuntimeError, match="arun"):
            repl.run(lambda m: iter([""]), "task")

    asyncio.run(main())


# --------------------------------------------------------------------------- small things that make it easy
def test_bare_decorator_context_manager_final_answer_and_run_result():
    with SpecRepl() as repl:

        @repl.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert (
            "add(a: int, b: int) -> int" in repl.system_prompt()
            and "final_answer" in repl.system_prompt()
        )
        result = repl.run(
            lambda messages: iter(["```repl\nfinal_answer(str(add(2, 3)))\n```"]), "add"
        )
        assert isinstance(result, Run) and result.answer == "5" and result.error is None
        # the RLM-style dict still works, including when the model rebinds `answer`
        assert (
            repl.run(
                lambda m: iter(["```repl\nanswer = {'content': 'ok', 'ready': True}\n```"]), "x"
            ).answer
            == "ok"
        )


def test_every_code_block_in_a_reply_runs_in_order():
    repl = SpecRepl()
    reply = "First.\n```repl\na = 1\n```\nthen\n```python\nb = a + 1\n```\nand\n```\nfinal_answer(str(a + b))\n```\n"
    assert (
        repl.run_turn(iter([reply[i : i + 7] for i in range(0, len(reply), 7)])).answer == "3"
    )


def test_a_tool_never_runs_twice_even_when_its_early_run_fails():
    repl = SpecRepl(judge=Yes())
    n = {"calls": 0}

    @repl.tool(early=True)
    def flaky(x: str) -> str:
        """Fails."""
        n["calls"] += 1
        raise OSError("down")

    turn = repl.run_turn(chunks("v = flaky('a')\nfinal_answer(v)"))
    assert "OSError" in turn.error and turn.answer is None and n["calls"] == 1


def test_planning_never_touches_model_defined_objects():
    repl = SpecRepl(judge=Yes())
    log = []
    reader(repl, log)
    code = (
        "class Sneaky:\n"
        "    def __format__(self, spec):\n"
        "        record('format ran on ' + threading.current_thread().name)\n"
        "        return 'k'\n"
        "s = Sneaky()\n"
        "v = kv('get', f'{s}')\n"
        "final_answer(v)"
    )
    where = []
    repl.ns.update(record=where.append, threading=threading)
    turn = repl.run_turn(chunks(code))
    assert turn.answer == "get:k"
    assert where == [
        "format ran on spec-repl-exec"
    ]  # once, by the program itself, never by planning


def test_completion_without_a_model_says_what_to_do():
    repl = SpecRepl()
    with pytest.raises(ValueError, match="model="):
        repl.completion("anything")
    repl.close()
