"""SpecRepl, offline. A rule-based judge stands in for Jev; the stream is a timed generator."""

import ast
import asyncio
import threading
import time

import pytest

from spec_ptc_jev import Decision
from spec_ptc_jev.repl import SpecRepl, closed_statements, extract_code, plan_calls
from spec_ptc_jev.worker import JudgeWorker


class ReadsOnly:
    """Allows a call whose first argument starts with 'get'. Optionally slow."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.seen = []

    def decide(self, *, inputs, **_):
        self.seen.append(inputs)
        time.sleep(self.delay)
        ok = str(next(iter(inputs.values()))).startswith("get")
        return Decision(ok, 1.0 if ok else 0.0, 0.0 if ok else 1.0)


def stream(code, tok_per_s=200, prose="ok\n"):
    text = f"{prose}```repl\n{code}\n```\ndone"
    for i in range(0, len(text), 4):
        time.sleep(4 / tok_per_s / 4)
        yield text[i : i + 4]


def make(judge=None, speculate=True, latency=0.2, worker=None):
    events, calls, lock = [], [], threading.Lock()
    repl = SpecRepl(
        judge=judge or ReadsOnly(),
        speculate=speculate,
        worker=worker,
        on_event=lambda kind, **d: events.append((kind, d)),
    )

    @repl.tool(early_when="reads only")
    def kv(op: str, key: str = "") -> str:
        """Key-value store: 'get <key>' reads, 'set <key>' writes."""
        with lock:
            calls.append((time.perf_counter(), op, key))
        time.sleep(latency)
        return f"{op}:{key}"

    @repl.tool(early=True)
    def ask(prompt: str) -> str:
        """Ask a sub-model."""
        with lock:
            calls.append((time.perf_counter(), "ask", prompt))
        time.sleep(latency)
        return f"answer({prompt})"

    @repl.tool()
    def send(msg: str) -> str:
        """Send a message. No early policy."""
        with lock:
            calls.append((time.perf_counter(), "send", msg))
        return "sent"

    return repl, events, calls


def commit_time(repl, events):
    return repl._t0 + next(d["t"] for k, d in events if k == "commit")


CODE = "a = kv('get', 'k1')\nb = kv('set', 'k2')\nc = kv('get', 'k3')\nanswer['content'] = a + '|' + b + '|' + c\nanswer['ready'] = True"


def test_allowed_calls_start_before_the_code_is_finished_and_refused_ones_after():
    repl, events, calls = make()
    turn = repl.run_turn(stream(CODE + "\n" + "\n".join(f"pad{i} = {i}" for i in range(40))))
    assert turn.error is None and turn.answer == "get:k1|set:k2|get:k3"
    commit = commit_time(repl, events)
    by_key = {key: t for t, _, key in calls}
    assert by_key["k1"] < commit  # started while the model was still writing
    assert by_key["k2"] >= commit  # refused: waited for the whole block
    assert by_key["k3"] > by_key["k2"]  # nothing jumps over a refused call
    assert len(calls) == 3


def test_baseline_runs_nothing_before_the_block_is_complete():
    repl, events, calls = make(speculate=False)
    turn = repl.run_turn(stream(CODE))
    assert turn.answer == "get:k1|set:k2|get:k3"
    assert all(t >= commit_time(repl, events) for t, _, _ in calls)
    assert not [k for k, _ in events if k == "judge_start"]  # the baseline never asks the judge


LOOP = (
    "keys = ['k1', 'k2', 'k3']\n"
    "out = []\n"
    "for k in keys:\n"
    "    v = kv('get', k)\n"
    "    if 'k' in v:\n"
    "        out.append(v)\n"
    "    else:\n"
    "        out.append('none')\n"
    "answer['content'] = '|'.join(out)\n"
    "answer['ready'] = True"
)


def test_a_loop_fans_out_and_every_call_runs_exactly_once():
    # the shape that makes stock spec-ptc run every call twice (tests/test_upstream_retraction.py)
    repl, _, calls = make(latency=0.4)
    turn = repl.run_turn(stream(LOOP))
    assert turn.error is None and turn.answer == "get:k1|get:k2|get:k3"
    starts = sorted(t for t, _, _ in calls)
    assert len(calls) == 3
    assert starts[-1] - starts[0] < 0.2  # side by side; one after another would be 0.8 s


def test_a_slow_judge_is_asked_about_every_call_at_once():
    judge = ReadsOnly(delay=0.3)
    repl, _, calls = make(judge=judge, latency=0.05)
    t0 = time.perf_counter()
    turn = repl.run_turn(stream(LOOP, tok_per_s=2000))
    assert turn.answer == "get:k1|get:k2|get:k3"
    assert (
        time.perf_counter() - t0 < 0.9
    )  # three 0.3 s verdicts in a row would already be 0.9 s


def test_a_call_that_depends_on_earlier_results_still_starts_early():
    code = (
        "a = kv('get', 'k1')\nb = ask('about ' + a)\n"
        + "\n".join(f"pad{i} = {i}" for i in range(60))
        + "\nanswer['content'] = b\nanswer['ready'] = True"
    )
    repl, events, calls = make()
    turn = repl.run_turn(stream(code))
    assert turn.answer == "answer(about get:k1)"
    ask_t = next(t for t, op, _ in calls if op == "ask")
    assert ask_t < commit_time(repl, events)


def test_order_is_kept_inside_a_loop_that_writes_then_reads():
    code = "for k in ['a', 'b']:\n    kv('set', k)\n    v = kv('get', k)\nanswer['content'] = v\nanswer['ready'] = True"
    repl, _, calls = make()
    turn = repl.run_turn(stream(code))
    assert turn.answer == "get:b"
    assert [(op, key) for _, op, key in sorted(calls)] == [
        ("set", "a"),
        ("get", "a"),
        ("set", "b"),
        ("get", "b"),
    ]


def test_a_loop_starts_fanning_out_before_its_body_is_finished():
    body = "\n".join(f"    pad{i} = {i}" for i in range(40))
    code = f"for k in ['k1', 'k2', 'k3']:\n    v = kv('get', k)\n{body}\nanswer['content'] = v\nanswer['ready'] = True"
    repl, events, calls = make()
    t_begin = time.perf_counter()
    turn = repl.run_turn(stream(code))
    assert turn.answer == "get:k3" and len(calls) == 3
    commit = commit_time(repl, events)
    assert (
        max(t for t, _, _ in calls) < t_begin + (commit - t_begin) / 2
    )  # long before the loop closed


def test_a_write_written_later_in_the_loop_never_leaves_a_stale_read():
    # `get` is started early for every item while the body is still open. Then the model adds a
    # write to the body. Each get must still see every write before it in program order.
    repl = SpecRepl(judge=ReadsOnly())
    state = {"writes": 0}

    @repl.tool(early_when="reads only")
    def kv(op: str, key: str = "") -> str:
        """Key-value store: 'get <key>' counts writes so far, 'set <key>' writes."""
        if op == "set":
            state["writes"] += 1
            return "ok"
        time.sleep(0.05)
        return str(state["writes"])

    pad = "\n".join(f"    pad{i} = {i}" for i in range(30))
    code = f"seen = []\nfor k in ['a', 'b', 'c']:\n    seen.append(kv('get', k))\n{pad}\n    kv('set', k)\nanswer['content'] = ','.join(seen)\nanswer['ready'] = True"
    turn = repl.run_turn(stream(code))
    assert turn.error is None and turn.answer == "0,1,2"


def test_the_harness_is_generic_any_tool_any_loop():
    repl = SpecRepl(judge=ReadsOnly())
    starts = []

    @repl.tool(early_when="reads only")
    def sql(query: str) -> str:
        """Run a SQL statement."""
        starts.append(time.perf_counter())
        time.sleep(0.3)
        return f"rows({query})"

    code = "qs = {'a': 'get users', 'b': 'get orders', 'c': 'get items'}\nout = []\nfor name, q in sorted(qs.items()):\n    out.append(sql(q))\nanswer['content'] = '|'.join(out)\nanswer['ready'] = True"
    turn = repl.run_turn(stream(code))
    assert turn.answer == "rows(get users)|rows(get orders)|rows(get items)"
    assert len(starts) == 3 and max(starts) - min(starts) < 0.2


def test_separate_lines_fan_out_just_like_a_loop():
    # no loop at all: five statements, each with its own call, and a dependent call after them
    lines = "\n".join(f"v{i} = kv('get', 'k{i}')" for i in range(5))
    code = f"{lines}\nb = ask(v0 + v4)\nkv('set', 'z')\nanswer['content'] = b\nanswer['ready'] = True"
    repl, events, calls = make(latency=0.4)
    turn = repl.run_turn(stream(code))
    assert turn.error is None and turn.answer == "answer(get:k0get:k4)"
    gets = sorted(t for t, op, _ in calls if op == "get")
    assert len(gets) == 5 and gets[-1] - gets[0] < 0.35  # side by side; in a row would be 1.6 s
    assert len(calls) == 7  # five reads, the question, the write: each exactly once
    set_t = next(t for t, op, _ in calls if op == "set")
    assert set_t >= commit_time(repl, events) and set_t > max(gets)


def test_a_loop_over_values_only_known_at_run_time_still_fans_out():
    code = "found = ask('list').split()\nout = []\nfor k in found:\n    out.append(kv('get', k))\nanswer['content'] = ','.join(out)\nanswer['ready'] = True"
    repl = SpecRepl(judge=ReadsOnly())
    starts = []

    @repl.tool(early=True)
    def ask(prompt: str) -> str:
        """Ask a sub-model."""
        time.sleep(0.2)
        return "a b c"

    @repl.tool(early_when="reads only")
    def kv(op: str, key: str = "") -> str:
        """Key-value store."""
        starts.append(time.perf_counter())
        time.sleep(0.3)
        return f"{op}:{key}"

    turn = repl.run_turn(stream(code, tok_per_s=2000))
    assert turn.answer == "get:a,get:b,get:c"
    assert len(starts) == 3 and max(starts) - min(starts) < 0.2


def test_the_same_call_twice_runs_twice():
    repl, _, calls = make()
    turn = repl.run_turn(
        stream(
            "a = kv('get', 'k')\nb = kv('get', 'k')\nanswer['content'] = a + b\nanswer['ready'] = True"
        )
    )
    assert turn.answer == "get:kget:k" and len(calls) == 2


def test_a_block_that_never_parses_never_runs_a_refused_call():
    repl, _, calls = make()
    turn = repl.run_turn(
        stream("a = kv('get', 'k1')\nkv('set', 'k2')\nsend('hi')\nthis is not python (((")
    )
    assert turn.error and "SyntaxError" in turn.error
    assert [op for _, op, _ in calls] == [
        "get"
    ]  # the allowed read ran; the write and the send never did


def test_a_runtime_error_stops_the_turn_and_later_calls_never_run():
    repl, _, calls = make()
    turn = repl.run_turn(stream("a = kv('get', 'k1')\nboom = 1 / 0\nkv('set', 'k2')"))
    assert "ZeroDivisionError" in turn.error
    assert [op for _, op, _ in calls] == ["get"]


@pytest.mark.parametrize(
    "judge", [type("Broken", (), {"decide": lambda self, **_: 1 / 0})(), ReadsOnly(delay=1.0)]
)
def test_a_broken_or_slow_judge_means_wait_not_skip(judge):
    repl, events, calls = make(judge=judge, worker=JudgeWorker(timeout_s=0.1))
    turn = repl.run_turn(stream(CODE))
    assert turn.answer == "get:k1|set:k2|get:k3"
    assert len(calls) == 3 and all(t >= commit_time(repl, events) for t, _, _ in calls)


def test_model_code_cannot_swallow_an_abort():
    repl, _, calls = make()
    code = "try:\n    kv('set', 'k')\nexcept Exception:\n    pass\nsend('after')\n((("
    turn = repl.run_turn(stream(code))
    assert "SyntaxError" in turn.error and calls == []


def test_async_tools_work_out_of_the_box_and_share_one_event_loop():
    repl = SpecRepl(
        judge=ReadsOnly(), max_parallel=2
    )  # async early starts must not need threads
    loops, starts, writes = set(), [], []

    @repl.tool(early_when="reads only")
    async def fetch(op: str, key: str = "") -> str:
        """Read a record."""
        loops.add(id(asyncio.get_running_loop()))
        starts.append(time.perf_counter())
        await asyncio.sleep(0.3)
        return f"{op}:{key}"

    @repl.tool()
    async def store(text: str) -> str:
        """Write a record. Changes state."""
        loops.add(id(asyncio.get_running_loop()))
        writes.append(time.perf_counter())
        return "stored"

    code = "out = []\nfor i in range(20):\n    out.append(fetch('get', str(i)))\nstore(','.join(out))\nanswer['content'] = out[-1]\nanswer['ready'] = True"
    events = []
    repl.on_event = lambda kind, **d: events.append((kind, d))
    t0 = time.perf_counter()
    turn = repl.run_turn(stream(code, tok_per_s=2000))
    wall = time.perf_counter() - t0
    assert turn.error is None and turn.answer == "get:19"
    assert (
        len(starts) == 20 and max(starts) - min(starts) < 0.2
    )  # twenty at once on two threads
    assert wall < 1.5  # one after another would be 6 s
    assert len(loops) == 1  # every async tool ran on the same loop
    assert (
        len(writes) == 1 and writes[0] > max(starts) and writes[0] >= commit_time(repl, events)
    )
    repl.close()


def test_sync_and_async_tools_mix_in_one_block():
    repl = SpecRepl(judge=ReadsOnly())

    @repl.tool(early=True)
    async def ask(prompt: str) -> str:
        """Ask a sub-model."""
        await asyncio.sleep(0.05)
        return f"answer({prompt})"

    @repl.tool(early_when="reads only")
    def kv(op: str, key: str = "") -> str:
        """Key-value store."""
        return f"{op}:{key}"

    turn = repl.run_turn(
        stream("a = kv('get', 'k')\nanswer['content'] = ask(a)\nanswer['ready'] = True")
    )
    assert turn.error is None and turn.answer == "answer(get:k)"
    assert "async" not in repl.system_prompt() and "await" in repl.system_prompt()


# --------------------------------------------------------------------------- pieces
def test_extract_code():
    assert extract_code("hi\n```repl\nx = 1\n") == ("x = 1\n", False)
    assert extract_code("hi\n```repl\nx = 1\n```\nbye") == ("x = 1\n", True)
    assert extract_code("no fence") == ("", False)


def test_closed_statements_waits_for_the_next_top_level_line():
    lines = (
        "for k in ks:\n    a = f(k)\n    if a:\n        g(a)\n    else:\n        h(a)".split(
            "\n"
        )
    )
    assert closed_statements(lines, 0, final=False) == ([], 0)  # the loop may still grow
    stmts, upto = closed_statements(lines + ["done = 1"], 0, final=False)
    assert [type(s).__name__ for s in stmts] == ["For"] and upto == len(lines)


def test_closed_statements_handles_else_comments_and_multiline_literals():
    lines = ["if a:", "    x = 1", "else:", "    x = 2", "# note", "    ", "y = ["]
    stmts, upto = closed_statements(lines, 0, final=False)
    assert [type(s).__name__ for s in stmts] == [
        "If"
    ] and upto == 6  # not split at `else:` or the comment
    lines = ["urls = [", "    'a',", "]", "z = 1"]
    stmts, upto = closed_statements(lines, 0, final=False)
    assert [type(s).__name__ for s in stmts] == ["Assign"] and upto == 3
    lines = [
        "for u in us:",
        "    a = f(u)",
        "# unindented comment inside the loop",
        "    b = g(a)",
    ]
    assert closed_statements(lines, 0, final=False) == ([], 0)


def plan(src, env=None):
    return plan_calls(ast.parse(src).body[0], {"kv", "ask"}, env or {})[0]


def may_continue(src, env=None, always=frozenset({"ask"})):
    return plan_calls(ast.parse(src).body[0], {"kv", "ask"}, env or {}, always)[1]


def test_planning_looks_past_a_statement_only_when_nothing_in_it_could_change_state():
    assert may_continue("x = kv('get', 'a')")  # known call: its verdict decides
    assert may_continue("x = ask(prompt)")  # unknown arguments, but the tool is always safe
    assert not may_continue(
        "x = kv(cmd)"
    )  # unknown arguments on a judged tool: could be a write
    assert not may_continue("if c:\n    kv('set', 'a')")
    assert may_continue("if c:\n    y = 1")
    assert may_continue("def helper():\n    return kv('set', 'a')")  # defines, does not call
    assert not may_continue(
        "for u in us:\n    p = kv('get', u)\n    if p:\n        kv('set', u)", {"us": ["a"]}
    )
    assert may_continue(
        "for u in us:\n    p = kv('get', u)\n    if p:\n        n = 1", {"us": ["a"]}
    )


def test_plan_resolves_literals_fstrings_and_unrolled_loops():
    assert plan("x = kv('get', 'a')") == [("kv", ("get", "a"), {})]
    assert plan("for u in urls:\n    p = kv(f'get {u}')", {"urls": ["a", "b"]}) == [
        ("kv", ("get a",), {}),
        ("kv", ("get b",), {}),
    ]
    assert plan("for i, u in enumerate(urls):\n    kv('get', u + str(i))", {"urls": ["a"]}) == [
        ("kv", ("get", "a0"), {})
    ]


def test_plan_handles_dict_views_on_real_dicts_only():
    assert plan(
        "for k, v in sorted(d.items()):\n    kv(v, k)", {"d": {"b": "get", "a": "get"}}
    ) == [("kv", ("get", "a"), {}), ("kv", ("get", "b"), {})]

    class Sneaky(dict):
        def items(self):
            raise AssertionError("planning must never run model-written code")

    assert plan("for k, v in d.items():\n    kv(v, k)", {"d": Sneaky(a="get")}) == []


def test_plan_follows_plain_assignments_it_can_already_work_out():
    src = "for k in ks:\n    key = 'pre-' + k\n    v = kv('get', key)\n    key = helper(v)\n    w = kv('get', key)"
    # it follows `key = 'pre-' + k`, and stops where it cannot know (`key = helper(v)`)
    assert plan(src, {"ks": ["a", "b"], "helper": str}) == [("kv", ("get", "pre-a"), {})]


def test_a_loop_that_builds_its_argument_on_the_line_before_still_fans_out():
    code = "out = []\nfor k in ['a', 'b', 'c']:\n    key = f'item-{k}'\n    out.append(kv('get', key))\nanswer['content'] = ','.join(out)\nanswer['ready'] = True"
    repl, _, calls = make(latency=0.4)
    turn = repl.run_turn(stream(code))
    assert turn.answer == "get:item-a,get:item-b,get:item-c"
    starts = sorted(t for t, _, _ in calls)
    assert len(starts) == 3 and starts[-1] - starts[0] < 0.2


def test_values_set_on_earlier_lines_that_have_not_run_yet_are_followed_too():
    code = "base = 'item'\nfirst = base + '-1'\na = kv('get', first)\nb = kv('get', base + '-2')\nanswer['content'] = a + b\nanswer['ready'] = True"
    repl, _, calls = make(latency=0.4)
    turn = repl.run_turn(stream(code, tok_per_s=2000))
    assert turn.answer == "get:item-1get:item-2"
    starts = sorted(t for t, _, _ in calls)
    assert len(starts) == 2 and starts[-1] - starts[0] < 0.2


def test_plan_never_guesses():
    assert plan("x = kv('get', unknown)") == []
    assert (
        plan("x = kv('get', helper())", {"helper": lambda: "a"}) == []
    )  # would run model code
    assert plan("if c:\n    kv('get', 'a')", {"c": True}) == []  # may never execute
    assert plan("for u in urls:\n    p = kv('get', u)\n    q = ask(p)", {"urls": ["a"]}) == [
        ("kv", ("get", "a"), {})
    ]  # q needs p
    assert plan("for u in urls:\n    if u:\n        kv('get', u)", {"urls": ["a"]}) == []
    assert plan("x = kv('get', 'a') if c else 1", {"c": False}) == []  # might not run
    assert plan("x = c and kv('get', 'a')", {"c": False}) == []
    assert plan("xs = [kv('get', k) for k in ks]", {"ks": ["a"]}) == []
    # stops at the first call it cannot resolve, so it never reorders around it
    assert plan("x = ask(kv('get', v)) + kv('get', 'b')", {}) == []
    assert (
        plan(
            "for u in sorted(urls):\n    kv('get', u)",
            {"urls": ["b", "a"], "sorted": lambda x: x},
        )
        == []
    )  # rebound builtin
