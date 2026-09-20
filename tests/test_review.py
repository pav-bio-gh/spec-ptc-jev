"""One test per defect found in the independent review. Each reproduces the reviewer's probe."""

import ast
import os
import random
import threading
import time

import pytest

from spec_ptc_jev import Decision, SpecRepl
from spec_ptc_jev.repl import _static_eval, _static_ok, extract_code, learn, plan_calls


class Yes:
    def decide(self, **_):
        return Decision(True, 1.0, 0.0)


def chunks(code, tok_per_s=400, fence="```repl", tail=""):
    text = f"{fence}\n{code}\n```\n{tail}"
    for i in range(0, len(text), 4):
        time.sleep(4 / tok_per_s / 4)
        yield text[i : i + 4]


def make(latency=0.0, events=None):
    calls = []
    repl = SpecRepl(
        judge=Yes(),
        on_event=(lambda kind, **d: events.append((kind, d))) if events is not None else None,
    )

    @repl.tool(early_when="reads only")
    def kv(op: str, key: str = "") -> str:
        """Key-value store."""
        calls.append((time.perf_counter(), op, key))
        time.sleep(latency)
        return f"{op}:{key}"

    return repl, calls


PAD = "\n".join(f"pad{i} = {i}" for i in range(40))


# 1. planning must never write into the program's live objects
def test_planning_never_performs_the_programs_assignments():
    repl, _ = make()
    code = f"d = {{'n': 0}}\n{PAD}\nimport time\ntime.sleep(0.4)\nseen = dict(d)\nd['n'] = 5\nfinal_answer(str(seen))"
    assert (
        repl.run_turn(chunks(code)).answer == "{'n': 0}"
    )  # the line BEFORE `d['n'] = 5` sees 0

    live = {"n": 0}
    scope = {"d": live}
    learn(ast.parse("d['n'] = 5").body[0], scope)
    assert (
        live == {"n": 0} and "d" not in scope
    )  # never written through; forgotten until it really runs

    os.environ.pop("SPEC_REPL_PLANNED", None)
    turn = make()[0].run_turn(
        chunks(
            f"import os\nbefore = os.environ.get('SPEC_REPL_PLANNED')\n{PAD}\nos.environ['SPEC_REPL_PLANNED'] = 'written'\nfinal_answer(str(before))"
        )
    )
    os.environ.pop("SPEC_REPL_PLANNED", None)
    assert turn.answer == "None"


# 2. no API key must not hang or error: it just means nothing starts early
def test_no_api_key_means_a_normal_run(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    log = []
    repl = SpecRepl()  # the real JevJudge cannot be built

    @repl.tool(early_when="reads only")
    def kv(op: str, key: str = "") -> str:
        """Key-value store."""
        log.append(op)
        return f"{op}:{key}"

    box = {}
    th = threading.Thread(
        target=lambda: box.setdefault(
            "t", repl.run_turn(chunks("v = kv('get', 'k')\nfinal_answer(v)"))
        ),
        daemon=True,
    )
    th.start()
    th.join(10)
    assert (
        not th.is_alive()
        and box["t"].error is None
        and box["t"].answer == "get:k"
        and log == ["get"]
    )


def test_an_argument_whose_repr_raises_does_not_hang():
    repl, _ = make()
    code = "class Bad:\n    def __repr__(self):\n        raise ValueError('no repr')\nb = Bad()\ntry:\n    kv('get', b)\nexcept Exception as e:\n    final_answer('handled ' + type(e).__name__)"
    box = {}
    th = threading.Thread(
        target=lambda: box.setdefault("t", repl.run_turn(chunks(code))), daemon=True
    )
    th.start()
    th.join(10)
    assert not th.is_alive()


# 3. planning must not use up the program's iterators
def test_planning_does_not_consume_the_programs_iterators():
    for source in (
        "iter(['a', 'b', 'c'])",
        "map(str, ['a', 'b', 'c'])",
        "(x for x in ['a', 'b', 'c'])",
    ):
        repl, calls = make()
        code = f"items = {source}\n{PAD}\nout = []\nfor k in items:\n    out.append(kv('get', k))\nfinal_answer(','.join(out))"
        turn = repl.run_turn(chunks(code))
        assert turn.answer == "get:a,get:b,get:c", source
        assert len(calls) == 3, source  # and no orphan calls


def test_a_huge_range_is_not_materialised():
    t0 = time.perf_counter()
    plan, _ = plan_calls(
        ast.parse("for i in range(1000000000):\n    kv('get', str(i))").body[0], {"kv"}, {}
    )
    assert len(plan) == 64 and time.perf_counter() - t0 < 0.5


# 4 / 5. what planning may evaluate: built-in data only, and nothing that calls out
@pytest.mark.parametrize(
    "expr",
    [
        "helper()",
        "obj.attr",
        "obj.method()",
        "x.pop()",
        "[f(i) for i in xs]",
        "(y := 1)",
        "a if b else c",
        "not a",
        "a < b",
        "-a",
        "lambda: 1",
        "open('f')",
        "__import__('os')",
        "getattr(a, 'b')",
        "a @ b",
        "a ** b",
        "a / b",
    ],
)
def test_the_whitelist_rejects_everything_that_could_run_code(expr):
    assert not _static_ok(ast.parse(expr, mode="eval").body)


def test_planning_refuses_any_value_that_is_not_built_in_data():
    ran = []

    class Loud:
        def __format__(self, spec):
            ran.append("format")
            return "x"

        def __add__(self, other):
            ran.append("add")
            return "x"

        def __getitem__(self, i):
            ran.append("getitem")
            return "x"

        def __iter__(self):
            ran.append("iter")
            return iter("x")

    class Sub(dict):
        def __iter__(self):
            ran.append("sub iter")
            return iter(())

    env = {"o": Loud(), "s": Sub(a=1), "plain": {"k": ["v", 1, (2.0, None)]}}
    for src in (
        "f'{o}'",
        "o + 'a'",
        "o[0]",
        "list(o)",
        "sorted(s)",
        "str(o)",
        "len(o)",
        "[o]",
        "s.items()",
    ):
        with pytest.raises(Exception):  # noqa: B017, PT011 (any refusal is fine; running it is not)
            _static_eval(ast.parse(src, mode="eval").body, env)
    assert ran == []
    assert _static_eval(ast.parse("plain['k'][0] + '!'", mode="eval").body, env) == "v!"


def test_helper_calls_in_arguments_are_never_run_by_planning():
    ran = []
    repl, calls = make()
    repl.ns["helper"] = lambda: ran.append(threading.current_thread().name) or "k"
    turn = repl.run_turn(chunks(f"{PAD}\nv = kv('get', helper())\nfinal_answer(v)"))
    assert turn.answer == "get:k" and ran == ["spec-repl-exec"] and len(calls) == 1


# 6. a syntax error is always reported, with the model's own line number
def test_a_trailing_syntax_error_is_reported_with_its_line():
    repl, _ = make()
    turn = repl.run_turn(chunks("x = 1\nprint('ran x')\ny = ((("))
    assert turn.answer is None and "SyntaxError" in turn.error and "line 3" in turn.error
    plain = SpecRepl()
    assert "line 3" in plain.run_turn(chunks("x = 1\nprint('ran x')\ny = (((")).error


def test_a_runtime_error_names_its_line():
    turn = make()[0].run_turn(chunks("a = 1\nb = 2\nc = a / 0\nd = 4"))
    assert "ZeroDivisionError" in turn.error and "line 3" in turn.error


# 7. prose is never run as code, whatever the fence looks like
@pytest.mark.parametrize(
    "fence",
    [
        "```Python",
        "```py3",
        "``` python",
        "```python3",
        "```repl\r",
        "```",
        "```PY",
        "~~~python",
    ],
)
def test_code_fences_of_every_spelling_run_the_code_and_never_the_prose(fence):
    close = "~~~" if fence.startswith("~") else "```"
    text = f"Sure.\n{fence}\nprint('hello')\n{close}\nPROSE_AFTER_FENCE = 1\nmore prose\n"
    code, closed = extract_code(text, final=True)
    assert code.replace("\r", "") == "print('hello')\n" and closed
    turn = SpecRepl().run_turn(iter([text]))
    assert (
        turn.error is None
        and turn.stdout == "hello\n"
        and "PROSE_AFTER_FENCE" not in SpecRepl().ns
    )


def test_non_code_blocks_are_skipped_and_never_confuse_the_pairing():
    text = "Data:\n```json\n{\"a\": 1}\n```\nNow:\n```python\nfinal_answer('ok')\n```\ntrailing prose\n"
    assert extract_code(text, final=True)[0] == "final_answer('ok')\n"
    assert SpecRepl().run_turn(iter([text])).answer == "ok"
    assert extract_code("```repl\nx = 1\n```", final=True) == (
        "x = 1\n",
        True,
    )  # no newline after the last fence


# 8. a tool name the model redefines is no longer our tool
def test_a_redefined_tool_name_is_not_started_early():
    repl, calls = make()
    code = f"def kv(op, key=''):\n    return 'shadowed'\n{PAD}\nv = kv('get', 'secret')\nfinal_answer(v)"
    turn = repl.run_turn(chunks(code))
    assert turn.answer == "shadowed" and calls == []  # the real tool never ran


# 10. what is dropped is cancelled
def test_unused_early_calls_are_cancelled_when_the_turn_ends():
    started, finished = [], []
    repl = SpecRepl(
        judge=Yes(), max_parallel=1
    )  # one worker: the rest are queued, so they can be cancelled

    @repl.tool(early_when="reads only")
    def slow(op: str, key: str = "") -> str:
        """Slow read."""
        started.append(key)
        time.sleep(0.3)
        finished.append(key)
        return key

    turn = repl.run_turn(
        chunks(
            "for k in ['a', 'b', 'c', 'd']:\n    v = slow('get', k)\n    break\nfinal_answer(v)"
        )
    )
    assert turn.answer == "a"
    time.sleep(0.8)
    # "b" may begin the instant the single worker frees up; "c" and "d" were still queued when
    # the turn ended, so they must have been cancelled
    assert "c" not in started and "d" not in started


# 11. a slow listener must not slow async tools down
def test_a_slow_listener_does_not_serialise_async_tools():
    import asyncio

    def listener(kind, **_):
        if kind == "call_start":
            time.sleep(0.2)

    repl = SpecRepl(judge=Yes(), on_event=listener)
    starts = []

    @repl.tool(early_when="reads only")
    async def fetch(key: str) -> str:
        """Read."""
        starts.append(time.perf_counter())
        await asyncio.sleep(0.1)
        return key

    turn = repl.run_turn(
        chunks(
            "out = []\nfor k in ['a', 'b', 'c', 'd', 'e', 'f']:\n    out.append(fetch(k))\nfinal_answer(''.join(out))",
            tok_per_s=4000,
        )
    )
    assert turn.answer == "abcdef" and max(starts) - min(starts) < 0.3


# tests that prove EARLY start, not just parallelism
def test_calls_really_start_before_the_code_is_complete():
    events = []
    repl, calls = make(latency=0.2, events=events)
    lines = "\n".join(f"v{i} = kv('get', 'k{i}')" for i in range(4))
    turn = repl.run_turn(
        chunks(f"{lines}\n{PAD}\n{PAD.replace('pad', 'more')}\nfinal_answer(v0 + v3)")
    )
    commit = repl._t0 + next(d["t"] for k, d in events if k == "commit")
    assert turn.answer == "get:k0get:k3" and len(calls) == 4
    assert all(
        t < commit for t, _, _ in calls
    )  # every one of them began while the model was still writing
    assert [d["early"] for k, d in events if k == "call_start"] == [True] * 4


def test_print_to_a_file_goes_to_that_file(capsys):
    turn = SpecRepl().run_turn(
        iter(
            [
                "```repl\nimport sys\nprint('to stderr', file=sys.stderr)\nprint('to the model')\n```"
            ]
        )
    )
    assert turn.stdout == "to the model\n" and "to stderr" in capsys.readouterr().err


def test_data_changed_in_place_by_a_line_that_has_not_run_is_not_trusted():
    repl, calls = make()
    code = f"cfg = {{'k': 'old'}}\n{PAD}\ncfg['k'] = 'new'\nv = kv('get', cfg['k'])\nfinal_answer(v)"
    turn = repl.run_turn(chunks(code))
    assert turn.answer == "get:new" and [k for _, _, k in calls] == [
        "new"
    ]  # no wasted call for 'old'
    for change in (
        "xs.append('c')",
        "xs.clear()",
        "del xs[0]",
        "random.shuffle(xs)",
        "xs += ['c']",
    ):
        repl, calls = make()
        code = f"import random\nxs = ['a', 'b']\n{PAD}\n{change}\nout = [kv('get', x) for x in xs]\nfor x in xs:\n    out.append(kv('get', x))\nfinal_answer(str(len(out)))"
        turn = repl.run_turn(chunks(code))
        assert turn.error is None, (change, turn.error)
        assert len(calls) == int(turn.answer), (
            change
        )  # every call made was one the program used


def test_a_loop_target_that_is_not_a_plain_name_is_never_bound_by_planning():
    live = {"k": "start"}
    plan, _ = plan_calls(
        ast.parse("for cfg['k'] in ['a', 'b']:\n    kv('get', cfg['k'])").body[0],
        {"kv"},
        {"cfg": live},
    )
    assert live == {"k": "start"} and plan == []


def test_no_call_is_ever_started_twice_under_contention():
    # A judge that answers in 5-20 ms and a near-instant tool put the planner and the executor on
    # the same call at the same moment. Found live: one run in twenty opened a page twice.
    class Jittery:
        def decide(self, **_):
            time.sleep(random.uniform(0.005, 0.02))
            return Decision(True, 1.0, 0.0)

    doubles = 0
    for _ in range(60):
        calls = []
        repl = SpecRepl(judge=Jittery())

        @repl.tool(early_when="reads only")
        def kv(op: str, key: str = "", calls=calls) -> str:
            """Key-value store."""
            calls.append(key)
            time.sleep(random.uniform(0.0, 0.005))
            return key

        lines = "\n".join(f"v{i} = kv('get', 'k{i}')" for i in range(8))
        turn = repl.run_turn(chunks(f"{lines}\nfinal_answer(v7)", tok_per_s=3000))
        repl.close()
        assert turn.answer == "k7"
        doubles += len(calls) - 8
    assert doubles == 0


def test_a_missing_api_key_is_said_out_loud(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    repl = SpecRepl()

    @repl.tool(early_when="reads only")
    def kv(op: str, key: str = "") -> str:
        """Key-value store."""
        return key

    with pytest.warns(UserWarning, match="no tool call will start early"):
        turn = repl.run_turn(chunks("v = kv('get', 'a')\nfinal_answer(v)"))
    repl.close()
    assert turn.answer == "a"
