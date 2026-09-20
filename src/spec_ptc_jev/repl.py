"""A streaming REPL for code-writing agents, with Jev deciding which tool calls may start early.

The model streams one block of Python that calls tools. Three threads share the work:

  stream    the caller's thread. Takes tokens off the model and hands the text on. Never waits.
  planner   reads AHEAD over everything written so far. For every tool call whose arguments are
            already known it asks the judge, all at once, and starts the allowed ones side by
            side. Five separate `browse(...)` lines, or one `for url in urls:` loop, or a loop
            still being typed: all become parallel calls while the model is still writing.
  executor  runs the code exactly once, statement by statement, as each statement is complete.
            A call that was started early just picks up its result. A call that was NOT allowed
            early blocks right there until the block is complete and parses ("commit"), then
            runs in place.

This is generic: any registered tool (plain or `async def`), any code shape. Nothing here
knows about a task.

Guarantees:
  - the code runs once, in one namespace, in program order.
  - a tool call only starts early AFTER its verdict says yes. A judge error or timeout is a no.
  - a call that was not allowed early never runs before commit, and never runs at all if the
    block does not parse.
  - no stale results ACROSS TOOLS: when a call that was not allowed early runs, every early
    result not yet used is dropped (and cancelled if it can be) and nothing new starts until
    it has finished. This sees only registered tools. If a tool's result depends on something
    the program changes in plain Python (a file it writes, os.environ, a global), do not mark
    that tool early: an early read can land before the change.
  - planning never runs model-written code: arguments are evaluated only when they are literals,
    names bound to plain built-in data, f-strings, indexing, `+ % *`, containers and a few pure
    builtins.

One difference from "generate everything, then run": statements run as they arrive, like an
interactive Python session. If a LATER line turns out to be a syntax error, the earlier
statements have already run (tool calls among them only if the judge allowed them).

    repl = SpecRepl()

    @repl.tool(early_when="Reading only. Never anything that writes, sends or submits.")
    def sql(query: str) -> list: ...

    @repl.tool(early=True)          # always safe to start early
    def llm_query(prompt: str) -> str: ...

    result = repl.run(chat, "the task")           # or: await repl.arun(chat, "the task")
    print(result.answer)
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import inspect
import itertools
import re
import threading
import time
import traceback
import warnings
from collections import deque
from collections.abc import AsyncIterable, Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from spec_ptc_jev.judge import Decision, Judge
from spec_ptc_jev.worker import JudgeWorker, default_worker

_FENCE = re.compile(r"^(```|~~~)[ \t]*([A-Za-z0-9_+.-]*)[ \t]*$")
_CODE_TAGS = {"", "repl", "python", "python3", "py", "py3"}
_CONTINUES = {"else", "elif", "except", "finally"}  # a line starting with these extends a block
_PURE_BUILTINS = (
    "zip",
    "enumerate",
    "range",
    "sorted",
    "reversed",
    "list",
    "tuple",
    "len",
    "str",
    "int",
)
_SIMPLE = (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)
_MAYBE = (
    ast.IfExp,
    ast.BoolOp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Lambda,
)
MAX_UNROLL = 64
MAX_FIELD_CHARS = 600

ALLOWED = Decision(True, 1.0, 0.0, reason="tool is marked early=True")
NO_POLICY = Decision(False, 0.0, 1.0, reason="tool has no early policy")

Call = tuple[str, tuple, dict]


class TurnAborted(BaseException):
    """Raised inside a blocked tool call when the code block never became valid. BaseException
    so that model-written `except Exception` cannot swallow it."""


@dataclass
class Turn:
    code: str
    stdout: str
    error: str | None
    answer: str | None


@dataclass
class _Tool:
    name: str
    fn: Callable[..., Any]
    early: bool
    policy: str | None
    description: str
    reduce: Callable[[tuple, dict], Any]
    is_async: bool = False


# --------------------------------------------------------------------------- reading the stream
def extract_code(text: str, final: bool = False) -> tuple[str, bool]:
    """The code in `text`: every fenced code block, in order, as one program. Also whether the
    last fence seen is closed. Every fence is paired (whatever its tag) so a closing fence can
    never be mistaken for an opening one; only blocks tagged as Python (or untagged) are code.
    Only ever grows as `text` grows, apart from a last, still-incomplete line."""
    code: list[str] = []
    tag: str | None = None  # None: outside a block
    mark = ""  # the fence that opened the current block: it takes the same one to close it
    lines = text.replace("\r\n", "\n").split("\n")
    for n, line in enumerate(lines):
        last = not final and n == len(lines) - 1  # may still be growing: not a fence yet
        m = None if last else _FENCE.match(line)
        if tag is None:
            if m is not None:
                mark, tag = m.group(1), m.group(2).lower()
        elif m is not None and m.group(1) == mark and m.group(2) == "":
            tag = None
        elif tag in _CODE_TAGS:
            code.append(line if last else line + "\n")
    return "".join(code), tag is None and bool(code)


def closed_statements(lines: list[str], start: int, final: bool) -> tuple[list[ast.stmt], int]:
    """Statements in `lines[start:]` that can no longer change, and the line index after them.

    While streaming, a statement is closed once a LATER complete line starts a new top-level
    statement (so a loop is not closed while its body may still grow). When `final`, everything
    left is closed; a SyntaxError propagates."""
    if final:
        return _numbered(ast.parse("\n".join(lines[start:])).body, start), len(lines)
    for k in range(len(lines) - 1, start, -1):
        nxt = lines[k]
        if not nxt.strip() or nxt[0] in " \t)]}#":  # blank, indented, closer, or a comment
            continue
        word = re.match(r"[A-Za-z_]+", nxt)
        if word and word.group(0) in _CONTINUES:
            continue
        try:
            return _numbered(ast.parse("\n".join(lines[start:k])).body, start), k
        except SyntaxError:
            continue
    return [], start


def _numbered(body: list[ast.stmt], start: int) -> list[ast.stmt]:
    for stmt in body:
        ast.increment_lineno(stmt, start)  # line numbers count from the top of the model's code
    return body


def open_statement(lines: list[str], start: int) -> ast.stmt | None:
    """The statement being written right now, as far as its complete lines parse. Complete
    lines cannot change, so what is there is certain; the statement may still grow."""
    for k in range(len(lines), start, -1):
        try:
            body = ast.parse("\n".join(lines[start:k])).body
        except SyntaxError:
            continue
        return body[0] if body else None
    return None


# --------------------------------------------------------------------------- knowing arguments early
def _is_dict_view(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("items", "keys", "values")
        and not node.args
        and not node.keywords
    )


def _static_ok(node: ast.AST) -> bool:
    """True if evaluating `node` cannot run model-written code."""
    if isinstance(node, (ast.Constant, ast.Name)):
        return True
    if isinstance(node, ast.JoinedStr):
        return all(_static_ok(v) for v in node.values)
    if isinstance(node, ast.FormattedValue):
        return _static_ok(node.value) and (
            node.format_spec is None or _static_ok(node.format_spec)
        )
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, (ast.Add, ast.Mod, ast.Mult))
            and _static_ok(node.left)
            and _static_ok(node.right)
        )
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_static_ok(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _static_ok(k) for k in node.keys) and all(
            _static_ok(v) for v in node.values
        )
    if isinstance(node, ast.Subscript):
        return _static_ok(node.value) and _static_ok(node.slice)
    if isinstance(node, ast.Slice):
        return all(p is None or _static_ok(p) for p in (node.lower, node.upper, node.step))
    if _is_dict_view(
        node
    ):  # d.items() / d.keys() / d.values(); the receiver must be a real dict
        return _static_ok(node.func.value)  # type: ignore[attr-defined]
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Name)
            and node.func.id in _PURE_BUILTINS
            and not node.keywords
            and all(_static_ok(a) for a in node.args)
        )
    return False


_PLAIN = (str, int, float, bool, bytes, type(None), range)


def _plain(value: Any, budget: list[int] | None = None) -> bool:
    """Built-in data only, all the way down. A model-defined object could run code from
    `__format__`, `__add__`, `__iter__`, `__repr__` ... so planning refuses to touch one."""
    budget = budget if budget is not None else [10_000]
    budget[0] -= 1
    if budget[0] < 0:
        return False
    if type(value) in _PLAIN:
        return True
    if type(value) in (list, tuple, set, frozenset):
        return all(_plain(v, budget) for v in value)
    if type(value) is dict:
        return all(_plain(k, budget) and _plain(v, budget) for k, v in value.items())
    return False


def _snapshot(ns: dict[str, Any]) -> dict[str, Any]:
    """A copy of the namespace taken while another thread may be adding names to it."""
    for _ in range(5):
        try:
            return dict(ns)
        except RuntimeError:  # it changed size while we copied; try again
            continue
    return {}


def _static_eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if not _static_ok(node):
        raise ValueError("not statically known")
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Name)
            and n.id in env
            and n.id not in _PURE_BUILTINS
            and not _plain(env[n.id])
        ):
            raise ValueError("not plain data")
    for n in ast.walk(node):
        if _is_dict_view(n):
            if type(_static_eval(n.func.value, env)) is not dict:  # type: ignore[attr-defined]
                raise ValueError(
                    "not a plain dict"
                )  # a model-written class could run code here
        elif isinstance(n, ast.Call):
            real = getattr(builtins, n.func.id)  # type: ignore[attr-defined]
            if env.get(n.func.id, real) is not real:  # type: ignore[attr-defined]
                raise ValueError("builtin was rebound")  # then it is no longer known to be pure
    expr = ast.fix_missing_locations(ast.Expression(body=node))
    scope = {name: getattr(builtins, name) for name in _PURE_BUILTINS}
    scope.update(env)
    return eval(compile(expr, "<plan>", "eval"), {"__builtins__": {}}, scope)  # noqa: S307 (whitelisted AST)


def _tool_calls(node: ast.AST, names: set[str]) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
    ]


def _certain_calls(stmt: ast.stmt, names: set[str]) -> tuple[list[ast.Call], list[ast.Call]]:
    """(calls one simple statement will certainly make, in evaluation order; tool calls that
    might not run or might run many times: `a if c else f()`, `x and f()`, comprehensions)."""
    certain: list[ast.Call] = []
    maybe: list[ast.Call] = []

    class V(ast.NodeVisitor):
        def generic_visit(self, node: ast.AST) -> None:
            if isinstance(node, _MAYBE):
                maybe.extend(_tool_calls(node, names))
                return
            super().generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            self.generic_visit(node)  # inner calls first
            if isinstance(node.func, ast.Name) and node.func.id in names:
                certain.append(node)

    V().visit(stmt)
    return certain, maybe


def _assigned(stmt: ast.AST) -> set[str]:
    return {
        n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


def _only_names(target: ast.expr) -> bool:
    if isinstance(target, ast.Name):
        return True
    return isinstance(target, (ast.Tuple, ast.List)) and all(
        _only_names(e) for e in target.elts
    )


def _bind(target: ast.expr, item: Any, scope: dict[str, Any]) -> None:
    """Bind `target = item` in planning's own scratch `scope`. Names only: a subscript or
    attribute target would write through to the program's live objects."""
    if not _only_names(target):
        raise ValueError("planning only binds names")
    mod = ast.Module([ast.Assign([target], ast.Name("__item", ast.Load()))], [])
    exec(
        compile(ast.fix_missing_locations(mod), "<bind>", "exec"),
        {"__builtins__": {}, "__item": item},
        scope,
    )  # noqa: S102 (binds names in a scratch dict)


def _binds(stmt: ast.AST) -> set[str]:
    """Every name `stmt` (re)binds: assignments, loop targets, `def`, `class`, imports."""
    names = {
        n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    for n in ast.walk(stmt):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in n.names)
    return names


def _base(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _touched(stmt: ast.AST) -> set[str]:
    """Names whose VALUE `stmt` may change in place: `cfg['k'] = v`, `del xs[0]`, `xs.append(v)`,
    `random.shuffle(xs)`. Planning forgets them until the statement has really run."""
    names: set[str | None] = set()
    for n in ast.walk(stmt):
        if isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(
            n.ctx, (ast.Store, ast.Del)
        ):
            names.add(_base(n))
        elif isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute):
                names.add(_base(n.func))  # a method call on it
            names.update(
                _base(a) for a in [*n.args, *(k.value for k in n.keywords)]
            )  # handed to a call
    return {n for n in names if n is not None}


def learn(stmt: ast.AST, scope: dict[str, Any]) -> None:
    """What planning knows after `stmt`, without running it. `url = f"https://x/{name}"` is a
    value we can already work out, so remember it. Anything else the statement binds is unknown
    until the statement really runs."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and _only_names(stmt.targets[0]):
        try:
            _bind(stmt.targets[0], _static_eval(stmt.value, scope), scope)
            return
        except Exception:
            pass
    for name in _binds(stmt) | _touched(stmt):
        scope.pop(name, None)


def plan_calls(
    stmt: ast.stmt, names: set[str], env: dict[str, Any], always: frozenset[str] = frozenset()
) -> tuple[list[Call], bool]:
    """(tool calls `stmt` will certainly make, in program order, whose arguments are known NOW;
    whether planning may continue PAST this statement).

    A `for` over a known sequence is unrolled across the straight-line start of its body.
    Planning stops at the first call it cannot see through, unless that call is to a tool in
    `always` (marked early=True: it cannot change state, so looking past it is safe)."""
    plan: list[Call] = []

    def blocks(calls: list[ast.Call]) -> bool:
        return any(c.func.id not in always for c in calls)  # type: ignore[attr-defined]

    def resolve(call: ast.Call, scope: dict[str, Any]) -> bool:
        try:
            if any(k.arg is None for k in call.keywords) or any(
                isinstance(a, ast.Starred) for a in call.args
            ):
                raise ValueError("star arguments")
            args = tuple(_static_eval(a, scope) for a in call.args)
            kwargs = {k.arg: _static_eval(k.value, scope) for k in call.keywords}
        except Exception:
            return call.func.id in always  # type: ignore[attr-defined]  # unknown: skip it, or stop
        plan.append((call.func.id, args, kwargs))  # type: ignore[attr-defined]
        return True

    def simple(s: ast.stmt, scope: dict[str, Any]) -> bool:
        certain, maybe = _certain_calls(s, names)
        if maybe:
            return not blocks(maybe) and not blocks(certain)  # plan nothing from this statement
        return all(resolve(c, scope) for c in certain)

    if isinstance(stmt, _SIMPLE):
        return plan, simple(stmt, env)
    if isinstance(stmt, ast.For) and not stmt.orelse:
        head: list[ast.stmt] = []
        for s in stmt.body:  # the straight-line start of the body
            if not isinstance(s, _SIMPLE):
                break
            head.append(s)
        rest_blocks = blocks([c for s in stmt.body[len(head) :] for c in _tool_calls(s, names)])
        try:
            items = list(itertools.islice(_static_eval(stmt.iter, env), MAX_UNROLL))
        except Exception:
            return plan, not blocks(_tool_calls(stmt, names))
        for item in items:
            scope = dict(env)
            try:
                _bind(stmt.target, item, scope)
            except Exception:
                return plan, False
            for s in head:
                if not simple(s, scope):
                    return plan, False
                learn(s, scope)
        return plan, not rest_blocks
    if isinstance(
        stmt,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Import,
            ast.ImportFrom,
            ast.Pass,
        ),
    ):
        return plan, True  # defines or imports; calls nothing of ours now
    return plan, not blocks(_tool_calls(stmt, names))  # if / while / try / with ...


def _default_reducer(fn: Callable[..., Any]) -> Callable[[tuple, dict], Any]:
    sig = inspect.signature(fn)

    def clip(v: Any) -> Any:
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        text = v if isinstance(v, str) else repr(v)
        return (
            text
            if len(text) <= MAX_FIELD_CHARS
            else text[:MAX_FIELD_CHARS] + f"... [{len(text) - MAX_FIELD_CHARS} more chars]"
        )

    def reduce(args: tuple, kwargs: dict) -> Any:
        try:
            bound = sig.bind(*args, **kwargs).arguments
        except TypeError:
            bound = {"args": list(args), **kwargs}
        return {k: clip(v) for k, v in bound.items()}

    return reduce


# --------------------------------------------------------------------------- one turn's shared state
@dataclass
class _TurnState:
    cond: threading.Condition = field(default_factory=threading.Condition)
    code: str = ""
    final: bool = False  # the block is complete (closing fence or end of stream)
    stmts: list[ast.stmt] = field(default_factory=list)  # closed statements, append-only
    parsed_all: bool = False  # every statement of the final block is in `stmts`
    open_stmt: ast.stmt | None = None  # the statement still being written
    next: int = 0  # index of the statement the executor will run next, or is running
    running: bool = False
    env_version: int = 0  # bumped whenever the executor finishes a statement
    executor_done: bool = False  # the executor has exited, normally or not


# --------------------------------------------------------------------------- the REPL
@dataclass
class Run:
    """The result of `SpecRepl.run` / `arun`."""

    answer: str | None
    turns: list[Turn]

    @property
    def error(self) -> str | None:
        return self.turns[-1].error if self.turns else None


class SpecRepl:
    """See the module docstring. One SpecRepl is one conversation: it keeps the namespace
    between turns and runs one turn at a time."""

    def __init__(
        self,
        *,
        judge: Judge | None = None,
        worker: JudgeWorker | None = None,
        speculate: bool = True,
        max_parallel: int = 16,
        on_event: Callable[..., None] | None = None,
    ) -> None:
        """Nothing starts early unless you mark a tool with `early=True` or `early_when=...`.
        With no marked tools (or `speculate=False`) this is a plain loop: the whole block is
        generated, then run in order, and the judge is never contacted."""
        self._judge = judge
        self._worker = worker
        self.speculate = speculate
        self.on_event = on_event
        self.tools: dict[str, _Tool] = {}
        self.ns: dict[str, Any] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=max_parallel, thread_name_prefix="spec-repl-tool"
        )
        self._notify = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spec-repl-notify")
        self._own_loop: asyncio.AbstractEventLoop | None = (
            None  # only for the sync entry points
        )
        self._loop: asyncio.AbstractEventLoop | None = None  # where async tools run this turn
        self._verdicts: dict[tuple, Future] = {}
        self._lock = threading.Lock()  # verdict cache, call ids
        self._call_ids = 0
        self._turn_lock = threading.Lock()  # one turn at a time
        # per turn
        self._t0 = 0.0
        self._on = False  # is anything allowed to start early this turn
        self._after_commit_only = (
            False  # measurement aid: run allowed calls side by side, but never early
        )
        self._st = _TurnState()
        self._threads: list[threading.Thread] = []
        self._result: dict[str, Any] = {}
        self._text = ""
        self._ready: dict[tuple, deque[Future]] = {}  # early results not yet used, per call
        self._launch = threading.Lock()  # guards _ready; held while a state-changing call runs
        self._commit = threading.Event()
        self._abort: str | None = None
        self._stdout: list[str] = []
        self._hooks: dict[str, Callable[..., Any]] = {}

    def __enter__(self) -> SpecRepl:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Stop the background threads this REPL owns. Never touches your own event loop."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._notify.shutdown(wait=False)
        if self._own_loop is not None:
            self._own_loop.call_soon_threadsafe(self._own_loop.stop)

    # ------------------------------------------------------------------ tools
    def tool(
        self,
        fn: Callable | None = None,
        *,
        early: bool = False,
        early_when: str | None = None,
        name: str | None = None,
        description: str | None = None,
        reduce: Callable[[tuple, dict], Any] | None = None,
    ) -> Callable:
        """Register a function (plain or `async def`) as a tool. Works bare (`@repl.tool`) or
        with options. `early=True`: every call may start early. `early_when="..."`: the judge
        decides per call against that policy. Neither: a normal tool, never started early."""
        if early and early_when:
            raise ValueError("pass early=True or early_when=..., not both")
        if early_when is not None and not early_when.strip():
            raise ValueError("early_when must be a non-empty policy")

        def deco(f: Callable) -> Callable:
            doc = (description or inspect.getdoc(f) or "").strip()
            t = _Tool(
                name or f.__name__,
                f,
                early,
                early_when and early_when.strip(),
                doc,
                reduce or _default_reducer(f),
                inspect.iscoroutinefunction(f),
            )
            self.tools[t.name] = t
            return f

        return deco(fn) if fn is not None else deco

    @property
    def judge(self) -> Judge:
        if self._judge is None:
            from spec_ptc_jev.judge import JevJudge

            self._judge = JevJudge()
        return self._judge

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event is None:
            return
        data["t"] = time.perf_counter() - self._t0  # stamped now, wherever it is delivered
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._deliver(kind, data)
        else:  # on an event loop (an async tool): never run listener code here
            self._notify.submit(self._deliver, kind, data)

    def _deliver(self, kind: str, data: dict[str, Any]) -> None:
        try:
            self.on_event(kind, **data)  # type: ignore[misc]
        except Exception:
            pass  # a broken listener must never break a turn

    @staticmethod
    def _label(name: str, args: tuple, kwargs: dict) -> str:
        parts = [a if isinstance(a, str) else repr(a) for a in args] + [
            f"{k}={v!r}" for k, v in kwargs.items()
        ]
        return f"{name}({', '.join(parts)})"[:200]

    # ------------------------------------------------------------------ verdicts
    @staticmethod
    def _key(name: str, args: tuple, kwargs: dict) -> tuple:
        return (name, repr(args), repr(sorted(kwargs.items())))

    def verdict(self, name: str, args: tuple, kwargs: dict) -> Future:
        """A future `Decision` for one call. Never blocks; asked once per distinct call."""
        tool = self.tools[name]
        key = self._key(name, args, kwargs)
        with self._lock:
            fut = self._verdicts.get(key)
            if fut is not None:
                return fut
            fut = self._verdicts[key] = Future()
        if tool.early or tool.policy is None:
            fut.set_result(ALLOWED if tool.early else NO_POLICY)
            return fut
        try:
            inputs = tool.reduce(args, kwargs)
        except Exception as e:
            fut.set_result(
                Decision(False, 0.0, 1.0, reason=f"reducer error: {type(e).__name__}")
            )
            return fut
        label = self._label(name, args, kwargs)
        try:
            judge = self.judge
        except Exception as e:  # e.g. TYPESAFE_API_KEY is not set: nothing starts early
            warnings.warn(  # shown once per call site: the code still runs, just never early
                f"spec-ptc-jev: no judge, so no tool call will start early ({e})", stacklevel=2
            )
            fut.set_result(Decision(False, 0.0, 1.0, reason=f"no judge: {e}"))
            return fut
        self._emit("judge_start", label=label)
        asked = (self._worker or default_worker()).submit(
            judge,
            tool=name,
            description=tool.description,
            policy=tool.policy,
            inputs=inputs,
        )

        def deliver(done: Future) -> None:  # on the judge worker's thread: stay tiny
            d: Decision = (
                done.result()
            )  # the worker never raises; errors and timeouts are refusals
            fut.set_result(d)
            self._notify.submit(
                self._emit,
                "judge_end",
                label=label,
                allowed=d.allowed,
                p_policy=d.p_policy,
                p_side_effect=d.p_side_effect,
                reason=d.reason,
            )

        asked.add_done_callback(deliver)
        return fut

    def _known_safe(self, tool: _Tool, args: tuple, kwargs: dict) -> bool:
        """True only if a verdict already says this exact call may run early. Never asks."""
        if tool.early:
            return True
        with self._lock:
            v = self._verdicts.get(self._key(tool.name, args, kwargs))
        return v is not None and v.done() and v.result().allowed

    # ------------------------------------------------------------------ running tools
    # The model calls every tool as a plain function. Async tools run on YOUR event loop when
    # you use `arun` / `arun_turn` (so clients and pools created in your app just work), and on
    # a loop this REPL owns when you use the sync `run` / `run_turn`. Early starts of async
    # tools cost no threads.
    def _begin(self, tool: _Tool, args: tuple, kwargs: dict) -> int:
        with self._lock:
            self._call_ids += 1
            cid = self._call_ids
        self._emit(
            "call_start",
            id=cid,
            tool=tool.name,
            label=self._label(tool.name, args, kwargs),
            early=not self._commit.is_set(),
        )
        return cid

    async def _arun_tool(self, tool: _Tool, args: tuple, kwargs: dict) -> Any:
        cid, ok = self._begin(tool, args, kwargs), True
        try:
            return await tool.fn(*args, **kwargs)
        except BaseException:
            ok = False
            raise
        finally:
            self._emit("call_end", id=cid, ok=ok)

    def _start(self, tool: _Tool, args: tuple, kwargs: dict) -> Future:
        """Start a call now, without waiting for it."""
        if tool.is_async:
            assert self._loop is not None
            return asyncio.run_coroutine_threadsafe(
                self._arun_tool(tool, args, kwargs), self._loop
            )
        return self._pool.submit(self._run_tool, tool, args, kwargs)

    def _run_tool(self, tool: _Tool, args: tuple, kwargs: dict) -> Any:
        """Run a call and wait for its result."""
        if tool.is_async:
            return self._start(tool, args, kwargs).result()
        cid, ok = self._begin(tool, args, kwargs), True
        try:
            return tool.fn(*args, **kwargs)
        except BaseException:
            ok = False
            raise
        finally:
            self._emit("call_end", id=cid, ok=ok)

    def _hook(self, tool: _Tool) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            key = self._key(tool.name, args, kwargs)
            with self._launch:
                started = self._ready.get(key)
                early = started.popleft() if started else None
            if early is not None:
                return early.result()  # its result or its exception: the call has run, once
            if (
                not self._commit.is_set()
                and not self.verdict(tool.name, args, kwargs).result().allowed
            ):
                self._commit.wait()
            if self._abort is not None:
                raise TurnAborted(self._abort)
            if self._known_safe(tool, args, kwargs):
                return self._run_tool(tool, args, kwargs)
            with (
                self._launch
            ):  # it may change state: drop what started before it, start nothing during it
                self._drop_early()
                return self._run_tool(tool, args, kwargs)

        call.__name__, call.__doc__ = tool.name, tool.description
        return call

    def _drop_early(self) -> None:
        """Forget every early result not yet used, and cancel the ones that can be cancelled.
        Call with `_launch` held."""
        for futures in self._ready.values():
            for fut in futures:
                fut.cancel()
        self._ready.clear()

    def _print(
        self, *args: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False
    ) -> None:
        if file is not None:
            print(*args, sep=sep, end=end, file=file, flush=flush)
        else:
            self._stdout.append(sep.join(map(str, args)) + end)

    def _final_answer(self, value: Any) -> None:
        self.ns["answer"] = {"content": value, "ready": True}

    # ------------------------------------------------------------------ planning ahead
    def _plan_ahead(self, first: int, only: bool = False) -> None:
        """Look over every statement from index `first` on (closed ones, then the one being
        written), ask about all their known calls at once, and start the allowed ones in program
        order. `only=True` looks at statement `first` alone (the executor, about to run it, so it
        never waits on verdicts it does not need). Safe to call from any thread, any number of
        times: a call is only started if fewer early results exist for it than the code will use.
        Planning is an optimisation: if it fails, it switches itself off and the turn runs normally."""
        if not self._on or (self._after_commit_only and not self._commit.is_set()):
            return
        try:
            st = self._st
            with st.cond:
                ahead = [(i, s) for i, s in enumerate(st.stmts) if i >= first]
                if st.open_stmt is not None and not st.parsed_all:
                    ahead.append((len(st.stmts), st.open_stmt))
            if only:
                ahead = ahead[:1]
            env = _snapshot(self.ns)
            names = {
                n for n in self.tools if env.get(n) is self._hooks.get(n)
            }  # not rebound by the model
            always = frozenset(n for n in names if self.tools[n].early)
            plan: list[tuple[int, Call]] = []
            for i, stmt in ahead:
                calls, go_on = plan_calls(stmt, names, env, always)
                plan += [(i, c) for c in calls]
                if not go_on:
                    break
                learn(stmt, env)  # this code has not run yet
                names -= _binds(
                    stmt
                )  # `def kv(...)` further down: from there on it is not our tool
            verdicts = [self.verdict(*c) for _, c in plan]
            wanted: dict[tuple, int] = {}
            for (i, (name, args, kwargs)), v in zip(plan, verdicts, strict=True):
                if not v.result().allowed:
                    return  # nothing past a call that may not run early
                key = self._key(name, args, kwargs)
                with self._launch:  # one step: "has the executor got there?" and "start it"
                    with st.cond:
                        passed = not only and i < st.next + (1 if st.running else 0)
                    if passed:
                        continue  # the executor is at or past this statement; it planned its own calls
                    wanted[key] = (
                        wanted.get(key, 0) + 1
                    )  # counts only calls still ahead of the executor
                    started = self._ready.setdefault(key, deque())
                    if len(started) < wanted[key]:
                        started.append(self._start(self.tools[name], args, kwargs))
        except Exception as e:
            self._on = False
            self._emit("planning_off", reason=f"{type(e).__name__}: {e}")

    def _planner(self) -> None:
        """Keeps `stmts` / `open_stmt` up to date as text arrives, commits, and plans ahead.
        Whatever happens in here, the turn is always committed so nothing can wait forever."""
        st = self._st
        try:
            self._planner_loop()
        except BaseException as e:
            self._abort = (
                self._abort or f"internal error while reading the code: {type(e).__name__}: {e}"
            )
        finally:
            with st.cond:
                st.parsed_all = True
                st.cond.notify_all()
            if not self._commit.is_set():
                self._emit("commit", ok=self._abort is None)
                self._commit.set()

    def _planner_loop(self) -> None:
        st, lines_done, seen = self._st, 0, None
        while True:
            with st.cond:
                while (st.final, st.code.count("\n"), st.env_version, st.executor_done) == seen:
                    st.cond.wait()
                final, code = st.final, st.code
                seen = (final, code.count("\n"), st.env_version, st.executor_done)
            if self._abort is not None:
                return
            lines = code.split("\n")
            usable = lines if final else lines[:-1]  # the last line may still be growing
            if final:
                try:
                    new, lines_done = closed_statements(usable, lines_done, True)
                except SyntaxError as e:
                    self._abort = _syntax_error(code) or f"SyntaxError: {e.msg}"
                    return
            else:
                new, lines_done = closed_statements(usable, lines_done, False)
            with st.cond:
                st.stmts.extend(new)
                st.open_stmt = None if final else open_statement(usable, lines_done)
                st.parsed_all = final
                st.cond.notify_all()
            if final and not self._commit.is_set():
                self._emit("commit", ok=True)
                self._commit.set()
            with st.cond:
                first = (
                    st.next
                )  # include the running statement: a refused call in it stops the look-ahead
            self._plan_ahead(first)
            if final:
                with st.cond:  # keep planning as the executor learns values, until it is done
                    if st.executor_done or st.next >= len(st.stmts) or not self._on:
                        return

    def _executor(self) -> None:
        st, result = self._st, self._result
        stmt: ast.stmt | None = None
        try:
            while True:
                with st.cond:
                    while st.next >= len(st.stmts) and not st.parsed_all:
                        st.cond.wait()
                    if st.next >= len(st.stmts):
                        break
                    stmt = st.stmts[st.next]
                if not self._on or self._after_commit_only:
                    self._commit.wait()  # the plain path: nothing runs until the block is complete
                if self._abort is not None:
                    break
                self._plan_ahead(st.next, only=True)  # its values may have just become known
                with st.cond:
                    st.running = True
                exec(
                    compile(
                        ast.fix_missing_locations(ast.Module([stmt], [])), "<repl>", "exec"
                    ),
                    self.ns,
                )  # noqa: S102 (the agent's own REPL)
                with st.cond:
                    st.running, st.next, st.env_version = False, st.next + 1, st.env_version + 1
                    st.cond.notify_all()
            if self._abort is not None:
                result["error"] = self._abort
        except TurnAborted as e:
            result["error"] = str(e)
        except BaseException as e:
            where = f" (line {stmt.lineno})" if stmt is not None else ""
            result["error"] = (
                "".join(traceback.format_exception_only(type(e), e)).strip() + where
            )
        finally:
            with st.cond:
                st.executor_done = True
                st.cond.notify_all()

    # ------------------------------------------------------------------ one turn
    def _begin_turn(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError(
                "this SpecRepl is already running a turn; use one SpecRepl per conversation"
            )
        self._loop = loop
        self._t0 = time.perf_counter()
        self._on = self.speculate and any(t.early or t.policy for t in self.tools.values())
        self._st, self._ready, self._abort, self._stdout, self._text = (
            _TurnState(),
            {},
            None,
            [],
            "",
        )
        self._result = {"error": None}
        self._commit = threading.Event()
        self.ns["answer"] = {"content": "", "ready": False}
        self.ns["final_answer"] = self._final_answer
        self.ns["print"] = self._print
        self._hooks = {t.name: self._hook(t) for t in self.tools.values()}
        self.ns.update(self._hooks)
        self.ns.setdefault("__name__", "__repl__")
        self._threads = [
            threading.Thread(target=self._planner, name="spec-repl-plan", daemon=True),
            threading.Thread(target=self._executor, name="spec-repl-exec", daemon=True),
        ]
        for th in self._threads:
            th.start()
        self._emit("stream_begin")

    def _feed(self, delta: str) -> None:
        self._text += delta
        self._emit("token", text=delta)
        with self._st.cond:
            self._st.code = extract_code(self._text)[0]
            self._st.cond.notify_all()

    def _end_turn(self, stream_error: BaseException | None = None) -> Turn:
        """The stream is over (or broke). Finish the turn; always leaves no thread behind."""
        try:
            st = self._st
            self._emit("stream_end")
            if stream_error is not None:  # nothing that was waiting for the full block may run
                self._abort = (
                    f"the model stream failed: {type(stream_error).__name__}: {stream_error}"
                )
            with st.cond:
                st.code, st.final = extract_code(self._text, final=True)[0], True
                st.cond.notify_all()
            for th in self._threads:
                th.join()
            with self._launch:
                self._drop_early()  # nothing started for this turn is wanted any more
            self._notify.submit(
                lambda: None
            ).result()  # every judge_end has reached the listener
            answer = self.ns.get("answer")
            done = (
                isinstance(answer, dict)
                and answer.get("ready")
                and self._result["error"] is None
            )
            self._emit("done", error=self._result["error"])
            return Turn(
                st.code,
                "".join(self._stdout),
                self._result["error"],
                str(answer.get("content")) if done else None,
            )
        finally:
            self._turn_lock.release()

    def run_turn(self, stream: Iterable[str]) -> Turn:
        """Run one turn from a stream of text deltas. Blocking; for scripts and threads. Inside
        an asyncio app use `await repl.arun_turn(stream)`."""
        _refuse_inside_a_loop("run_turn", "arun_turn")
        self._begin_turn(self._private_loop())
        try:
            for delta in stream:
                self._feed(delta)
        except BaseException as e:
            self._end_turn(e)
            raise
        return self._end_turn()

    async def arun_turn(self, stream: AsyncIterable[str] | Iterable[str]) -> Turn:
        """Run one turn without blocking your event loop. Async tools run on THIS loop."""
        loop = asyncio.get_running_loop()
        if not hasattr(stream, "__aiter__"):  # a blocking iterator: drive it from a thread
            return await asyncio.to_thread(self._drive_sync, stream, loop)
        self._begin_turn(loop)
        try:
            async for delta in stream:
                self._feed(delta)
        except BaseException as e:
            await asyncio.to_thread(self._end_turn, e)
            raise
        return await asyncio.to_thread(self._end_turn)

    def _drive_sync(self, stream: Iterable[str], loop: asyncio.AbstractEventLoop) -> Turn:
        self._begin_turn(loop)
        try:
            for delta in stream:
                self._feed(delta)
        except BaseException as e:
            self._end_turn(e)
            raise
        return self._end_turn()

    def _private_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._own_loop is None:
                self._own_loop = asyncio.new_event_loop()
                threading.Thread(
                    target=self._own_loop.run_forever, name="spec-repl-async", daemon=True
                ).start()
            return self._own_loop

    # ------------------------------------------------------------------ a whole task
    def system_prompt(self) -> str:
        """Instructions for the model, written from the registered tools."""
        lines = []
        for t in self.tools.values():
            doc = " ".join(t.description.split()) or "(no description)"
            lines.append(f"  {t.name}{inspect.signature(t.fn)}  # {doc}")
        return (
            "You solve the task by writing Python that runs in a persistent REPL. Reply with exactly "
            "one fenced code block and nothing else, like this:\n```repl\n# your code\n```\n"
            "Variables persist between your replies.\n"
            "Functions you can call:\n" + "\n".join(lines) + "\n"
            "Call them as plain functions (never `await`). Use only these functions and the "
            "standard library. Do as much as you can in each block. Use print() for anything you "
            "want to see; you get the output back and may write another block.\n"
            "When you are done, call final_answer(<the final answer as a string>)."
        )

    def _messages(self, task: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": task},
        ]

    @staticmethod
    def _feedback(turn: Turn) -> str:
        if not turn.code.strip():
            return "No ```repl block found. Reply with one ```repl ... ``` block."
        text = f"Output:\n{turn.stdout[-4000:] or '(nothing printed)'}"
        return text + (f"\nError: {turn.error}" if turn.error else "")

    def run(
        self, chat: Callable[[list[dict]], Iterable[str]], task: str, *, max_turns: int = 6
    ) -> Run:
        """Run a task to completion. `chat(messages)` streams your model's reply to OpenAI-style
        messages. Each turn's output (or error) goes back to the model until it gives a final
        answer or `max_turns` is reached. Blocking; inside an asyncio app use `await repl.arun`."""
        _refuse_inside_a_loop("run", "arun")
        messages, turns = self._messages(task), []
        for _ in range(max_turns):
            reply: list[str] = []

            def tee(stream: Iterable[str], sink: list[str] = reply) -> Iterable[str]:
                for delta in stream:
                    sink.append(delta)
                    yield delta

            turns.append(self.run_turn(tee(chat(messages))))
            if turns[-1].answer is not None:
                break
            messages += [
                {"role": "assistant", "content": "".join(reply)},
                {"role": "user", "content": self._feedback(turns[-1])},
            ]
        return Run(turns[-1].answer if turns else None, turns)

    async def arun(
        self,
        chat: Callable[[list[dict]], AsyncIterable[str] | Iterable[str]],
        task: str,
        *,
        max_turns: int = 6,
    ) -> Run:
        """`run` for asyncio apps. `chat(messages)` may return an async or a plain iterator."""
        messages, turns = self._messages(task), []
        for _ in range(max_turns):
            reply: list[str] = []
            stream = chat(messages)
            if hasattr(stream, "__aiter__"):

                async def atee(s: AsyncIterable[str] = stream, sink: list[str] = reply):
                    async for delta in s:
                        sink.append(delta)
                        yield delta

                turns.append(await self.arun_turn(atee()))
            else:

                def tee(s: Iterable[str] = stream, sink: list[str] = reply) -> Iterable[str]:
                    for delta in s:
                        sink.append(delta)
                        yield delta

                turns.append(await self.arun_turn(tee()))
            if turns[-1].answer is not None:
                break
            messages += [
                {"role": "assistant", "content": "".join(reply)},
                {"role": "user", "content": self._feedback(turns[-1])},
            ]
        return Run(turns[-1].answer if turns else None, turns)


def _refuse_inside_a_loop(name: str, async_name: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{name}() would block your event loop; use `await repl.{async_name}(...)` instead"
    )


def _syntax_error(code: str) -> str | None:
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    return None
