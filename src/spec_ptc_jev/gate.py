"""Adapter: the same judge as a per-call gate for Alex Zhang's `spec-ptc` library.

`spec_ptc_jev.repl.SpecRepl` is this package's own loop and the recommended way to use it. This
module is for code already built on spec-ptc: `GatedTool` answers spec-ptc's per-call gate
(`Tool.speculatable_call`) with a judge instead of a hand-written rule.

The gate WAITS for its verdict, so spec-ptc only ever launches calls the judge allowed; a
refused call gets spec-ptc's inert `NonSpeculated` marker and runs normally in the real run.
Waiting costs one judge round trip on spec-ptc's look-ahead thread; `attach_prejudge` asks as
soon as a statement closes on the stream, so that wait is usually already over.
"""

from __future__ import annotations

import ast
import inspect
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from spec_ptc import Speculator, Tool
from spec_ptc.contracts.events import NULL_BUS, EventBus

from spec_ptc_jev.judge import Decision, Judge
from spec_ptc_jev.worker import JudgeWorker, default_worker

Reducer = Callable[[tuple, dict], Any]
MAX_FIELD_CHARS = 600
# `gate` events are delivered here, never on the judge worker's thread: subscribers are arbitrary code
_NOTIFIER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spec-ptc-jev-notify")


def flush_events(timeout: float = 5.0) -> None:
    """Wait until every `gate` event queued so far has reached its subscribers."""
    _NOTIFIER.submit(lambda: None).result(timeout)


def _clip(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= MAX_FIELD_CHARS:
        return text
    return text[:MAX_FIELD_CHARS] + f"... [{len(text) - MAX_FIELD_CHARS} more chars]"


def default_reducer(fn: Callable[..., Any]) -> Reducer:
    """Bind the call to the tool's parameter names and clip long values, so the
    judge reads `{"command": "ls"}` rather than a positional tuple."""
    sig = inspect.signature(fn)

    def reduce(args: tuple, kwargs: dict) -> Any:
        try:
            bound = sig.bind(*args, **kwargs).arguments
        except TypeError:
            bound = {"args": list(args), **kwargs}
        return {name: _clip(value) for name, value in bound.items()}

    return reduce


class GatedTool(Tool):
    """A spec-ptc tool whose early execution is decided per call by a judge.

    It registers as `speculatable=True, pure=True` because that is the only state spec-ptc's
    registry accepts for a tool that may ever run early. Purity is asserted per call by the
    judge; every call the judge refuses stays on the normal path."""

    speculatable = True
    pure = True

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        speculate_when: str,
        judge: Judge,
        description: str | None = None,
        reduce: Reducer | None = None,
        name: str | None = None,
        latency_hint_ms: float = 1000.0,
        bus: EventBus = NULL_BUS,
        worker: JudgeWorker | None = None,
    ) -> None:
        if inspect.iscoroutinefunction(fn):
            raise NotImplementedError("GatedTool supports sync tools only")
        if not speculate_when.strip():
            raise ValueError("speculate_when must be a non-empty policy")
        if getattr(self, "batched", False):
            # spec-ptc splits a batched launch into launches of the single-item tool, which
            # would run without this gate's verdict
            raise NotImplementedError("GatedTool cannot be batched")
        self.fn = fn
        self.name = name or fn.__name__
        self.description = (description or inspect.getdoc(fn) or "").strip()
        self.speculate_when = speculate_when.strip()
        self.latency_hint_ms = latency_hint_ms
        self.judge = judge
        self.reduce = reduce or default_reducer(fn)
        self.bus = bus
        self.worker = worker
        self.decisions: list[tuple[Any, Decision]] = []  # audit trail, in verdict order
        self._verdicts: dict[
            str, Future
        ] = {}  # one future per distinct call: cache and in-flight
        self._lock = threading.Lock()

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def request(self, args: tuple, kwargs: dict | None = None) -> Future:
        """A future `Decision` for one call. Never blocks; asked once per distinct call. Keyed
        on the raw call, not the reduced view, so a lossy reducer cannot merge two calls."""
        args, kwargs = tuple(args), dict(kwargs or {})
        key = repr((args, sorted(kwargs.items())))
        with self._lock:
            fut = self._verdicts.get(key)
            if fut is not None:
                return fut
            fut = self._verdicts[key] = Future()
        try:
            inputs = self.reduce(args, kwargs)
        except Exception as e:  # a broken reducer never launches a call early
            fut.set_result(
                Decision(False, 0.0, 1.0, reason=f"reducer error: {type(e).__name__}")
            )
            return fut
        self.bus.emit("gate_begin", tool=self.name, inputs=inputs)
        asked = (self.worker or default_worker()).submit(
            self.judge,
            tool=self.name,
            description=self.description,
            policy=self.speculate_when,
            inputs=inputs,
        )

        def deliver(done: Future) -> None:  # on the judge worker's thread: stay tiny
            decision: Decision = done.result()  # never raises: errors and timeouts are refusals
            with self._lock:
                self.decisions.append((inputs, decision))
            fut.set_result(decision)
            _NOTIFIER.submit(
                self.bus.emit,
                "gate",
                tool=self.name,
                allowed=decision.allowed,
                p_policy=decision.p_policy,
                p_side_effect=decision.p_side_effect,
                reason=decision.reason,
                inputs=inputs,
            )

        asked.add_done_callback(deliver)
        return fut

    def decide(self, args: tuple, kwargs: dict | None = None) -> Decision:
        return self.request(args, kwargs).result()

    def speculatable_call(self, args: tuple, kwargs: dict) -> bool:
        """spec-ptc's per-call gate. Waits for the verdict: only allowed calls are launched."""
        return self.decide(tuple(args), dict(kwargs)).allowed


def attach_prejudge(bus: EventBus, tools: dict[str, GatedTool]) -> None:
    """Ask the judge the moment a statement closes on the token stream (spec-ptc's `Harness`
    emits `stmt_closed`), for calls whose arguments are literals. Nothing is executed."""

    def on(ev: Any) -> None:
        if ev.kind != "stmt_closed":
            return
        try:
            tree = ast.parse(ev.data.get("src", ""))
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in tools
            ):
                try:
                    args = tuple(ast.literal_eval(a) for a in node.args)
                    kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
                except (ValueError, SyntaxError):
                    continue
                tools[node.func.id].request(args, kwargs)

    bus.subscribe(on)


class JevSpeculator(Speculator):
    """`Speculator` whose `tool()` decorator also accepts `speculate_when=`, `reduce=` and
    `description=`. Tools registered without `speculate_when` behave exactly as in spec-ptc."""

    def __init__(self, *, judge: Judge | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._judge = judge
        self.gated: dict[str, GatedTool] = {}

    @property
    def judge(self) -> Judge:
        if self._judge is None:
            from spec_ptc_jev.judge import JevJudge

            self._judge = JevJudge()
        return self._judge

    def tool(  # type: ignore[override]
        self,
        *,
        speculate_when: str | None = None,
        reduce: Reducer | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> Callable:
        if speculate_when is None:
            if reduce is not None:
                raise ValueError("reduce= only applies together with speculate_when=")
            return super().tool(**kwargs)
        if kwargs.get("speculatable") or kwargs.get("pure"):
            raise ValueError(
                "pass either speculatable=True/pure=True (always early) or "
                "speculate_when= (judged per call), not both"
            )

        def deco(fn: Callable) -> Callable:
            gated = GatedTool(
                fn,
                speculate_when=speculate_when,
                judge=self.judge,
                description=description,
                reduce=reduce,
                name=kwargs.get("name"),
                latency_hint_ms=kwargs.get("latency_hint_ms", 1000.0),
                bus=self.bus,
            )
            self.add(gated)
            self.gated[gated.name] = gated
            return fn

        return deco
