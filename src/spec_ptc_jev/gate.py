"""`speculate_when`: a natural-language policy in place of the `speculatable` boolean.

spec-ptc already evaluates a per-call gate (`Tool.speculatable_call`) before it
launches a call early. `GatedTool` answers that gate with a judge instead of a
hand-written rule. A refused call returns spec-ptc's inert `NonSpeculated`
marker in the shadow, so the rest of the turn keeps speculating, and the real
run executes the tool normally: refusing only ever costs speed.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from typing import Any

from spec_ptc import Speculator, Tool
from spec_ptc.contracts.events import NULL_BUS, EventBus

from spec_ptc_jev.judge import Decision, Judge

Reducer = Callable[[tuple, dict], Any]
MAX_FIELD_CHARS = 600


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
    """A tool whose early execution is decided per call by a judge.

    It registers with spec-ptc as `speculatable=True, pure=True` because that is
    the only state spec-ptc's registry accepts for a tool that may ever run
    early. The tool is NOT unconditionally pure: purity is asserted per call by
    the judge, and every call the judge refuses stays on the normal path.
    """

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
    ) -> None:
        if inspect.iscoroutinefunction(fn):
            raise NotImplementedError("GatedTool supports sync tools only for now")
        if not speculate_when.strip():
            raise ValueError("speculate_when must be a non-empty policy")
        self.fn = fn
        self.name = name or fn.__name__
        self.description = (description or inspect.getdoc(fn) or "").strip()
        self.speculate_when = speculate_when.strip()
        self.latency_hint_ms = latency_hint_ms
        self.judge = judge
        self.reduce = reduce or default_reducer(fn)
        self.bus = bus
        self.decisions: list[tuple[Any, Decision]] = []  # audit trail, in order
        self._cache: dict[str, Decision] = {}
        self._lock = threading.Lock()

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def decide(self, args: tuple, kwargs: dict) -> Decision:
        inputs = self.reduce(args, kwargs)
        key = repr(inputs)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.bus.emit("gate_begin", tool=self.name, inputs=inputs)
        decision = self.judge.decide(
            tool=self.name,
            description=self.description,
            policy=self.speculate_when,
            inputs=inputs,
        )
        with self._lock:
            self._cache[key] = decision
            self.decisions.append((inputs, decision))
        self.bus.emit(
            "gate",
            tool=self.name,
            allowed=decision.allowed,
            p_policy=decision.p_policy,
            p_side_effect=decision.p_side_effect,
            inputs=inputs,
        )
        return decision

    def speculatable_call(self, args: tuple, kwargs: dict) -> bool:
        try:
            return self.decide(tuple(args), dict(kwargs)).allowed
        except Exception:
            return False  # a broken reducer or judge never launches a call early


class JevSpeculator(Speculator):
    """`Speculator` whose `tool()` decorator also accepts `speculate_when=`,
    `reduce=` and `description=`. Tools registered without `speculate_when`
    behave exactly as in spec-ptc."""

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
