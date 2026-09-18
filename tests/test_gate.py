"""Gate mechanics on spec-ptc's real harness, offline.

These tests inject a tiny rule-based `Judge` so they cover only OUR plumbing
(gating, caching, reducers, refusal semantics). They say nothing about Jev's
judgment quality: that is measured live by `evals/run_gate_eval.py`.
"""

import time

import pytest
from spec_ptc.contracts.events import EventBus
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

from spec_ptc_jev import Decision, GatedTool, JevSpeculator

FAST = MockTiming(main_tok_per_s=2000, sub_base_s=0.1, sub_jitter_s=0.0, sub_tokens=2)


class PrefixJudge:
    """Allows a call when its first argument value starts with 'get'."""

    def __init__(self):
        self.seen = []

    def decide(self, *, tool, description, policy, inputs):
        self.seen.append(inputs)
        ok = str(next(iter(inputs.values()))).startswith("get")
        return Decision(ok, 1.0 if ok else 0.0, 0.0 if ok else 1.0)


class BrokenJudge:
    def decide(self, **_):
        raise RuntimeError("judge down")


class Engine:
    def __init__(self, tool):
        self.mock = MockLM(FAST)
        self.tool = tool

    def stream_main(self, script):
        return self.mock.stream_main(script)

    def make_tools(self, reg, bus):
        self.mock.make_tools(reg, bus)
        reg.register_tool(self.tool)


def run_turn(tool, code):
    engine = Engine(tool)
    bus = EventBus()
    harness = Harness(engine, "spec", bus=bus, context="c" * 50)
    out = harness.run_turn(engine.stream_main(f"go\n```repl\n{code}\n```\n"))
    harness.launcher.shutdown()
    exec_begin = next(e.t for e in bus.history if e.kind == "exec_begin")
    return out, bus, exec_begin


def make_kv(judge):
    calls = []

    def kv(op: str, key: str = "") -> str:
        """Key-value store: 'get <key>' reads, 'set <key>' writes."""
        calls.append((time.perf_counter(), op))
        time.sleep(0.1)
        return f"{op}:{key}"

    return GatedTool(kv, speculate_when="reads only", judge=judge), calls


CODE = (
    "a = kv('get', 'k1')\n"
    "b = kv('set', 'k2')\n"
    "c = kv('get', 'k3')\n"
    "answer['content'] = a + '|' + b + '|' + c\nanswer['ready'] = True"
)


def test_allowed_calls_run_early_refused_calls_run_once_on_the_real_path():
    tool, calls = make_kv(PrefixJudge())
    out, bus, exec_begin = run_turn(tool, CODE)
    assert out.final_answer == "get:k1|set:k2|get:k3"
    early = [op for t, op in calls if t < exec_begin]
    assert early == ["get", "get"]  # the get AFTER the refused set still speculated
    assert [op for _, op in calls].count("set") == 1
    assert len(calls) == 3  # nothing ran twice
    hits = [e for e in bus.history if e.kind == "claim_hit" and e.data["tool"] == "kv"]
    assert len(hits) == 2


def test_broken_judge_fails_closed():
    tool, calls = make_kv(BrokenJudge())
    out, _, exec_begin = run_turn(tool, CODE)
    assert out.final_answer == "get:k1|set:k2|get:k3"
    assert [op for t, op in calls if t < exec_begin] == []


def test_decisions_are_cached_per_reduced_input():
    judge = PrefixJudge()
    tool, _ = make_kv(judge)
    for _ in range(3):
        assert tool.speculatable_call(("get", "k1"), {})
    assert len(judge.seen) == 1
    assert judge.seen[0] == {"op": "get", "key": "k1"}  # bound to parameter names


def test_reducer_controls_what_the_judge_sees():
    judge = PrefixJudge()

    def send(op: str, token: str) -> str:
        """Do `op` with a bearer token."""
        return op

    tool = GatedTool(
        send, speculate_when="reads only", judge=judge, reduce=lambda a, k: {"op": a[0]}
    )
    assert tool.speculatable_call(("get", "sk-secret"), {})
    assert "sk-secret" not in repr(judge.seen)


def test_default_reducer_clips_long_values():
    judge = PrefixJudge()
    tool, _ = make_kv(judge)
    tool.speculatable_call(("get", "x" * 5000), {})
    assert len(judge.seen[0]["key"]) < 700


def test_decorator_accepts_speculate_when_and_rejects_mixing():
    spec = JevSpeculator(judge=PrefixJudge())

    @spec.tool(speculate_when="reads only")
    def kv(op: str) -> str:
        """Key-value store."""
        return op

    @spec.tool(speculatable=True, pure=True)
    def llm_query(prompt: str) -> str:
        return prompt

    assert set(spec.registry.names()) == {"kv", "llm_query"}
    assert spec.gated["kv"].speculate_when == "reads only"
    with pytest.raises(ValueError):
        spec.tool(speculate_when="x", speculatable=True, pure=True)
    with pytest.raises(ValueError):
        spec.tool(reduce=lambda a, k: a)
