"""Race: one agent turn ("why is CI failing, fix it") run three ways, recorded as timelines.

    uv run python -m examples.race            # prints the result, writes results/race.json

The root model is scripted (spec-ptc's MockLM streams the code at a fixed
tokens/s). Everything else is real: bash runs in a throwaway git repo (plus a
fixed extra latency per command standing in for slow tools), and every Jev
judgment is a live API call. `results/race.html` replays the recording in a browser.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from spec_ptc.contracts.events import EventBus
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

from spec_ptc_jev import GatedTool, JevJudge

TOOL_LATENCY_S = 0.9
SUB_LLM_LATENCY_S = 1.6
TOKENS_PER_S = 45
POLICY = (
    "Read-only shell commands: listing, reading, searching, counting. Never anything "
    "that writes, deletes, installs, pushes, sends or runs a script."
)
TURN = """CI is red on main. I'll look at the failure, the code and its test, then fix it.
```repl
status = bash("git status --short")
history = bash("git log --oneline -5")
failure = bash("grep -n FAILED logs/ci.log")
source = bash("cat src/retry.py")
test = bash("cat tests/test_retry.py")
todos = bash("grep -rn TODO src")
sizes = bash("wc -l src/retry.py tests/test_retry.py")
diagnosis = llm_query("Why is CI failing? " + failure + source + test + history)
patched = bash("sed -i.bak 's/retries = 0/retries = 3/' src/retry.py")
noted = bash("echo 'fix: restore retry budget' >> CHANGELOG.md")
answer["content"] = diagnosis
answer["ready"] = True
```
Patched the retry budget and noted it in the changelog."""
OUT = Path(__file__).parent / "results" / "race.json"
TEMPLATE = Path(__file__).parent / "race_template.html"
GIT = ["git", "-c", "user.name=demo", "-c", "user.email=demo@example.com"]


def make_repo() -> Path:
    root = Path(tempfile.mkdtemp(prefix="spec-ptc-jev-race-"))
    for d in ("src", "tests", "logs"):
        (root / d).mkdir()
    (root / "src" / "retry.py").write_text(
        "# TODO: make the budget configurable\nretries = 0\n\n"
        "def fetch(get):\n    for _ in range(retries):\n        return get()\n"
    )
    (root / "tests" / "test_retry.py").write_text(
        "from src.retry import fetch\n\ndef test_fetch():\n    assert fetch(lambda: 1) == 1\n"
    )
    (root / "logs" / "ci.log").write_text("collected 1 item\nFAILED tests/test_retry.py\n")
    (root / "CHANGELOG.md").write_text("# Changelog\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run([*GIT, "add", "-A"], cwd=root, check=True)
    subprocess.run([*GIT, "commit", "-qm", "retry: drop budget"], cwd=root, check=True)
    return root


class Engine:
    def __init__(self, workdir: Path, arm: str, judge: JevJudge) -> None:
        self.mock = MockLM(MockTiming(main_tok_per_s=TOKENS_PER_S))
        self.workdir, self.arm, self.judge = workdir, arm, judge
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def stream_main(self, script: str):
        return self.mock.stream_main(script)

    def _record(self, tool: str, label: str, t0: float) -> None:
        with self._lock:
            self.calls.append(
                {"tool": tool, "label": label, "t0": t0, "t1": time.perf_counter()}
            )

    def bash(self, command: str) -> str:
        """Run a shell command in the project directory and return its stdout."""
        t0 = time.perf_counter()
        time.sleep(TOOL_LATENCY_S)
        done = subprocess.run(
            command, shell=True, cwd=self.workdir, capture_output=True, text=True
        )
        self._record("bash", command, t0)
        return done.stdout

    def llm_query(self, prompt: str) -> str:
        """Ask a language model one question and return its text answer."""
        t0 = time.perf_counter()
        time.sleep(SUB_LLM_LATENCY_S)
        self._record("llm_query", prompt[:40], t0)
        return "fetch() never runs its body: the retry budget is 0."

    def make_tools(self, reg, bus) -> None:
        reg.register("llm_query", self.llm_query, speculatable=True, pure=True)
        if self.arm == "gate":
            reg.register_tool(
                GatedTool(
                    self.bash, speculate_when=POLICY, judge=self.judge, name="bash", bus=bus
                )
            )
        else:
            reg.register("bash", self.bash)  # side effects: unmarked, never early


def run_arm(arm: str, judge: JevJudge) -> dict:
    workdir = make_repo()
    try:
        engine = Engine(workdir, arm, judge)
        bus = EventBus()
        harness = Harness(
            engine, "baseline" if arm == "serial" else "spec", bus=bus, context=""
        )
        t_start = time.perf_counter()
        out = harness.run_turn(engine.stream_main(TURN))
        wall = time.perf_counter() - t_start
        harness.launcher.shutdown()

        def rel(t: float) -> float:
            return round(t - t_start, 3)

        first = {
            k: next(e.t for e in bus.history if e.kind == k)
            for k in ("stream_begin", "stream_end", "exec_begin")
        }
        judged, pending = [], {}
        for e in bus.history:
            if e.kind == "gate_begin":
                pending[e.data["inputs"]["command"]] = e.t
            elif e.kind == "gate":
                cmd = e.data["inputs"]["command"]
                judged.append(
                    {
                        "label": cmd,
                        "t0": rel(pending.pop(cmd)),
                        "t1": rel(e.t),
                        "allowed": e.data["allowed"],
                        "p_policy": e.data["p_policy"],
                        "p_side_effect": e.data["p_side_effect"],
                    }
                )
        calls = sorted(engine.calls, key=lambda c: c["t0"])
        changelog = (workdir / "CHANGELOG.md").read_text()
        assert out.final_answer, arm
        assert changelog.count("fix: restore retry budget") == 1, (arm, changelog)
        assert "retries = 3" in (workdir / "src" / "retry.py").read_text(), arm
        assert len(calls) == 10, (arm, len(calls))  # 9 bash + 1 llm_query, none twice
        return {
            "arm": arm,
            "wall": round(wall, 3),
            "stream": [rel(first["stream_begin"]), rel(first["stream_end"])],
            "exec_begin": rel(first["exec_begin"]),
            "tokens": [[rel(e.t), e.data["text"]] for e in bus.history if e.kind == "token"],
            "judged": judged,
            "calls": [
                {
                    "tool": c["tool"],
                    "label": c["label"],
                    "t0": rel(c["t0"]),
                    "t1": rel(c["t1"]),
                    "early": c["t0"] < first["exec_begin"],
                }
                for c in calls
            ],
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    judge = JevJudge()
    arms = [run_arm(arm, judge) for arm in ("serial", "spec", "gate")]
    serial = arms[0]["wall"]
    for a in arms:
        early = [c["label"] for c in a["calls"] if c["early"]]
        writes_early = [c for c in early if c.startswith(("sed", "echo"))]
        assert not writes_early, writes_early
        print(f"{a['arm']:7} {a['wall']:6.2f}s  {serial / a['wall']:.2f}x  early={len(early)}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "recorded": time.strftime("%Y-%m-%d"),
                "policy": POLICY,
                "tokens_per_s": TOKENS_PER_S,
                "tool_latency_s": TOOL_LATENCY_S,
                "sub_llm_latency_s": SUB_LLM_LATENCY_S,
                "arms": arms,
            }
        )
    )
    page = OUT.with_suffix(".html")
    page.write_text(TEMPLATE.read_text().replace("/*DATA*/", OUT.read_text()))
    print(f"checks passed: each write ran once, never early. wrote {OUT} and {page}")


if __name__ == "__main__":
    main()
