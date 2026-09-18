"""End-to-end demo: one scripted agent turn with a mixed read/write `bash` tool.

    uv run python -m examples.bash_demo

The root model is scripted (spec-ptc's MockLM streams the code at a fixed
tokens/s); the bash tool really runs, in a throwaway directory, with a fixed
extra latency standing in for a slow command; the Jev judgments are LIVE.

Three arms over the same turn:
  baseline    no speculation
  spec        spec-ptc as-is: bash has side effects, so it must stay unmarked
  spec+gate   bash registered with speculate_when=, judged per call by Jev
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from spec_ptc.contracts.events import EventBus
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

from spec_ptc_jev import GatedTool, JevJudge

TOOL_LATENCY_S = 0.8
POLICY = (
    "Read-only shell commands: listing, reading, searching, counting. Never anything "
    "that writes, deletes, installs, pushes, sends or runs a script."
)
TURN = """I'll survey the project, clean up, and log the result.
```repl
files = bash("ls src")
todos = bash("grep -rn TODO src")
sizes = bash("wc -l src/a.py src/b.py")
readme = bash("cat README.md")
gone = bash("rm -f scratch.tmp")
logged = bash("echo surveyed >> progress.log")
summary = llm_query("Summarize this project survey: " + files + todos + sizes + readme)
answer["content"] = summary
answer["ready"] = True
```
Done."""


def make_workdir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="spec-ptc-jev-demo-"))
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("# TODO: handle retries\nx = 1\n")
    (root / "src" / "b.py").write_text("y = 2\n# TODO: add tests\n")
    (root / "README.md").write_text("# demo project\n")
    (root / "scratch.tmp").write_text("tmp\n")
    return root


class Engine:
    """MockLM's tools plus a bash tool, registered three different ways."""

    def __init__(self, workdir: Path, arm: str, judge: JevJudge | None) -> None:
        self.mock = MockLM(MockTiming(main_tok_per_s=60, sub_base_s=1.0, sub_jitter_s=0.0))
        self.workdir = workdir
        self.arm = arm
        self.judge = judge
        self.runs: list[tuple[float, str]] = []  # (time, command) of every real execution
        self.gated: GatedTool | None = None

    def stream_main(self, script: str):
        return self.mock.stream_main(script)

    def bash(self, command: str) -> str:
        """Run a shell command in the project directory and return its stdout."""
        self.runs.append((time.perf_counter(), command))
        time.sleep(TOOL_LATENCY_S)
        done = subprocess.run(
            command, shell=True, cwd=self.workdir, capture_output=True, text=True
        )
        return done.stdout

    def make_tools(self, reg, bus) -> None:
        self.mock.make_tools(reg, bus)
        if self.arm == "spec+gate":
            assert self.judge is not None
            self.gated = GatedTool(
                self.bash, speculate_when=POLICY, judge=self.judge, name="bash", bus=bus
            )
            reg.register_tool(self.gated)
        else:
            reg.register("bash", self.bash)  # side effects: unmarked, never early


def run_arm(arm: str, judge: JevJudge | None) -> dict:
    workdir = make_workdir()
    try:
        engine = Engine(workdir, arm, judge)
        bus = EventBus()
        harness = Harness(
            engine, "baseline" if arm == "baseline" else "spec", bus=bus, context=""
        )
        t0 = time.perf_counter()
        out = harness.run_turn(engine.stream_main(TURN))
        wall = time.perf_counter() - t0
        harness.launcher.shutdown()
        exec_begin = next(e.t for e in bus.history if e.kind == "exec_begin")
        early = [cmd for t, cmd in engine.runs if t < exec_begin]
        counts: dict[str, int] = {}
        for _, cmd in engine.runs:
            counts[cmd] = counts.get(cmd, 0) + 1
        return {
            "arm": arm,
            "wall_s": wall,
            "answered": out.final_answer is not None,
            "early": early,
            "ran_twice": [c for c, n in counts.items() if n > 1],
            "log_lines": (workdir / "progress.log").read_text().count("surveyed"),
            "scratch_deleted": not (workdir / "scratch.tmp").exists(),
            "decisions": engine.gated.decisions if engine.gated else [],
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    judge = JevJudge()
    results = [run_arm(arm, judge) for arm in ("baseline", "spec", "spec+gate")]
    base = results[0]["wall_s"]
    print(f"{'arm':10} {'wall':>7} {'speedup':>8}  early bash calls")
    for r in results:
        print(f"{r['arm']:10} {r['wall_s']:6.2f}s {base / r['wall_s']:7.2f}x  {r['early']}")
    gate = results[2]
    print("\nJev decisions (p_policy / p_side_effect):")
    for inputs, d in gate["decisions"]:
        verdict = "EARLY " if d.allowed else "wait  "
        print(f"  {verdict} {d.p_policy:.2f} / {d.p_side_effect:.2f}  {inputs['command']}")
    for r in results:
        assert r["answered"], r
        assert r["log_lines"] == 1, f"{r['arm']}: write ran {r['log_lines']} times"
        assert r["scratch_deleted"], r
        assert not r["ran_twice"], f"{r['arm']}: ran twice: {r['ran_twice']}"
        assert not [c for c in r["early"] if c.startswith(("rm", "echo"))], r
    print("\nchecks passed: every arm answered; each write ran exactly once, never early")


if __name__ == "__main__":
    main()
