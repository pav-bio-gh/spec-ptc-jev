"""Documents a spec-ptc behaviour our gate has to live with, using NO gate at all.

A stock `speculatable=True` tool inside a `for` loop whose body contains an `if`: the peek
planner fans the calls out while the loop is still streaming, then a later line makes the
re-plan return nothing, spec-ptc retracts ("peek-retracted") the launches it just made, and
the look-ahead re-dispatches each call. The tool runs twice per item.
"""

import threading
import time

from spec_ptc.contracts.events import EventBus
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

CODE = (
    "urls = ['a', 'b', 'c']\n"
    "out = []\n"
    "for u in urls:\n"
    "    page = fetch(u)\n"
    "    if 'x' in page:\n"
    "        out.append(page)\n"
    "    else:\n"
    "        out.append('none')\n"
    "answer['content'] = str(len(out))\n"
    "answer['ready'] = True"
)


class Engine:
    def __init__(self):
        self.mock = MockLM(
            MockTiming(main_tok_per_s=60, sub_base_s=0.1, sub_jitter_s=0.0, sub_tokens=2)
        )
        self.calls = []
        self._lock = threading.Lock()

    def stream_main(self, script):
        return self.mock.stream_main(script)

    def fetch(self, url: str) -> str:
        with self._lock:
            self.calls.append(url)
        time.sleep(0.3)
        return f"page {url}"

    def make_tools(self, reg, bus):
        reg.register("fetch", self.fetch, speculatable=True, pure=True)


def run():
    engine = Engine()
    bus = EventBus()
    harness = Harness(engine, "spec", bus=bus, context="")
    out = harness.run_turn(engine.stream_main(f"go\n```repl\n{CODE}\n```\n"))
    harness.launcher.shutdown()
    return engine, bus, out


def test_stock_spec_ptc_retracts_peeks_and_runs_the_tool_twice():
    engine, bus, out = run()
    assert out.final_answer == "3"
    retracted = [
        e for e in bus.history if e.kind == "evict" and e.data.get("reason") == "peek-retracted"
    ]
    # If upstream fixes this, both numbers drop (0 retractions, 3 calls) and this test should be
    # deleted along with GatedTool's orphan re-attachment.
    assert len(retracted) == 3
    assert sorted(engine.calls) == ["a", "a", "b", "b", "c", "c"]
