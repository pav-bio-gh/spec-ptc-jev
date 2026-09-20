"""Out of the box, on a different job: no browser, no hand-written prompt, nothing task-specific.

    uv run --env-file <env with OPENAI_API_KEY + TYPESAFE_API_KEY> python -m examples.pypi_check [rounds]

Three plain tools are registered and `repl.run(chat, task)` does the rest: it writes the model's
instructions from the tools, streams the model, starts the calls Jev allows while the code is
still being written, feeds output back, and stops when the model sets its answer.

  http(method, url)   judged per call: a GET may start early, anything else waits
  llm_query(prompt)   always safe to start early
  save_note(text)     no early policy: runs only once the code is complete

The same task runs with `speculate=False` (generate, then run) for comparison, alternating.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from openai import OpenAI

from spec_ptc_jev import JevJudge, SpecRepl

MODEL = "gpt-4.1-mini"
PACKAGES = ["requests", "numpy", "pandas", "flask", "django", "pydantic"]
TASK = (
    "For each of these PyPI packages: " + ", ".join(PACKAGES) + ". Fetch "
    "https://pypi.org/pypi/<name>/json with http('GET', url) and read info.version and "
    "info.requires_python from the JSON. Then ask llm_query, in one call, which of them require the "
    "newest minimum Python version, giving it the six (name, version, requires_python) triples, and "
    "to reply with only those package names, comma separated. "
    "Save a one-paragraph note with save_note. The final answer is llm_query's reply."
)


OUT = Path(__file__).parent / "results" / "pypi_check.json"


def ground_truth() -> set[str]:
    """Fetched directly, outside any timing: which packages need the newest minimum Python."""
    floor: dict[str, tuple[int, ...]] = {}
    for name in PACKAGES:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=20) as r:  # noqa: S310
            spec = json.load(r)["info"]["requires_python"] or ""
        m = re.search(r">=\s*(\d+(?:\.\d+)*)", spec)
        floor[name] = tuple(int(x) for x in m.group(1).split(".")) if m else (0,)
    top = max(floor.values())
    return {name for name, v in floor.items() if v == top}


def correct(answer: str | None, truth: set[str]) -> bool:
    """The task asks for only the winning names, so the names in the answer must equal the truth."""
    if not answer:
        return False
    return {name for name in PACKAGES if name in answer.lower()} == truth


def build(speculate: bool, judge: JevJudge, notes: Path, log: list) -> SpecRepl:
    client = OpenAI()
    repl = SpecRepl(
        judge=judge, speculate=speculate, on_event=lambda kind, **d: log.append((kind, d))
    )

    @repl.tool(early_when="GET requests that only read data. Never POST, PUT, PATCH or DELETE.")
    def http(method: str, url: str) -> str:
        """Make an HTTP request and return the response body as text."""
        req = urllib.request.Request(
            url, method=method.upper(), headers={"User-Agent": "spec-ptc-jev example"}
        )
        with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 (https URLs from the task)
            return r.read().decode("utf-8", "replace")

    @repl.tool(early=True)
    def llm_query(prompt: str) -> str:
        """Ask a language model one question and return its text answer."""
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": str(prompt)}],
            max_tokens=300,
            temperature=0,
        )
        return r.choices[0].message.content or ""

    @repl.tool()
    def save_note(text: str) -> str:
        """Append a note to the project's notes file. Changes a file on disk."""
        with notes.open("a") as fh:
            fh.write(text.strip() + "\n")
        return "saved"

    def chat(messages):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, stream=True, temperature=0.2, max_tokens=900
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    repl.chat = chat  # type: ignore[attr-defined]
    return repl


def one(speculate: bool, judge: JevJudge, truth: set[str]) -> dict:
    notes = Path(tempfile.mkdtemp(prefix="pypi-check-")) / "notes.txt"
    log: list = []
    repl = build(speculate, judge, notes, log)
    t0 = time.perf_counter()
    result = repl.run(repl.chat, TASK)  # type: ignore[attr-defined]
    turns = result.turns
    wall = time.perf_counter() - t0
    starts = [d for k, d in log if k == "call_start"]
    verdicts = [d for k, d in log if k == "judge_end"]
    return {
        "speculate": speculate,
        "wall": round(wall, 2),
        "turns": len(turns),
        "answer": turns[-1].answer,
        "correct": correct(turns[-1].answer, truth),
        "error": turns[-1].error,
        "gets": sum(1 for d in starts if d["tool"] == "http"),
        "early": sum(1 for d in starts if d["early"]),
        "notes_saved": notes.read_text().count("\n") if notes.exists() else 0,
        "note_early": any(d["early"] for d in starts if d["tool"] == "save_note"),
        "allowed": sum(d["allowed"] for d in verdicts),
        "judged": len(verdicts),
    }


def main() -> None:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    judge, results, truth = JevJudge(), [], ground_truth()
    print("ground truth:", sorted(truth))
    for i in range(rounds):
        for speculate in random.sample((False, True), 2):  # neither arm always goes second
            r = one(speculate, judge, truth)
            results.append(r)
            print(
                f"round {i + 1} {'ours  ' if speculate else 'serial'} wall {r['wall']:5.2f}s  turns={r['turns']}  GETs={r['gets']}  "
                f"started early={r['early']}  jev allowed {r['allowed']}/{r['judged']}  notes saved={r['notes_saved']}"
                f"{' NOTE RAN EARLY' if r['note_early'] else ''}  {'correct' if r['correct'] else 'WRONG or failed: ' + str(r['error'] or r['answer'])[:120]}",
                flush=True,
            )
    for speculate in (False, True):
        walls = [r["wall"] for r in results if r["speculate"] == speculate and r["correct"]]
        if walls:
            print(
                f"{'ours  ' if speculate else 'serial'} median {statistics.median(walls):5.2f}s  ({len(walls)} runs)"
            )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "recorded": time.strftime("%Y-%m-%d"),
                "model": MODEL,
                "truth": sorted(truth),
                "runs": results,
            },
            indent=1,
        )
    )
    print(f"correct {sum(r['correct'] for r in results)}/{len(results)}; wrote {OUT}")
    print("last answer:", (results[-1]["answer"] or "")[:300])


if __name__ == "__main__":
    main()
