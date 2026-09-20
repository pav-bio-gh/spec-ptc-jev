"""Browser race: one code-writing agent turn, run three ways, with nothing faked.

    uv run playwright install chromium chromium-headless-shell          # once
    uv run --env-file <env with OPENAI_API_KEY + TYPESAFE_API_KEY> \
        python -m examples.browse_race [rounds]
    BROWSE_RACE_HEADED=1 ...   # watch the Chromium tabs open

The model streams ONE Python block that opens five Wikipedia pages, asks a sub-model to order
the people by birth year, then submits a Wikipedia search (the one call that changes state).

Arms, in a fresh random order every round:
  serial    generate everything, then run it, one call after another.   (SpecRepl, speculate=False)
  spec      Alex Zhang's spec-ptc as shipped. `browse` can submit forms, so it must stay
            unmarked; only `llm_query` may run early.                    (spec-ptc Harness)
  parallel  our loop with early start switched OFF: allowed calls run side by side, but only
            once the code is complete. Separates "parallel" from "early". (SpecRepl)
  gate      our loop: Jev judges each `browse` call as it is written.    (SpecRepl)

Every arm gets the same prompt, generated from the tools by `SpecRepl.system_prompt()`. It says
nothing about how to write the code: no "use literal URLs", no "write straight-line code".

Real: the model stream (OpenAI), headless Chromium via Playwright waiting for network idle, the
sub-model call, every Jev verdict. No injected latency anywhere. `examples.live` shows the same
three arms racing live in a browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import statistics
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI
from spec_ptc.contracts.events import EventBus
from spec_ptc.runtime.harness import Harness

from spec_ptc_jev import JevJudge, SpecRepl

ROOT_MODEL = os.environ.get("BROWSE_RACE_ROOT_MODEL", "gpt-4.1-mini")
SUB_MODEL = os.environ.get(
    "BROWSE_RACE_SUB_MODEL", "gpt-4.1"
)  # the mini model knew the years but mis-sorted them
HEADED = os.environ.get("BROWSE_RACE_HEADED", "") == "1"
ARMS = ("serial", "spec", "gate")  # the live page
BENCH_ARMS = ("serial", "spec", "parallel", "gate")
POLICY = "Opening a URL and reading the page text. Never searching, typing, clicking, submitting a form, or logging in."
URLS = [
    "https://en.wikipedia.org/wiki/Alan_Turing",
    "https://en.wikipedia.org/wiki/Grace_Hopper",
    "https://en.wikipedia.org/wiki/Ada_Lovelace",
    "https://en.wikipedia.org/wiki/John_von_Neumann",
    "https://en.wikipedia.org/wiki/Claude_Shannon",
]
TASK = (
    "Here are five Wikipedia pages:\n"
    + "\n".join(URLS)
    + "\nOpen each page and keep the first 700 characters of its text. Then ask llm_query, in one "
    "call, to first list each person's birth year, one per line, and then to write on the LAST "
    "line ONE sentence listing the five people from oldest to youngest with their birth years. "
    "The last non-empty line of its reply is the sentence. Then search Wikipedia for 'Turing "
    "Award' with browse('search Turing Award'). Print the sentence and the URL the search "
    "returned. The final answer is the sentence."
)
# born 1815, 1903, 1906, 1912, 1916
EXPECTED_ORDER = ("Lovelace", "Neumann", "Hopper", "Turing", "Shannon")
BROWSE_DOC = (
    "Drive a real browser. 'open <url>' loads the page in a new tab and returns the article "
    "text, first 4000 characters (read-only). 'search <text>' opens Wikipedia, types into its "
    "search box, submits the form and returns the URL it lands on (submits a form)."
)
LLM_DOC = "Ask a language model one question and return its text answer."
OUT = Path(__file__).parent / "results" / "browse_race.json"


class BrowserWorker:
    """One Chromium on its own asyncio thread; `browse` calls come from any thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self._run(self._start())

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=120)

    async def _start(self) -> None:
        from playwright.async_api import async_playwright

        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=not HEADED)
        self.context = None

    async def _fresh_context(self) -> None:
        if self.context is not None:
            await self.context.close()
        self.context = await self.browser.new_context()

    async def _open(self, url: str) -> str:
        page = await self.context.new_page()
        await page.goto(url, wait_until="networkidle", timeout=30000)
        article = page.locator("#mw-content-text")  # the article itself on MediaWiki sites
        text = await (
            article if await article.count() else page.locator("body")
        ).first.inner_text()
        return text[:4000]

    async def _search(self, text: str) -> str:
        page = (
            await self.context.new_page()
        )  # its own tab: independent of what was opened before
        await page.goto("https://en.wikipedia.org/wiki/Main_Page", wait_until="networkidle")
        await page.fill("#searchInput", text)
        await page.press("#searchInput", "Enter")
        await page.wait_for_load_state("networkidle")
        return page.url

    def fresh_context(self) -> None:
        self._run(self._fresh_context())

    def browse(self, command: str) -> str:
        verb, _, rest = command.strip().partition(" ")
        if verb == "open":
            return self._run(self._open(rest.strip()))
        if verb == "search":
            return self._run(self._search(rest.strip()))
        return f"unknown command: {command!r}"

    def close(self) -> None:
        async def _close():
            await self.browser.close()
            await self.pw.stop()

        self._run(_close())
        self.loop.call_soon_threadsafe(self.loop.stop)


# --------------------------------------------------------------------------- the model and the sub-model
_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def system_prompt() -> str:
    """The same instructions for every arm: `SpecRepl.system_prompt()` over the two tools."""
    repl = SpecRepl()

    @repl.tool(description=BROWSE_DOC)
    def browse(command: str) -> str: ...

    @repl.tool(description=LLM_DOC)
    def llm_query(prompt: str) -> str: ...

    return repl.system_prompt()


def model_stream():
    resp = client().chat.completions.create(
        model=ROOT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": TASK},
        ],
        stream=True,
        temperature=0.2,
        max_tokens=1200,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def sub_model(prompt: str) -> str:
    r = client().chat.completions.create(
        model=SUB_MODEL,
        messages=[{"role": "user", "content": str(prompt)}],
        max_tokens=400,
        temperature=0,
    )
    return r.choices[0].message.content or ""


def _display(label: str) -> str:
    """`browse(open https://…)` -> `open https://…`; any llm_query -> one fixed row name."""
    m = re.match(r"browse\((.*)\)$", label, re.S)
    return m.group(1) if m else "llm_query(order by birth year)"


# --------------------------------------------------------------------------- arms
Emit = Callable[..., None]


def run_ours(
    arm: str, worker: BrowserWorker, judge: JevJudge, emit: Emit, go: Callable[[], None]
) -> tuple[str | None, list[str]]:
    def on_event(kind: str, t: float, **d) -> None:
        if kind in ("judge_start", "judge_end", "call_start"):
            d["label"] = _display(d["label"])
        emit(kind, t=t, **d)

    repl = SpecRepl(judge=judge, speculate=(arm != "serial"), on_event=on_event)
    repl._after_commit_only = (
        arm == "parallel"
    )  # side by side, but nothing before the code is complete

    @repl.tool(early_when=POLICY, description=BROWSE_DOC)
    def browse(command: str) -> str:
        return worker.browse(command)

    @repl.tool(early=True, description=LLM_DOC)
    def llm_query(prompt: str) -> str:
        return sub_model(prompt)

    go()
    turn = repl.run_turn(model_stream())
    return turn.answer, [turn.error] if turn.error else []


def run_stock(
    worker: BrowserWorker, emit: Emit, go: Callable[[], None]
) -> tuple[str | None, list[str]]:
    """spec-ptc exactly as shipped: `browse` unmarked, `llm_query` speculatable."""
    t0 = [0.0]
    committed = threading.Event()
    ids = iter(range(1, 10**6))
    lock = threading.Lock()

    def now() -> float:
        return time.perf_counter() - t0[0]

    def traced(name: str, fn):
        def call(arg: str):
            with lock:
                cid = next(ids)
            label = arg if name == "browse" else "llm_query(order by birth year)"
            emit(
                "call_start",
                t=now(),
                id=cid,
                tool=name,
                label=label,
                early=not committed.is_set(),
            )
            try:
                return fn(arg)
            finally:
                emit("call_end", t=now(), id=cid, ok=True)

        call.__name__ = name
        return call

    class Engine:
        def make_tools(self, reg, bus) -> None:
            reg.register(
                "llm_query",
                traced("llm_query", sub_model),
                speculatable=True,
                pure=True,
                latency_hint_ms=1500,
            )
            reg.register(
                "browse", traced("browse", worker.browse)
            )  # can submit forms: never early

    bus = EventBus()
    harness = Harness(Engine(), "spec", bus=bus, context="")

    def on(ev) -> None:
        if ev.kind == "token":
            emit("token", t=now(), text=ev.data["text"])
        elif ev.kind in ("stream_begin", "stream_end"):
            emit(ev.kind, t=now())
        elif ev.kind == "exec_begin":
            committed.set()
            emit("commit", t=now(), ok=True)

    bus.subscribe(on)

    def final_answer(value) -> None:
        harness.repl.locals["answer"]["content"] = value
        harness.repl.locals["answer"]["ready"] = True

    harness.repl.locals["final_answer"] = final_answer
    go()
    t0[0] = time.perf_counter()
    out = harness.run_turn(model_stream())
    harness.launcher.shutdown()
    errors = [r.stderr.strip()[-300:] for r in out.results if getattr(r, "stderr", "").strip()]
    return out.final_answer, errors


def run_arm(
    arm: str,
    worker: BrowserWorker,
    judge: JevJudge,
    listener: Emit | None = None,
    go: Callable[[], None] = lambda: None,
) -> dict:
    """Run one arm; return its record. `listener(kind, t=, **data)` sees every event as it
    happens (the live page); `go` is called right before the clock starts (a start barrier)."""
    worker.fresh_context()
    events: list[dict] = []
    lock = threading.Lock()

    def emit(kind: str, t: float, **d) -> None:
        with lock:
            events.append({"kind": kind, "t": round(t, 4), **d})
        if listener is not None:
            listener(kind, t=round(t, 4), **d)

    started = [time.perf_counter()]

    def clocked_go() -> None:
        go()
        started[0] = time.perf_counter()

    try:
        answer, errors = (
            run_stock(worker, emit, clocked_go)
            if arm == "spec"
            else run_ours(arm, worker, judge, emit, clocked_go)
        )
    except Exception as e:
        answer, errors = None, [f"{type(e).__name__}: {e}"]
    wall = time.perf_counter() - started[0]

    def first(kind: str) -> float | None:
        return next((e["t"] for e in events if e["kind"] == kind), None)

    ends = {e["id"]: e["t"] for e in events if e["kind"] == "call_end"}
    calls = [
        {
            "tool": e["tool"],
            "label": e["label"],
            "t0": e["t"],
            "t1": ends.get(e["id"], round(wall, 4)),
            "early": e["early"],
        }
        for e in events
        if e["kind"] == "call_start"
    ]
    asked: dict[str, float] = {}
    judged = []
    for e in events:
        if e["kind"] == "judge_start":
            asked[e["label"]] = e["t"]
        elif e["kind"] == "judge_end":
            judged.append(
                {
                    "label": e["label"],
                    "t0": asked.get(e["label"], e["t"]),
                    "t1": e["t"],
                    "allowed": e["allowed"],
                    "p_policy": e["p_policy"],
                    "p_side_effect": e["p_side_effect"],
                    "reason": e.get("reason", ""),
                }
            )
    searches = [c for c in calls if c["label"].startswith("search")]
    opens = [c for c in calls if c["label"].startswith("open")]
    return {
        "arm": arm,
        "wall": round(wall, 3),
        "stream_end": first("stream_end"),
        "commit": first("commit"),
        "code": "".join(e["text"] for e in events if e["kind"] == "token"),
        "final_answer": answer,
        "answer_correct": answer_correct(answer),
        "errors": errors,
        "judged": judged,
        "calls": calls,
        "n_early": sum(c["early"] for c in calls),
        "n_open_calls": len(opens),
        "n_search_calls": len(searches),
        "search_early": any(c["early"] for c in searches),
    }


def answer_correct(answer: str | None) -> bool:
    """The five people appear in the answer in true birth order."""
    if not answer:
        return False
    at = [answer.find(name) for name in EXPECTED_ORDER]
    return all(i >= 0 for i in at) and at == sorted(at)


def summarize(results: list[dict]) -> dict:
    out: dict = {"arms": {}}
    for arm in BENCH_ARMS:
        runs = [r for r in results if r["arm"] == arm]
        # only runs that did the job count; the rest are reported as failures
        walls = [r["wall"] for r in runs if r["answer_correct"] and not r["errors"]]
        if walls:
            out["arms"][arm] = {
                "attempts": len(runs),
                "n": len(walls),
                "median": round(statistics.median(walls), 2),
                "min": round(min(walls), 2),
                "max": round(max(walls), 2),
            }
    g = [r for r in results if r["arm"] == "gate"]
    out["opens_allowed"] = sum(
        j["allowed"] for r in g for j in r["judged"] if j["label"].startswith("open")
    )
    out["opens_judged"] = sum(
        1 for r in g for j in r["judged"] if j["label"].startswith("open")
    )
    out["searches_refused"] = sum(
        not j["allowed"] for r in g for j in r["judged"] if j["label"].startswith("search")
    )
    out["searches_judged"] = sum(
        1 for r in g for j in r["judged"] if j["label"].startswith("search")
    )
    # safety checks hold for EVERY run, including ones whose answer was wrong or that ran nothing
    out["search_never_early"] = not any(r["search_early"] for r in results)
    labels = [[c["label"] for c in r["calls"]] for r in results]
    out["no_call_ran_twice"] = all(len(ls) == len(set(ls)) for ls in labels)
    out["answers_correct"] = sum(r["answer_correct"] for r in results)
    out["runs"] = len(results)
    out["errors"] = sum(len(r["errors"]) for r in results)
    return out


def main() -> None:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    worker, judge, results = BrowserWorker(), JevJudge(), []
    try:
        for i in range(rounds):
            for arm in random.sample(BENCH_ARMS, len(BENCH_ARMS)):  # no arm always goes last
                r = run_arm(arm, worker, judge)
                results.append(r)
                print(
                    f"round {i + 1} {arm:6} wall {r['wall']:6.2f}s  code done {r['commit'] or 0:5.2f}s  early={r['n_early']}  "
                    f"opens={r['n_open_calls']}  searches={r['n_search_calls']}{' EARLY' if r['search_early'] else ''}  answer={'ok' if r['answer_correct'] else 'WRONG'}  errors={len(r['errors'])}",
                    flush=True,
                )
                OUT.parent.mkdir(exist_ok=True)
                OUT.write_text(
                    json.dumps(
                        {
                            "recorded": time.strftime("%Y-%m-%d"),
                            "root_model": ROOT_MODEL,
                            "sub_model": SUB_MODEL,
                            "policy": POLICY,
                            "summary": summarize(results),
                            "runs": results,
                        },
                        indent=1,
                    )
                )
    finally:
        worker.close()
    s = summarize(results)
    print("\narm       ok/runs  median    range")
    for arm, a in s["arms"].items():
        print(
            f"{arm:9} {a['n']}/{a['attempts']}      {a['median']:6.2f}s  {a['min']:5.2f}-{a['max']:5.2f}s"
        )
    print(
        f"opens allowed {s['opens_allowed']}/{s['opens_judged']}  searches refused {s['searches_refused']}/{s['searches_judged']}  "
        f"search never early: {s['search_never_early']}  no call ran twice: {s['no_call_ran_twice']}  answers correct: {s['answers_correct']}/{s['runs']}  errors: {s['errors']}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
