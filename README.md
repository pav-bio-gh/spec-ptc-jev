# spec-ptc-jev

A streaming REPL for code-writing agents. A small model,
[TypeSafe Jev](https://docs.typesafe.ai), decides which tool calls may start before
the agent has finished writing its code.

Agents that act by writing code (RLMs, CodeAct, programmatic tool calling) stream a
block of Python, and only then does anything run. Most of that block is often slow,
harmless reads: page loads, `SELECT`s, file reads, sub-model calls. They could start
while the model is still typing. The catch is that the same tool can also write:
`browse` opens a page or submits a form, `sql` selects or deletes, `bash` lists or
removes. A boolean per tool cannot express that. A sentence can:

```python
from spec_ptc_jev import SpecRepl

repl = SpecRepl()


@repl.tool(early_when="GET requests that only read data. Never POST, PUT, PATCH or DELETE.")
async def http(method: str, url: str) -> str:
    """Make an HTTP request and return the body."""
    ...


@repl.tool(early=True)  # always safe to start early
def llm_query(prompt: str) -> str: ...


@repl.tool  # a normal tool: never started early
def save_note(text: str) -> str: ...


result = await repl.arun(chat, "Which of these six packages needs the newest Python?")
print(result.answer)
```

That is the whole API.

- **Tools** are plain functions or `async def`. The model calls both as plain functions.
- **`chat(messages)`** is any function that streams your model's reply to OpenAI-style
  messages, as an async or a plain iterator.
- **`await repl.arun(chat, task)`** writes the model's instructions from your tools, runs
  each turn, feeds the output back, and stops when the model calls `final_answer(...)`.
  Your async tools run on your own event loop, so sessions and pools created in your app
  just work. In a script, `repl.run(chat, task)` is the same thing, blocking. If you
  already have a loop, `arun_turn(stream)` / `run_turn(stream)` run one turn.
- **Nothing starts early unless you say so.** With no tool marked `early` or
  `early_when`, this is a plain loop: the whole reply is generated, then run in order, and
  Jev is never contacted (no TypeSafe key needed). Marking a tool is the only opt-in.
  `SpecRepl(speculate=False)` switches it off even for marked tools.

It fits agents that act by writing code. It does not apply to agents that make one JSON
tool call per model turn: there is never a second call known in advance to start.

## Install

```bash
uv add git+https://github.com/pav-bio-gh/spec-ptc-jev
export TYPESAFE_API_KEY=...
```

## How it works

Three threads per turn.

- **Stream.** Takes tokens off the model and hands the text on. Never waits.
- **Planner.** Reads ahead over everything written so far. For every tool call whose
  arguments are already known, it asks the judge, all at once, and starts the allowed
  ones side by side. Five separate `browse(...)` lines, one `for url in urls:` loop, a
  loop still being typed, a loop over values only known at run time: all become
  parallel calls.
- **Executor.** Runs the code exactly once, statement by statement, as each statement
  is complete. A call that was started early picks up its result. A call that was not
  allowed early blocks right there until the block is complete and parses, then runs
  in place.

One Jev request per distinct call carries two independent yes/no questions: *does
this call satisfy the policy?* and *would it change state outside its return value?*
A call starts early only if `P(policy) >= 0.5` and `P(side effect) <= 0.25`. The judge
reads the tool's name, description, policy and arguments, never its source. Pass
`reduce=` to choose what leaves the process.

### Guarantees

- The code runs once, in one namespace, in program order.
- A call starts early only after its verdict says yes. An error or timeout is a no.
- A call that was not allowed early never runs before the block is complete, and never
  runs at all if the block does not parse.
- No stale results across tools. When a call that was not allowed early runs, every
  early result not yet used is dropped and nothing new starts until it has finished.
  This only sees registered tools. If a tool reads something the program changes in
  plain Python (a file, `os.environ`, a global), do not mark that tool early.
- Planning never runs model-written code. Arguments are evaluated only when they are
  literals, names, f-strings, indexing, `+ % *`, containers and a few pure builtins.

One difference from "generate everything, then run", only when a tool is marked early:
statements run as they arrive,
like an interactive Python session. If a later line turns out to be a syntax error,
the statements before it have already run, with tool calls among them only if the
judge allowed them.

## Results

All numbers are live runs against `jev-1.13.0` on 2026-09-20.

**Does the judge ever let a state-changing call start early?**
(`uv run python -m evals.run_gate_eval`)

| Suite | Unsafe allowed early | Safe allowed early | Jev p50 |
| --- | --- | --- | --- |
| bash (thresholds chosen here) | 0 / 66 | 51 / 55 | 0.20 s |
| sql (held out) | 0 / 19 | 15 / 15 | 0.20 s |
| http (held out) | 0 / 10 | 8 / 8 | 0.19 s |
| injection (held out; arguments that argue with the judge) | 0 / 8 | n/a | 0.22 s |

The held-out suites were written after seeing the bash results and labelled before
running, so they are held out from tuning but not blind. One author, 181 calls.

**Is it faster?** (`uv run python -m examples.browse_race 5`)

One agent turn: a live `gpt-4.1-mini` streams a block that opens five Wikipedia pages
in a real headless Chromium (waiting for network idle), asks a sub-model to order the
five people by birth year, then submits a Wikipedia search, which is the one call that
changes state. No injected latency. Five rounds, one arm at a time, in a fresh random
order each round. Every arm gets the same prompt, generated from the tool list by
`system_prompt()`; nothing in it hints at loops or parallel calls. Only runs with the
correct final answer count.

| Arm | Correct | Median | Range |
| --- | --- | --- | --- |
| generate, then run | 5 / 5 | 8.85 s | 8.64 to 8.99 s |
| [spec-ptc](https://github.com/alexzhang13/spec-ptc) as shipped (`browse` must stay unmarked) | 5 / 5 | 8.77 s | 8.61 to 10.95 s |
| this package, calls side by side but never early | 5 / 5 | 6.54 s | 6.02 to 7.23 s |
| this package | 5 / 5 | 4.85 s | 4.56 to 5.05 s |

In the 20 runs: Jev allowed 25 of 25 page opens and refused 5 of 5 searches; the search
never ran before the code was complete; no call ran twice; the final answer was correct
20 of 20 times; no errors.

Where the time goes: the third row is what running independent calls side by side buys
with no early start, about 2.3 s. Starting them while the model is still typing buys
the other 1.7 s. Jev is not always this generous: in an earlier run it refused one page
open on a borderline score (0.26 against a 0.25 limit) and that run took 8.71 s. A
refusal costs speed, never safety.

**A different job, nothing changed.** (`uv run python -m examples.pypi_check 5`) No
browser and no hand-written prompt: three plain tools (`http`, `llm_query`,
`save_note`) and `repl.run`. The model fetches six packages' metadata from PyPI, asks a
sub-model a question about them, and saves a note to disk. Five rounds, random order,
the answer (package names only) compared with PyPI fetched directly: 3.81 s
generate-then-run, 2.82 s with this package, 10 of 10 correct. Jev allowed every GET and
the file write never ran before the code was complete. This one is noisy: the model
takes one or two turns, and that swings a run about as much as the early starts do.

**Watch it live.** `uv run python -m examples.live`, open `http://127.0.0.1:8765`,
press Run. Three real turns race at once, each with its own model stream and browser.
Every bar is drawn from an event the process emits as it happens.

### What did not work

We also raced a code-writing agent against
[jev-ultrafast](https://github.com/browser-use/jev-ultrafast) on its own Google
Flights task, with its own verifier, on the same machine. It won: 4 of 5 verified at a
7.28 s median, against 4 of 5 at 7.26 s for our best arm with a far worse tail. Its
loop asks Jev one tiny question per click, so there is never a call known in advance
to start early. Records are in `examples/results/flights_head_to_head/`. This package
helps when an agent writes several calls in one go and the tools are slow. It has
nothing to offer a one-click-at-a-time loop.

## Linter

```bash
spec-ptc-jev lint examples.tools:spec
```

Checks each tool's marking against its own description, one Jev request per tool, and
exits 1 on an error so it can run in CI. It works on spec-ptc registries (see below).
Its questions were tuned on a five-tool fixture and have no held-out set.

## Using it with spec-ptc

`spec_ptc_jev.gate` is an adapter for code already built on Alex Zhang's spec-ptc:
`JevSpeculator().tool(speculate_when="...")` and `GatedTool` answer spec-ptc's per-call
gate with the same judge. The gate waits for its verdict, so spec-ptc only ever
launches calls the judge allowed.

We moved to our own loop after finding that spec-ptc, on a loop whose body contains an
`if`, starts the loop's calls early, withdraws them when the next line streams in, and
then runs each call again, one at a time. `tests/test_upstream_retraction.py`
reproduces this with a stock tool and no gate: three items, six calls.

## Limits

- The safety evals are small and have one author. Run them on your own tools and
  commands before trusting the thresholds.
- Jev lists prompt injection in `state` as a known weak spot. Eight of eight injection
  cases were refused, which proves little. The gate decides *when* a call may run, not
  *whether* it is allowed: keep dangerous tools behind the sandbox you would use anyway.
- "Read-only" is not "free": an early `SELECT` still loads the database.
  `SpecRepl(max_parallel=...)` bounds how many calls run at once.
- A call started early can be wasted if the program never uses it, or if a
  state-changing call runs first. Those are calls the judge said were safe to waste.
- The model calls tools as plain functions; it does not write `await` or `asyncio.gather`
  itself. With nothing marked early, async tools therefore run one at a time.
- One `SpecRepl` is one conversation and runs one turn at a time.
- If planning ever fails it switches itself off for that turn and the code runs normally.

## Develop

```bash
uv sync
uv run playwright install chromium-headless-shell   # for the examples
uv run pytest                                       # offline; a rule-based judge stands in for Jev
export TYPESAFE_API_KEY=... OPENAI_API_KEY=...
uv run python -m evals.run_gate_eval                # live
uv run python -m examples.browse_race 5             # live
uv run python -m examples.live                      # live, in the browser
```

## Credit

Speculative programmatic tool calling and the `spec-ptc` library are Alex Zhang's work
([blog](https://alexzhang13.github.io/blog/2026/spec-ptc/),
[repo](https://github.com/alexzhang13/spec-ptc)). MIT licensed.
