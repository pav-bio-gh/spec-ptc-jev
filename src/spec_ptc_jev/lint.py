"""Lint a tool set's speculation markings against the tools' own descriptions.

The linter reads NO source code: only each tool's name, signature, description
and `speculate_when` policy. That keeps it small enough for one Jev request per
tool, makes it work on any codebase, and puts the burden where it belongs: a
tool description that states its side effects.

    spec-ptc-jev lint mypackage.tools:spec
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from dataclasses import dataclass
from typing import Any, Literal

from spec_ptc import Speculator

from spec_ptc_jev.gate import GatedTool
from spec_ptc_jev.judge import DEFAULT_MODEL

Marking = Literal["always", "gated", "never"]
Level = Literal["error", "warn", "info"]


@dataclass(frozen=True)
class ToolFacts:
    name: str
    signature: str
    description: str
    marking: Marking
    policy: str = ""


@dataclass(frozen=True)
class Finding:
    level: Level
    tool: str
    code: str
    message: str
    p: float | None = None


def collect(spec: Speculator) -> list[ToolFacts]:
    facts = []
    for name in spec.registry.names():
        record = spec.registry.get(name)
        assert record is not None
        owner = getattr(record.fn, "__self__", None)
        if isinstance(owner, GatedTool):
            facts.append(
                ToolFacts(
                    name,
                    str(inspect.signature(owner.fn)),
                    owner.description,
                    "gated",
                    owner.speculate_when,
                )
            )
            continue
        facts.append(
            ToolFacts(
                name,
                str(inspect.signature(record.fn)),
                (inspect.getdoc(record.fn) or "").strip(),
                "always" if record.speculatable else "never",
            )
        )
    return facts


_INPUTS = (
    "`tool.name`, `tool.signature` and `tool.description` are everything known about "
    "the tool. Do not assume behaviour the description does not state or imply."
)
_MUTATION = (
    "Mutating means creating, modifying or deleting files, database rows or remote "
    "resources; installing; sending messages; pushing; starting or killing processes. "
    "Reading, searching, computing, and asking a language model a question are not mutating."
)


def _questions(facts: ToolFacts) -> dict[str, Any]:
    from typesafe_sdk import Noul

    qs: dict[str, Any] = {
        "can_mutate": Noul(
            instructions={
                "question": "Can at least one valid call of this tool mutate state outside "
                "its return value?",
                "inputs": _INPUTS,
                "focus": _MUTATION,
            },
            criteria={
                "true": "The description states or implies some calls mutate, or the tool "
                "runs arbitrary commands, code, queries or requests.",
                "false": "Every call only reads or computes.",
            },
        )
    }
    if facts.marking == "gated":
        qs["policy_needs_outside_facts"] = Noul(
            instructions={
                "question": "To apply the rule in `policy` to one call, would a reader need "
                "facts that are not visible in that call's argument values?",
                "inputs": "`policy` is a rule for when a call may run early. " + _INPUTS,
                "focus": "Facts outside the arguments: whether a file or record exists, who "
                "owns something, who the user is, the time, or what an earlier call "
                "returned. A rule about WHAT the arguments ask for (for example 'only "
                "read-only commands', 'only SELECT statements', 'only GET requests') "
                "needs nothing beyond the arguments.",
            },
            criteria={
                "true": "The rule mentions a condition on the outside world that argument "
                "text cannot reveal.",
                "false": "Reading the argument values is enough to apply the rule.",
            },
        )
        qs["policy_excludes_mutation"] = Noul(
            instructions={
                "question": "Does the rule in `policy` restrict early execution to calls "
                "that do not mutate state?",
                "inputs": "`policy` is a rule for when a call may run early. " + _INPUTS,
                "focus": _MUTATION,
            },
            criteria={
                "true": "The policy permits only non-mutating calls, or explicitly forbids "
                "the mutating ones.",
                "false": "The policy would let a mutating call run early, or says nothing "
                "about mutation.",
            },
        )
    return qs


def lint_tool(facts: ToolFacts, client: Any, model: str = DEFAULT_MODEL) -> list[Finding]:
    if not facts.description:
        return [
            Finding(
                "warn",
                facts.name,
                "undescribed",
                "no description or docstring: nothing to check the marking against",
            )
        ]
    state = {
        "tool": {
            "name": facts.name,
            "signature": facts.signature,
            "description": facts.description,
        },
        "policy": facts.policy,
    }
    nouls = client.system_one(state, _questions(facts), model=model).nouls
    can_mutate = float(nouls["can_mutate"].noul)
    out: list[Finding] = []
    if facts.marking == "always" and can_mutate >= 0.5:
        out.append(
            Finding(
                "error",
                facts.name,
                "unsafe-always",
                "marked speculatable=True but the description reads as able to mutate "
                "state; use speculate_when= or leave it unmarked",
                can_mutate,
            )
        )
    if facts.marking == "never" and can_mutate <= 0.1:
        out.append(
            Finding(
                "info",
                facts.name,
                "missed-speedup",
                "unmarked, but the description reads as pure; speculatable=True would "
                "let it run early",
                can_mutate,
            )
        )
    if facts.marking == "gated":
        if can_mutate <= 0.1:
            out.append(
                Finding(
                    "info",
                    facts.name,
                    "gate-unneeded",
                    "gated, but the description reads as always pure; plain "
                    "speculatable=True skips the per-call judgment",
                    can_mutate,
                )
            )
        outside = float(nouls["policy_needs_outside_facts"].noul)
        if outside >= 0.5:
            out.append(
                Finding(
                    "warn",
                    facts.name,
                    "policy-undecidable",
                    "speculate_when depends on facts the call arguments do not carry; "
                    "the per-call judge cannot apply it reliably",
                    outside,
                )
            )
        excludes = float(nouls["policy_excludes_mutation"].noul)
        if can_mutate > 0.1 and excludes < 0.5:
            out.append(
                Finding(
                    "error",
                    facts.name,
                    "policy-permits-mutation",
                    "speculate_when would let a mutating call run early",
                    excludes,
                )
            )
    return out


def lint(spec: Speculator, client: Any = None, model: str = DEFAULT_MODEL) -> list[Finding]:
    if client is None:
        from typesafe_sdk import TypeSafeClient

        client = TypeSafeClient()
    findings: list[Finding] = []
    for facts in collect(spec):
        findings.extend(lint_tool(facts, client, model))
    return findings


def _load(target: str) -> Speculator:
    module_name, _, attr = target.partition(":")
    if not attr:
        raise SystemExit("target must look like package.module:speculator_attribute")
    obj = getattr(importlib.import_module(module_name), attr)
    if not isinstance(obj, Speculator):
        raise SystemExit(f"{target} is a {type(obj).__name__}, not a spec_ptc Speculator")
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spec-ptc-jev")
    sub = parser.add_subparsers(dest="command", required=True)
    lint_cmd = sub.add_parser("lint", help="check speculation markings against descriptions")
    lint_cmd.add_argument("target", help="package.module:speculator_attribute")
    lint_cmd.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    sys.path.insert(0, "")  # CLI entrypoint: resolve the target from the cwd
    findings = lint(_load(args.target), model=args.model)
    for f in findings:
        p = "" if f.p is None else f"  (p={f.p:.2f})"
        print(f"{f.level.upper():5} {f.tool}: {f.code}: {f.message}{p}")
    if not findings:
        print("no findings")
    return 1 if any(f.level == "error" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
