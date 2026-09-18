"""Judges: decide whether ONE concrete tool call may execute early.

A judge sees only what the tool author chose to expose: the tool's name and
description, the natural-language `speculate_when` policy, and the reduced
call inputs. It never sees tool source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_MODEL = "jev-1.13.0"
# Defaults chosen on the bash suite (evals/commands.py), then run unchanged on the
# held-out sql, http and injection suites (evals/holdout.py). Across all three, unsafe calls
# scored p_policy <= 0.15 and p_side_effect >= 0.70, so both cuts sit far from
# the unsafe cluster. Re-run the evals before moving them or changing the model.
DEFAULT_MIN_POLICY = 0.5
DEFAULT_MAX_SIDE_EFFECT = 0.25


@dataclass(frozen=True)
class Decision:
    """`allowed` is the only field the gate acts on; the rest is for logs/evals."""

    allowed: bool
    p_policy: float  # P(call satisfies the speculate_when policy)
    p_side_effect: float  # P(call changes state outside its return value)
    reason: str = ""


class Judge(Protocol):
    def decide(self, *, tool: str, description: str, policy: str, inputs: Any) -> Decision: ...


POLICY_QUESTION = {
    "question": "Does the call in `call` satisfy the early-execution policy in `policy`?",
    "inputs": "`tool` names and describes the tool. `call` holds the concrete arguments. "
    "`policy` is the tool author's rule for when the call may run early.",
    "focus": "Early execution means the call runs BEFORE the program is known to need it, "
    "and its result may be thrown away. Judge this exact call, not the tool in general. "
    "If the call could do anything the policy forbids, even as a side branch of a "
    "compound command, the answer is no.",
}
POLICY_CRITERIA = {
    "true": "Every part of the call is clearly inside what the policy permits.",
    "false": "Any part of the call is forbidden by the policy, outside what it permits, "
    "or too opaque to tell (for example it runs arbitrary code or an unknown script).",
}

SIDE_EFFECT_QUESTION = {
    "question": "Would executing the call in `call` change any state outside its own "
    "return value?",
    "inputs": "`tool` names and describes the tool. `call` holds the concrete arguments.",
    "focus": "State means files, databases, remote services, running processes, "
    "installed packages, version-control remotes, or messages sent to anyone. "
    "Reading, listing, searching, counting and fetching with a plain GET change nothing.",
}
SIDE_EFFECT_CRITERIA = {
    "true": "The call creates, modifies, deletes, installs, sends, pushes, posts, kills, "
    "or otherwise mutates something, or runs opaque code that might.",
    "false": "The call only reads or computes and returns a value.",
}


class JevJudge:
    """TypeSafe Jev judge. Two independent Nouls in ONE request (they run in
    parallel, so the second costs no latency); the call is allowed only when
    P(policy) >= min_policy AND P(side effect) <= max_side_effect. Any API
    failure declines."""

    def __init__(
        self,
        *,
        min_policy: float = DEFAULT_MIN_POLICY,
        max_side_effect: float = DEFAULT_MAX_SIDE_EFFECT,
        model: str = DEFAULT_MODEL,
        client: Any = None,
    ) -> None:
        from typesafe_sdk import Noul, TypeSafeClient

        if not (0.0 < min_policy < 1.0 and 0.0 < max_side_effect < 1.0):
            raise ValueError("min_policy and max_side_effect must be in (0, 1)")
        if client is None and not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("JevJudge needs TYPESAFE_API_KEY in the environment")
        self.min_policy = min_policy
        self.max_side_effect = max_side_effect
        self.model = model
        self._client = client or TypeSafeClient()
        self._questions = {
            "policy": Noul(instructions=POLICY_QUESTION, criteria=POLICY_CRITERIA),
            "side_effect": Noul(
                instructions=SIDE_EFFECT_QUESTION, criteria=SIDE_EFFECT_CRITERIA
            ),
        }

    def decide(self, *, tool: str, description: str, policy: str, inputs: Any) -> Decision:
        state = {
            "tool": {"name": tool, "description": description},
            "policy": policy,
            "call": inputs,
        }
        try:
            resp = self._client.system_one(state, self._questions, model=self.model)
            p_policy = float(resp.nouls["policy"].noul)
            p_side = float(resp.nouls["side_effect"].noul)
        except Exception as e:  # fail closed: no judgment, no early execution
            return Decision(False, 0.0, 1.0, reason=f"judge error: {type(e).__name__}")
        allowed = p_policy >= self.min_policy and p_side <= self.max_side_effect
        return Decision(allowed, p_policy, p_side)
