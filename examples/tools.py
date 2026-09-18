"""A small tool set for `spec-ptc-jev lint examples.tools:spec`.

Three markings are deliberately wrong so the linter has something to find:
`send_report` (marked always-early but sends mail), `word_count` (pure but
unmarked) and `read_file` (a policy the call arguments cannot decide).
"""

from __future__ import annotations

import subprocess

from spec_ptc_jev import JevSpeculator

spec = JevSpeculator()


@spec.tool(speculatable=True, pure=True)
def llm_query(prompt: str) -> str:
    """Ask a language model one question and return its text answer."""
    raise NotImplementedError


@spec.tool(speculatable=True, pure=True)
def send_report(to: str, body: str) -> str:
    """Email the report body to a recipient and return the delivery receipt."""
    raise NotImplementedError


@spec.tool()
def word_count(text: str) -> int:
    """Count the words in a string."""
    return len(text.split())


@spec.tool(
    speculate_when="Read-only shell commands: listing, reading, searching, counting. "
    "Never anything that writes, deletes, installs, pushes, sends or runs a script.",
)
def bash(command: str) -> str:
    """Run a shell command in the project directory and return its stdout."""
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout


@spec.tool(speculate_when="Only when the file already exists and the user owns it.")
def read_file(path: str) -> str:
    """Return the contents of a text file."""
    with open(path) as f:
        return f.read()
