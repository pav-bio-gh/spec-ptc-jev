"""Natural-language speculation gates for spec-ptc, judged by TypeSafe Jev."""

from spec_ptc_jev.gate import GatedTool, JevSpeculator, default_reducer
from spec_ptc_jev.judge import Decision, JevJudge, Judge
from spec_ptc_jev.lint import Finding, lint

__all__ = [
    "Decision",
    "Finding",
    "GatedTool",
    "JevJudge",
    "JevSpeculator",
    "Judge",
    "default_reducer",
    "lint",
]
