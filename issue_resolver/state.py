"""
AgentState — The shared state that flows through every node in the graph.

Each field is designed to be populated incrementally as the graph executes:
  - issue:        The raw GitHub issue text (immutable input).
  - file_context:  Code snippets discovered by the Researcher.
  - proposed_fix:  The diff / patch produced by the Coder.
  - errors:        Linter or test errors from the Reviewer (Phase 2+).
  - next_step:     Routing key set by the Supervisor ("researcher" | "coder" | "end").
  - iterations:    Loop counter to prevent infinite re-tries.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    """Typed dictionary representing the shared state of the agent graph."""

    issue: str
    file_context: list[str]
    proposed_fix: str
    errors: str
    next_step: str
    iterations: int
