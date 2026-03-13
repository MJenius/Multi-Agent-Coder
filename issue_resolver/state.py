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

import operator
from typing import TypedDict, Annotated


class AgentState(TypedDict):
    """Typed dictionary representing the shared state of the agent graph."""

    # Input & routing
    issue: str
    repo_path: str
    next_step: str
    iterations: int
    is_resolved: bool
    
    # Context gathering
    file_context: list[str]
    symbol_map: str  # Tab-separated symbol map from Setup node
    
    # Planning & test generation
    plan: str  # Plain-text strategy from Planner node
    plan_iteration: int  # Counter for Planner refinements
    test_code: str  # Generated test code from TestGen node
    test_file_path: str  # Path where test should be written
    test_framework_used: str  # Detected framework (pytest, jest, xunit, etc.)
    test_runs_initially: bool  # Flag: confirm test fails before fix
    
    # Coding & validation
    proposed_fix: str  # Diff/patch from Coder node
    errors: str  # Linter/test errors from Reviewer
    validation_status: str  # "passed" | "failed" | "inconclusive"
    
    # Error categorization for debugging mode
    error_category: str  # SyntaxError | EnvironmentError | LogicFailure | FrameworkError
    test_error_context: str  # First 500 chars of error output
    error_line_numbers: str  # Extracted line numbers (e.g., "lines 45, 120")
    
    # Environment & guidelines
    environment_config: dict  # Language, framework, test detection results
    contribution_guidelines: str  # Project-specific coding standards
    history: Annotated[list[dict], operator.add]  # Audit trail of all decisions
