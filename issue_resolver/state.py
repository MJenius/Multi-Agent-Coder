from __future__ import annotations

import operator
from typing import TypedDict, Annotated


class AgentState(TypedDict):
    issue: str
    repo_path: str
    next_step: str
    iterations: int
    is_resolved: bool

    file_context: list[str]
    symbol_map: str

    plan: str
    plan_iteration: int
    test_code: str
    test_file_path: str
    test_framework_used: str
    test_runs_initially: bool

    proposed_fix: str
    errors: str
    validation_status: str

    error_category: str
    test_error_context: str
    error_line_numbers: str

    coder_retry_budget: int
    failure_summary: str
    ast_validation_passed: bool
    ast_error_detail: str

    environment_config: dict
    contribution_guidelines: str
    history: Annotated[list[dict], operator.add]
