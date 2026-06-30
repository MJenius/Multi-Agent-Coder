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
    current_view_file: str
    current_view_line: int

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

    # v2 State Fields
    issue_category: str
    verification_type: str
    repo_intelligence: dict
    repo_profile: dict
    context_confidence: dict
    structured_plan: dict
    candidate_patches: list[dict]
    candidate_scores: list[dict]
    critique_results: list[dict]
    verification_report: dict
    execution_trace: list[dict]

    # Localization and Suitability State
    localization_result: dict
    localization_confidence: float
    issue_suitability: dict
    classification_method: str

    # Execution Intelligence and Strategy
    execution_intelligence: dict
    adaptive_strategy: str

    # Metrics
    metrics: dict


