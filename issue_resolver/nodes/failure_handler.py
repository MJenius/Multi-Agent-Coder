from __future__ import annotations

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history


def failure_handler_node(state: AgentState) -> dict:
    print("[FailureHandler] Retry budget exhausted. Generating failure report...")

    errors = state.get("errors", "")
    proposed_fix = state.get("proposed_fix", "")
    iterations = state.get("iterations", 0)
    coder_retry_budget = state.get("coder_retry_budget", 0)
    ast_error_detail = state.get("ast_error_detail", "")
    error_category = state.get("error_category", "")
    history = state.get("history", [])

    error_entries = [
        entry for entry in history
        if entry.get("action") in ("Error", "Parse Failed", "Apply Patch Failed", "Test Execution")
    ]

    diagnostic_lines = [
        f"Total iterations: {iterations}",
        f"Remaining retry budget: {coder_retry_budget}",
        f"Last error category: {error_category}",
        f"Last errors: {errors[:500]}",
    ]

    if ast_error_detail:
        diagnostic_lines.append(f"Last AST validation error: {ast_error_detail}")

    if proposed_fix:
        diagnostic_lines.append(f"Last proposed fix preview: {proposed_fix[:300]}")

    if error_entries:
        diagnostic_lines.append(f"Total error events in history: {len(error_entries)}")
        for i, entry in enumerate(error_entries[-3:], 1):
            diagnostic_lines.append(
                f"  Error {i}: [{entry.get('node', '?')}] {entry.get('content', '')[:200]}"
            )

    failure_summary = "\n".join(diagnostic_lines)

    print(f"[FailureHandler] Failure summary:\n{failure_summary}")

    return {
        "is_resolved": False,
        "next_step": "end",
        "failure_summary": failure_summary,
        "history": append_to_history(
            "FailureHandler",
            "Budget Exhausted",
            failure_summary,
        ),
    }
