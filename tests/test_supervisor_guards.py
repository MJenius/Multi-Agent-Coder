"""Supervisor guard tests."""

from unittest.mock import patch
from langchain_core.messages import AIMessage
from issue_resolver.nodes.supervisor import supervisor_node


def test_terminal_coder_failure_ends_run():
    state = {
        "issue": "Some issue",
        "file_context": ["# --- file: foo.py ---\nprint('x')"],
        "proposed_fix": "",
        "errors": "CODE FIX FAILED after 3 attempts. Last failure: ...",
        "validation_status": "",
        "iterations": 1,
    }

    out = supervisor_node(state)
    assert out["next_step"] == "end"
    assert out["is_resolved"] is False


def test_researcher_loop_breaks_after_two_iterations():
    """After iterations >= 2 with empty file_context, supervisor must force coder."""
    state = {
        "issue": "Always use UTF-8 ECI mode when encoding in UTF-8",
        "file_context": [],
        "proposed_fix": "",
        "errors": "Search Dead-End",
        "validation_status": "",
        "iterations": 2,
    }
    with patch(
        "issue_resolver.nodes.supervisor.invoke_with_role_fallback",
        return_value=(AIMessage(content="researcher"), "fake-model"),
    ):
        out = supervisor_node(state)
    assert out["next_step"] == "coder", (
        "Supervisor must escape researcher loop after 2 iterations with no context"
    )


def test_researcher_forced_on_early_iterations():
    """During iterations < 2, any LLM decision must be overridden to 'researcher'."""
    state = {
        "issue": "Bug in fooBarBaz function",
        "file_context": [],
        "proposed_fix": "",
        "errors": "",
        "validation_status": "",
        "iterations": 0,
    }
    with patch(
        "issue_resolver.nodes.supervisor.invoke_with_role_fallback",
        return_value=(AIMessage(content="end"), "fake-model"),
    ):
        out = supervisor_node(state)
    assert out["next_step"] == "researcher", (
        "Supervisor must force researcher when file_context is empty and iterations < 2"
    )