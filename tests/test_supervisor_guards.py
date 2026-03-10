"""Supervisor guard tests."""

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