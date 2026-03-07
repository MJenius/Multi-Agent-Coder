"""
Coder Node (Stub) -- Phase 1 placeholder.

In Phase 2 this will use an LLM + code-editing tools to produce
a patch / diff that resolves the issue.
"""

from __future__ import annotations

from issue_resolver.state import AgentState


def coder_node(state: AgentState) -> dict:
    """Stub: simulate generating a code fix."""
    print("[Coder] Agent Coder is thinking...")

    # Dummy diff output
    dummy_fix = (
        "--- a/src/utils.py\n"
        "+++ b/src/utils.py\n"
        "@@ -42,3 +42,5 @@\n"
        " def calculate_total(items):\n"
        "-    return sum(item.price for item in items)\n"
        "+    if not items:\n"
        "+        return 0\n"
        "+    return sum(item.price for item in items)\n"
    )

    print("[Coder] [OK] Proposed fix generated.")
    return {"proposed_fix": dummy_fix}
