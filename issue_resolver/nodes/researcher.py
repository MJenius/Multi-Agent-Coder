"""
Researcher Node (Stub) — Phase 1 placeholder.

In Phase 2 this will use tools to search the codebase, read files,
and populate `file_context` with relevant code snippets.
"""

from __future__ import annotations

from issue_resolver.state import AgentState


def researcher_node(state: AgentState) -> dict:
    """Stub: simulate discovering relevant code snippets."""
    print("[Researcher] 🔍  Agent Researcher is thinking...")

    # Dummy update — pretend we found some relevant code
    dummy_snippet = (
        "# --- file: src/utils.py  lines 42-58 ---\n"
        "def calculate_total(items):\n"
        "    return sum(item.price for item in items)\n"
    )

    existing_context = list(state.get("file_context", []))
    existing_context.append(dummy_snippet)

    print(f"[Researcher] 📄  Found {len(existing_context)} snippet(s) so far.")
    return {"file_context": existing_context}
