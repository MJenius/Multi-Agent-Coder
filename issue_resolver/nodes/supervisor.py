"""
Supervisor Node — The "brain" of the agent graph.

Uses llama3.2:3b via ChatOllama to decide the next step:
  1. If file_context is empty        → route to "researcher"
  2. If file_context exists but no fix → route to "coder"
  3. If a fix exists with no errors    → route to "end"

The LLM acts as a reasoning layer on top of the deterministic rules,
providing a natural-language justification for each decision.  In Phase 1
the rules above dominate; in later phases the LLM will handle ambiguity
(e.g., partial fixes, flaky tests).
"""

from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState


# ---------------------------------------------------------------------------
# LLM setup — swap model / base_url as needed
# ---------------------------------------------------------------------------
_llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0,          # deterministic routing
    base_url="http://localhost:11434",
)

_SYSTEM_PROMPT = """\
You are the Supervisor of a multi-agent system that resolves GitHub issues.
You must decide the NEXT action.  Reply with EXACTLY one word — one of:
  researcher  |  coder  |  end

Rules:
• If no relevant code snippets have been gathered yet → researcher
• If code snippets exist but no fix has been proposed  → coder
• If a fix exists and there are no outstanding errors   → end
"""


def supervisor_node(state: AgentState) -> dict:
    """Evaluate the current state and decide where to route next."""

    # ------------------------------------------------------------------
    # Deterministic fast-path (cheap, no LLM call needed)
    # ------------------------------------------------------------------
    file_context = state.get("file_context", [])
    proposed_fix = state.get("proposed_fix", "")
    errors = state.get("errors", "")
    iterations = state.get("iterations", 0)

    # Safety valve: prevent infinite loops
    if iterations >= 5:
        print("[Supervisor] [WARN] Max iterations reached -- forcing end.")
        return {"next_step": "end", "iterations": iterations + 1}

    # ------------------------------------------------------------------
    # LLM-assisted reasoning (adds flexibility for later phases)
    # ------------------------------------------------------------------
    context_summary = (
        f"File context items: {len(file_context)}\n"
        f"Proposed fix present: {'yes' if proposed_fix else 'no'}\n"
        f"Errors: {errors if errors else 'none'}\n"
        f"Iterations so far: {iterations}"
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"GitHub Issue:\n{state.get('issue', '(not provided)')}\n\n"
            f"Current State:\n{context_summary}\n\n"
            "What is the next step? Reply with ONE word."
        )),
    ]

    try:
        response = _llm.invoke(messages)
        decision = response.content.strip().lower().split()[0]
    except Exception as exc:
        # If Ollama is unreachable, fall back to deterministic logic
        print(f"[Supervisor] [WARN] LLM call failed ({exc}); using rule-based fallback.")
        decision = _deterministic_decision(file_context, proposed_fix, errors)

    # Validate the decision
    if decision not in ("researcher", "coder", "end"):
        print(f"[Supervisor] [WARN] Unexpected LLM output '{decision}'; using fallback.")
        decision = _deterministic_decision(file_context, proposed_fix, errors)

    print(f"[Supervisor] [ROUTE] Decision -> {decision}  (iteration {iterations + 1})")
    return {"next_step": decision, "iterations": iterations + 1}


# ---------------------------------------------------------------------------
# Fallback: pure rule-based routing
# ---------------------------------------------------------------------------
def _deterministic_decision(
    file_context: list[str],
    proposed_fix: str,
    errors: str,
) -> str:
    if not file_context:
        return "researcher"
    if not proposed_fix:
        return "coder"
    if not errors:
        return "end"
    # Has errors → re-research
    return "researcher"
