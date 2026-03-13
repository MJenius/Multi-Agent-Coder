"""
Supervisor Node -- The "brain" of the agent graph.

Uses Groq-hosted models to decide the next step:
  1. If file_context is empty        -> route to "researcher"
  2. If file_context exists but no fix -> route to "coder"
  3. If a fix exists with no errors    -> route to "end"

The LLM acts as a reasoning layer on top of the deterministic rules,
providing a natural-language justification for each decision.  In Phase 1
the rules above dominate; in later phases the LLM will handle ambiguity
(e.g., partial fixes, flaky tests).
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.config import MAX_ITERATIONS, SUPERVISOR_MODEL_CANDIDATES
from issue_resolver.llm_utils import invoke_with_role_fallback


_SYSTEM_PROMPT = """\
You are the Supervisor of a multi-agent system that resolves GitHub issues.
You must decide the NEXT action.  Reply with EXACTLY one word -- one of:
  researcher  |  coder  |  end

Rules:
- If no relevant code snippets have been gathered yet -> researcher
- If code snippets exist but no fix has been proposed  -> coder
- If a fix exists and there are no outstanding errors   -> end
"""


def supervisor_node(state: AgentState) -> dict:
    """Evaluate the current state and decide where to route next."""

    # ------------------------------------------------------------------
    # Deterministic fast-path (cheap, no LLM call needed)
    # ------------------------------------------------------------------
    file_context = state.get("file_context", [])
    proposed_fix = state.get("proposed_fix", "")
    errors = state.get("errors", "")
    validation_status = state.get("validation_status", "")
    iterations = state.get("iterations", 0)

    # [HIGHEST PRIORITY] Prevent infinite loops: end if MAX_ITERATIONS reached
    if iterations >= MAX_ITERATIONS:
        print(f"[Supervisor] [GUARD] Max iterations ({MAX_ITERATIONS}) reached. Forcing end.")
        summary_prompt = f"The system failed to resolve the issue after {MAX_ITERATIONS} iterations. Errors: {errors[:200]}"
        try:
            summary_response, _ = invoke_with_role_fallback(
                role="Supervisor",
                candidates=SUPERVISOR_MODEL_CANDIDATES,
                messages=[HumanMessage(content=summary_prompt)],
                temperature=0,
            )
            failure_summary = summary_response.content.strip()
        except Exception:
            failure_summary = f"System reached iteration limit ({MAX_ITERATIONS}). Last error: {errors[:100]}"
        return {
            "next_step": "end",
            "iterations": iterations + 1,
            "history": append_to_history("Supervisor", "Iteration Limit", failure_summary),
        }

    # Terminal coder failure guard: avoid routing back to coder forever.
    if isinstance(errors, str) and errors.startswith("CODE FIX FAILED after"):
        print("[Supervisor] [GUARD] Terminal coder failure detected. Ending run.")
        return {
            "next_step": "end",
            "iterations": iterations + 1,
            "is_resolved": False,
            "history": append_to_history(
                "Supervisor",
                "Failure Summary",
                "Coder exhausted retries with no applicable fix. Ending run to avoid loop.",
            ),
        }

    # HARD GUARD: If we have code but no fix, route to coder (MUST happen before any LLM call)
    if file_context and not proposed_fix:
        print("[Supervisor] [GUARD] Code found but no fix proposed. Routing to coder.")
        return {"next_step": "coder", "iterations": iterations + 1}

    # HARD GUARD: Tri-state validation check
    if proposed_fix and not errors:
        if validation_status == "passed":
            print("[Supervisor] [GUARD] Tests passed. Terminating graph.")
            return {"next_step": "end", "iterations": iterations + 1, "is_resolved": True}
        if validation_status == "inconclusive":
            print("[Supervisor] [GUARD] Validation inconclusive (Docker unavailable). Accepting fix with warning.")
            return {
                "next_step": "end",
                "iterations": iterations + 1,
                "is_resolved": True,
                "history": append_to_history(
                    "Supervisor", "Warning",
                    "Fix accepted but NOT validated in sandbox. Manual verification recommended."
                ),
            }
        # validation_status is empty or unknown but no errors — treat as passed
        # (backwards compat: reviewer may not have set validation_status yet)
        if not validation_status:
            print("[Supervisor] [GUARD] No errors and no validation_status. Accepting fix.")
            return {"next_step": "end", "iterations": iterations + 1, "is_resolved": True}

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
        response, chosen_model = invoke_with_role_fallback(
            role="Supervisor",
            candidates=SUPERVISOR_MODEL_CANDIDATES,
            messages=messages,
            temperature=0,
        )
        print(f"[Supervisor] Using model: {chosen_model}")
        decision = response.content.strip().lower().split()[0]
    except Exception as exc:
        # If LLM call fails, fall back to deterministic logic
        print(f"[Supervisor] [WARN] LLM call failed ({exc}); using rule-based fallback.")
        decision = _deterministic_decision(file_context, proposed_fix, errors, validation_status)

    # HARD GUARD: Override LLM if it violates basic logic
    # For the first 2 iterations, force researcher if no context has been gathered yet.
    # After that, if researcher keeps returning empty-handed, force coder to try anyway —
    # better to attempt a fix with minimal context than to loop indefinitely.
    new_errors = errors
    if not file_context:
        if iterations < 2 and decision != "researcher":
            print(f"[Supervisor] [GUARD] Overriding '{decision}' -> 'researcher' (no context, early stage).")
            decision = "researcher"
        elif iterations >= 2 and decision == "researcher":
            # Researcher had enough chances; force coder to try with issue context alone.
            print("[Supervisor] [GUARD] Researcher exhausted attempts with no context found. Forcing coder.")
            decision = "coder"
            new_errors = (
                "Research Dead-End: Could not locate relevant source files after multiple attempts. "
                "Use the issue title and what you know about the codebase to propose the most likely fix. "
                "Call generate_repo_map or list_files first to orient yourself."
            )
        elif iterations > 0 and decision == "researcher":
            # Still in early stage but signalling dead-end for next researcher pass
            print("[Supervisor] [GUARD] Search Dead-End detected. Forcing broader search guidelines.")
            new_errors = (
                "Search Dead-End: Previous search found no relevant logic. "
                "You MUST broaden your search: try generate_repo_map, search for keywords "
                "from the issue title, or check build files and documentation."
            )

    # Validate the decision
    if decision not in ("researcher", "coder", "end"):
        print(f"[Supervisor] [WARN] Unexpected LLM output '{decision}'; using fallback.")
        decision = _deterministic_decision(file_context, proposed_fix, errors, validation_status)
        
    history_addition = append_to_history("Supervisor", "Routing Decision", decision)

    print(f"[Supervisor] [ROUTE] Decision -> {decision}  (iteration {iterations + 1})")
    
    out_state = {
        "next_step": decision, 
        "iterations": iterations + 1,
        "history": history_addition
    }
    if new_errors != errors:
        out_state["errors"] = new_errors
        
    return out_state


# ---------------------------------------------------------------------------
# Fallback: pure rule-based routing
# ---------------------------------------------------------------------------
def _deterministic_decision(
    file_context: list[str],
    proposed_fix: str,
    errors: str,
    validation_status: str = "",
) -> str:
    if not file_context:
        return "researcher"
    if not proposed_fix:
        return "coder"
    if not errors and validation_status in ("passed", "inconclusive", ""):
        return "end"
    # Has errors -> retry coder with error feedback
    return "coder"
