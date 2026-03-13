"""
Supervisor Node -- The "brain" of the agent graph.

Uses Groq-hosted models to decide the next step in the test-driven loop:
  1. If file_context is empty          -> route to "researcher"
  2. If file_context exists but no plan -> route to "planner"
  3. If plan exists but no test         -> route to "test_generator"
  4. If test exists but no fix          -> route to "coder"
  5. If fix validated successfully      -> route to "end"

The LLM acts as a reasoning layer on top of the deterministic rules,
providing a natural-language justification for each decision. Guards
prevent infinite loops (e.g., max 2 Planner refinements before forcing Coder).
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.config import MAX_ITERATIONS, SUPERVISOR_MODEL_CANDIDATES, PLANNER_MAX_ITERATIONS
from issue_resolver.llm_utils import invoke_with_role_fallback


_SYSTEM_PROMPT = """\
You are the Supervisor of a multi-agent system that resolves GitHub issues.
You must decide the NEXT action. Reply with EXACTLY one word -- one of:
  researcher  |  planner  |  test_generator  |  coder  |  end

Rules:
- If no relevant code snippets have been gathered yet     -> researcher
- If code snippets exist but no strategy is developed    -> planner
- If a strategy exists but no test has been generated    -> test_generator
- If a test exists but no fix has been proposed          -> coder
- If a fix exists and validation passed without errors   -> end
"""


def supervisor_node(state: AgentState) -> dict:
    """Evaluate the current state and decide where to route next."""

    # ------------------------------------------------------------------
    # Deterministic fast-path (cheap, no LLM call needed)
    # ------------------------------------------------------------------
    file_context = state.get("file_context", [])
    plan = state.get("plan", "")
    test_code = state.get("test_code", "")
    proposed_fix = state.get("proposed_fix", "")
    errors = state.get("errors", "")
    validation_status = state.get("validation_status", "")
    error_category = state.get("error_category", "")
    plan_iteration = state.get("plan_iteration", 0)
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

    # GUARD: Prevent infinite Planner loops (max 2 refinements)
    # After 2 Planner iterations, force progression to TestGen or Coder
    if plan and plan_iteration >= (PLANNER_MAX_ITERATIONS or 2):
        print(f"[Supervisor] [GUARD] Planner iteration limit ({PLANNER_MAX_ITERATIONS}) reached. Forcing test_generator.")
        return {
            "next_step": "test_generator",
            "iterations": iterations + 1,
            "history": append_to_history("Supervisor", "Planner Limit", "Planner refinements exhausted. Proceeding to test generation."),
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

    # HARD GUARD: Tier-1 routing (deterministic; no LLM needed)
    # Priority order: Researcher → Planner → TestGen → Coder → Validate
    if not file_context:
        print("[Supervisor] [GUARD] No code context found. Routing to researcher.")
        return {"next_step": "researcher", "iterations": iterations + 1}

    if file_context and not plan:
        print("[Supervisor] [GUARD] Code found but no plan. Routing to planner.")
        return {"next_step": "planner", "iterations": iterations + 1}

    if plan and not test_code:
        print("[Supervisor] [GUARD] Plan found but no test. Routing to test_generator.")
        return {"next_step": "test_generator", "iterations": iterations + 1}

    # NEW: Test validation guard (Phase 5 test-driven topology)
    # After test is generated, validate it runs (to reproduce the issue)
    if test_code and not isinstance(test_runs_initially, str):
        # test_runs_initially is a bool, not yet validated
        print("[Supervisor] [GUARD] Test generated but not yet validated. Routing to test_validator.")
        return {"next_step": "test_validator", "iterations": iterations + 1}

    if test_code and not proposed_fix:
        print("[Supervisor] [GUARD] Test validated. Now routing to coder for fix.")
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

    # SMART GUARD: On test failure with LogicFailure, consider Planner refinement
    # if we're still within iteration budget (plan_iteration < PLANNER_MAX_ITERATIONS - 1)
    if (proposed_fix and errors and error_category == "LogicFailure" and 
        plan and plan_iteration < (PLANNER_MAX_ITERATIONS - 1 or 1)):
        print("[Supervisor] [GUARD] LogicFailure detected and Planner refinement budget available. Routing to planner.")
        return {
            "next_step": "planner",
            "iterations": iterations + 1,
            "errors": f"Previous fix caused logic failure. Refine strategy. Error: {errors[:300]}",
            "history": append_to_history("Supervisor", "Planner Refinement", "Test failure detected. Requesting strategy refinement."),
        }

    # ------------------------------------------------------------------
    # LLM-assisted reasoning (adds flexibility for later phases)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # LLM-assisted reasoning (adds flexibility for later phases)
    # ------------------------------------------------------------------
    context_summary = (
        f"File context items: {len(file_context)}\n"
        f"Plan status: {'yes' if plan else 'no'}\n"
        f"Test code present: {'yes' if test_code else 'no'}\n"
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
        decision = _deterministic_decision(file_context, plan, test_code, proposed_fix, errors, validation_status)

    # HARD GUARD: Override LLM if it violates basic logic
    # Validate that decision aligns with state progression
    new_errors = errors
    if not file_context and decision != "researcher":
        if iterations < 2:
            print(f"[Supervisor] [GUARD] Overriding '{decision}' -> 'researcher' (no context, early stage).")
            decision = "researcher"
        else:
            print("[Supervisor] [GUARD] Researcher exhausted. Forcing planner with minimal context.")
            decision = "planner"
            new_errors = (
                "Research Dead-End: Could not locate relevant source files. "
                "Generate a strategy based on the issue description and general knowledge of the codebase."
            )
    
    # Validate the decision
    if decision not in ("researcher", "planner", "test_generator", "coder", "end"):
        print(f"[Supervisor] [WARN] Unexpected LLM output '{decision}'; using fallback.")
        decision = _deterministic_decision(file_context, plan, test_code, proposed_fix, errors, validation_status)
        
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
    plan: str,
    test_code: str,
    proposed_fix: str,
    errors: str,
    validation_status: str = "",
) -> str:
    """Pure rule-based routing without LLM call."""
    if not file_context:
        return "researcher"
    if not plan:
        return "planner"
    if not test_code:
        return "test_generator"
    if not proposed_fix:
        return "coder"
    if not errors and validation_status in ("passed", "inconclusive", ""):
        return "end"
    # Has errors -> retry coder with error feedback
    return "coder"
