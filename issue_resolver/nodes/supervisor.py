from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.config import (
    MAX_ITERATIONS,
    SUPERVISOR_MODEL_CANDIDATES,
    PLANNER_MAX_ITERATIONS,
    CODER_RETRY_BUDGET,
)
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
    file_context = state.get("file_context", [])
    plan = state.get("plan", "")
    test_code = state.get("test_code", "")
    test_runs_initially = state.get("test_runs_initially", None)
    proposed_fix = state.get("proposed_fix", "")
    errors = state.get("errors", "")
    validation_status = state.get("validation_status", "")
    error_category = state.get("error_category", "")
    plan_iteration = state.get("plan_iteration", 0)
    iterations = state.get("iterations", 0)
    ast_error_detail = state.get("ast_error_detail", "")
    ast_validation_passed = state.get("ast_validation_passed", True)

    coder_retry_budget = state.get("coder_retry_budget", -1)
    if coder_retry_budget == -1:
        coder_retry_budget = CODER_RETRY_BUDGET

    if iterations >= MAX_ITERATIONS:
        print(f"[Supervisor] [GUARD] Max iterations ({MAX_ITERATIONS}) reached. Forcing end.")
        summary_prompt = (
            f"The system failed to resolve the issue after {MAX_ITERATIONS} iterations. "
            f"Errors: {errors[:200]}"
        )
        try:
            summary_response, _ = invoke_with_role_fallback(
                role="Supervisor",
                candidates=SUPERVISOR_MODEL_CANDIDATES,
                messages=[HumanMessage(content=summary_prompt)],
                temperature=0,
            )
            failure_summary = summary_response.content.strip()
        except Exception:
            failure_summary = (
                f"System reached iteration limit ({MAX_ITERATIONS}). Last error: {errors[:100]}"
            )
        return {
            "next_step": "end",
            "iterations": iterations + 1,
            "coder_retry_budget": coder_retry_budget,
            "history": append_to_history("Supervisor", "Iteration Limit", failure_summary),
        }

    if coder_retry_budget <= 0 and proposed_fix and errors:
        print("[Supervisor] [GUARD] Coder retry budget exhausted. Routing to failure_handler.")
        return {
            "next_step": "failure_handler",
            "iterations": iterations + 1,
            "coder_retry_budget": coder_retry_budget,
            "history": append_to_history(
                "Supervisor",
                "Budget Exhausted",
                f"Coder retry budget is 0. Last errors: {errors[:300]}",
            ),
        }

    if plan and plan_iteration >= (PLANNER_MAX_ITERATIONS or 2):
        print(
            f"[Supervisor] [GUARD] Planner iteration limit ({PLANNER_MAX_ITERATIONS}) "
            f"reached. Forcing test_generator."
        )
        return {
            "next_step": "test_generator",
            "iterations": iterations + 1,
            "coder_retry_budget": coder_retry_budget,
            "history": append_to_history(
                "Supervisor",
                "Planner Limit",
                "Planner refinements exhausted. Proceeding to test generation.",
            ),
        }

    if isinstance(errors, str) and errors.startswith("CODE FIX FAILED after"):
        print("[Supervisor] [GUARD] Terminal coder failure detected. Ending run.")
        return {
            "next_step": "end",
            "iterations": iterations + 1,
            "is_resolved": False,
            "coder_retry_budget": coder_retry_budget,
            "history": append_to_history(
                "Supervisor",
                "Failure Summary",
                "Coder exhausted retries with no applicable fix. Ending run to avoid loop.",
            ),
        }

    if ast_error_detail and not ast_validation_passed:
        print(
            f"[Supervisor] [GUARD] AST validation failed. "
            f"Syntax error: {ast_error_detail[:200]}"
        )
        new_budget = coder_retry_budget - 1
        if new_budget <= 0:
            print("[Supervisor] [GUARD] AST failures exhausted retry budget. Routing to failure_handler.")
            return {
                "next_step": "failure_handler",
                "iterations": iterations + 1,
                "coder_retry_budget": new_budget,
                "errors": f"AST syntax error: {ast_error_detail}",
                "history": append_to_history(
                    "Supervisor",
                    "AST Failure → Budget Exhausted",
                    f"Syntax error detail: {ast_error_detail[:400]}",
                ),
            }
        print(f"[Supervisor] [GUARD] Routing back to coder with syntax context. Budget: {new_budget}")
        return {
            "next_step": "coder",
            "iterations": iterations + 1,
            "coder_retry_budget": new_budget,
            "proposed_fix": "",
            "errors": f"AST SYNTAX ERROR in your last fix:\n{ast_error_detail}\n\nFix the syntax error and regenerate the diff.",
            "ast_error_detail": "",
            "ast_validation_passed": True,
            "history": append_to_history(
                "Supervisor",
                "AST Failure → Coder Retry",
                f"Syntax error forwarded to coder: {ast_error_detail[:300]}",
            ),
        }

    # HARD GUARD: Tier-1 routing (deterministic; no LLM needed)
    # Priority order: Researcher/Planner (depending on context/iterations) -> TestGen -> Coder
    if not file_context and not plan:
        if iterations < 2:
            print("[Supervisor] [GUARD] No code context found. Routing to researcher.")
            return {
                "next_step": "researcher",
                "iterations": iterations + 1,
                "coder_retry_budget": coder_retry_budget,
            }
        else:
            print("[Supervisor] [GUARD] Researcher exhausted. Forcing planner.")
            return {
                "next_step": "planner",
                "iterations": iterations + 1,
                "coder_retry_budget": coder_retry_budget,
                "errors": (
                    "Research Dead-End: Could not locate relevant source files. "
                    "Generate a strategy based on the issue description."
                ),
            }

    if file_context and not plan:
        print("[Supervisor] [GUARD] Code found but no plan. Routing to planner.")
        return {
            "next_step": "planner",
            "iterations": iterations + 1,
            "coder_retry_budget": coder_retry_budget,
        }

    if plan and not test_code:
        print("[Supervisor] [GUARD] Plan found but no test. Routing to test_generator.")
        return {
            "next_step": "test_generator",
            "iterations": iterations + 1,
            "coder_retry_budget": coder_retry_budget,
        }

    if test_code and test_runs_initially is None:
        print("[Supervisor] [GUARD] Test generated but not yet validated. Routing to test_validator.")
        return {
            "next_step": "test_validator",
            "iterations": iterations + 1,
            "coder_retry_budget": coder_retry_budget,
        }

    if test_code and not proposed_fix:
        print("[Supervisor] [GUARD] Test validated. Now routing to coder for fix.")
        return {
            "next_step": "coder",
            "iterations": iterations + 1,
            "coder_retry_budget": coder_retry_budget,
        }

    if proposed_fix and not errors:
        if validation_status == "passed":
            print("[Supervisor] [GUARD] Tests passed. Terminating graph.")
            return {
                "next_step": "end",
                "iterations": iterations + 1,
                "is_resolved": True,
                "coder_retry_budget": coder_retry_budget,
            }
        if validation_status == "inconclusive":
            print("[Supervisor] [GUARD] Validation inconclusive. Accepting fix with warning.")
            return {
                "next_step": "end",
                "iterations": iterations + 1,
                "is_resolved": True,
                "coder_retry_budget": coder_retry_budget,
                "history": append_to_history(
                    "Supervisor",
                    "Warning",
                    "Fix accepted but NOT validated in sandbox. Manual verification recommended.",
                ),
            }
        if not validation_status:
            print("[Supervisor] [GUARD] No errors and no validation_status. Accepting fix.")
            return {
                "next_step": "end",
                "iterations": iterations + 1,
                "is_resolved": True,
                "coder_retry_budget": coder_retry_budget,
            }

    if (
        proposed_fix
        and errors
        and error_category == "LogicFailure"
        and plan
        and plan_iteration < (PLANNER_MAX_ITERATIONS - 1 or 1)
    ):
        print("[Supervisor] [GUARD] LogicFailure detected. Routing to planner for refinement.")
        return {
            "next_step": "planner",
            "iterations": iterations + 1,
            "coder_retry_budget": coder_retry_budget,
            "errors": f"Previous fix caused logic failure. Refine strategy. Error: {errors[:300]}",
            "history": append_to_history(
                "Supervisor",
                "Planner Refinement",
                "Test failure detected. Requesting strategy refinement.",
            ),
        }

    if proposed_fix and errors:
        new_budget = coder_retry_budget - 1
        if new_budget <= 0:
            print("[Supervisor] [GUARD] Coder retry budget exhausted after test failure. Routing to failure_handler.")
            return {
                "next_step": "failure_handler",
                "iterations": iterations + 1,
                "coder_retry_budget": new_budget,
                "history": append_to_history(
                    "Supervisor",
                    "Budget Exhausted",
                    f"Retry budget exhausted. Errors: {errors[:300]}",
                ),
            }
        print(f"[Supervisor] [GUARD] Test failed. Routing back to coder. Budget: {new_budget}")
        return {
            "next_step": "coder",
            "iterations": iterations + 1,
            "coder_retry_budget": new_budget,
            "proposed_fix": "",
            "history": append_to_history(
                "Supervisor",
                "Coder Retry",
                f"Budget {new_budget}. Errors: {errors[:200]}",
            ),
        }

    context_summary = (
        f"File context items: {len(file_context)}\n"
        f"Plan status: {'yes' if plan else 'no'}\n"
        f"Test code present: {'yes' if test_code else 'no'}\n"
        f"Proposed fix present: {'yes' if proposed_fix else 'no'}\n"
        f"Errors: {errors if errors else 'none'}\n"
        f"Iterations so far: {iterations}\n"
        f"Coder retry budget: {coder_retry_budget}"
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"GitHub Issue:\n{state.get('issue', '(not provided)')}\n\n"
                f"Current State:\n{context_summary}\n\n"
                "What is the next step? Reply with ONE word."
            )
        ),
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
        print(f"[Supervisor] [WARN] LLM call failed ({exc}); using rule-based fallback.")
        decision = _deterministic_decision(
            file_context, plan, test_code, proposed_fix, errors, validation_status
        )

    new_errors = errors
    if not file_context and decision != "researcher":
        if iterations < 2:
            print(f"[Supervisor] [GUARD] Overriding '{decision}' -> 'researcher' (no context).")
            decision = "researcher"
        else:
            print("[Supervisor] [GUARD] Researcher exhausted. Forcing planner.")
            decision = "planner"
            new_errors = (
                "Research Dead-End: Could not locate relevant source files. "
                "Generate a strategy based on the issue description."
            )

    valid_decisions = {"researcher", "planner", "test_generator", "coder", "end"}
    if decision not in valid_decisions:
        print(f"[Supervisor] [WARN] Unexpected LLM output '{decision}'; using fallback.")
        decision = _deterministic_decision(
            file_context, plan, test_code, proposed_fix, errors, validation_status
        )

    history_addition = append_to_history("Supervisor", "Routing Decision", decision)

    print(f"[Supervisor] [ROUTE] Decision -> {decision}  (iteration {iterations + 1})")

    out_state: dict = {
        "next_step": decision,
        "iterations": iterations + 1,
        "coder_retry_budget": coder_retry_budget,
        "history": history_addition,
    }
    if new_errors != errors:
        out_state["errors"] = new_errors

    return out_state


def _deterministic_decision(
    file_context: list[str],
    plan: str,
    test_code: str,
    proposed_fix: str,
    errors: str,
    validation_status: str = "",
) -> str:
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
    return "coder"
