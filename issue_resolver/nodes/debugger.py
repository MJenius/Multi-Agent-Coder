"""Debugger Node — performs root cause analysis on verification failures.

Runs when checks fail, inspects traces and knowledge graph calls, and enriches
context to instruct the next coding attempts.
"""

from __future__ import annotations

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.llm_utils import invoke_with_role_fallback
from issue_resolver.core.prompt_registry import get_prompt_registry
import issue_resolver.runtime_context as runtime_context


# Register prompt template on import
_DEFAULT_PROMPT = """\
You are an expert Debugger. You are given a repository issue description, a proposed patch that failed verification, and the compiler/lint/test execution error messages.
Your job is to analyze the root cause of the failure and output a bug diagnostics report.

You must output a single JSON block wrapped in a ```json markdown block:
{
  "root_cause": "Detailed explanation of why the proposed patch failed",
  "knowledge_graph_findings": "Related functions or classes that might be connected",
  "strategy_adjustments": "Specific adjustments recommended for the next coding attempt"
}
"""

get_prompt_registry().register("debugger", "1.0", _DEFAULT_PROMPT)


def debugger_node(state: AgentState) -> dict:
    """Diagnose build/test failure and provide strategy adjustments."""
    print("[Debugger] Diagnosing validation failure...")
    issue = state.get("issue", "")
    proposed_fix = state.get("proposed_fix", "")
    errors = state.get("errors", "")
    prompt = get_prompt_registry().get("debugger")

    graph = runtime_context.get_knowledge_graph()

    # Query knowledge graph for symbols related to error line numbers or trace
    graph_context = ""
    if graph:
        # Check error lines for function calls
        error_lines = state.get("error_line_numbers", "")
        m = re.findall(r"\d+", error_lines)
        related_symbols = []
        for line in m:
            for fn in graph.functions.values():
                if fn.line_number <= int(line) <= fn.end_line:
                    related_symbols.append(fn.qualified_name)
                    # Query callers
                    callers = graph.get_callers(fn.name)
                    related_symbols.extend(callers[:3])

        if related_symbols:
            graph_context = f"Knowledge Graph Proximity:\n- Related symbols: {', '.join(set(related_symbols))}"

    debug_input = f"""Issue Description:
{issue}

Proposed Patch that Failed:
{proposed_fix}

Verification Errors:
{errors}

{graph_context}
"""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=debug_input),
    ]

    try:
        resp, model_name = invoke_with_role_fallback(
            role="debugger",
            candidates=["deepseek-ai/deepseek-v4-flash"],
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        raw = getattr(resp, "content", "") or ""
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        json_str = m.group(1).strip() if m else raw.strip()
        diagnostics = json.loads(json_str)
    except Exception as exc:
        print(f"[Debugger] [ERROR] Diagnostics execution failed: {exc}")
        diagnostics = {
            "root_cause": f"Verification error details: {errors}",
            "knowledge_graph_findings": "None",
            "strategy_adjustments": "Retry the implementation step with additional verification checks.",
        }

    # Record trace
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "debugging_completed",
            "Debugger",
            "Completed root cause diagnosis",
            details=diagnostics,
        )

    # Append recommendations to plan
    plan_adjusted = (
        f"{state.get('plan', '')}\n\n"
        f"## Debugger Diagnostics:\n"
        f"- Root Cause: {diagnostics.get('root_cause')}\n"
        f"- Recommended Adjustments: {diagnostics.get('strategy_adjustments')}"
    )

    print(f"[Debugger] Root cause diagnosed: {diagnostics.get('root_cause')[:200]}...")

    return {
        "plan": plan_adjusted,
        "history": append_to_history(
            "Debugger",
            "Diagnose",
            f"Diagnosed failure: {diagnostics.get('root_cause')[:150]}",
        ),
    }
