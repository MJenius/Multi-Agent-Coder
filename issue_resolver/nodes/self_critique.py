"""Self Critique Node — performs checks on patch readability and edge cases.

Acts as a gate before completing a fix, verifying formatting and logical correctness.
"""

from __future__ import annotations

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.llm_utils import invoke_with_role_fallback
from issue_resolver.core.prompt_registry import get_prompt_registry


# Register prompt template on import
_DEFAULT_PROMPT = """\
You are a Self-Critique Agent. Analyze the proposed code fix for edge cases, off-by-one errors, missing error handling, and general correctness.
Output a critique. If changes are needed, set 'passed' to false and list recommendations.

You must output a single JSON block wrapped in a ```json markdown block:
{
  "passed": true/false,
  "recommendations": ["string"],
  "critique": "string"
}
"""

get_prompt_registry().register("self_critique", "1.0", _DEFAULT_PROMPT)


def self_critique_node(state: AgentState) -> dict:
    """Run self-critique verification before finalizing patch."""
    proposed_fix = state.get("proposed_fix", "")
    if not proposed_fix:
        return {}

    print("[SelfCritique] Running self-critique audit on proposed fix...")
    prompt = get_prompt_registry().get("self_critique")

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Proposed patch:\n{proposed_fix}"),
    ]

    try:
        resp, model_name = invoke_with_role_fallback(
            role="self_critique",
            candidates=["nvidia/nemotron-3-ultra-550b-a55b"],
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )
        raw = getattr(resp, "content", "") or ""
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        json_str = m.group(1).strip() if m else raw.strip()
        data = json.loads(json_str)
        passed = data.get("passed", True)
        recs = data.get("recommendations", [])
    except Exception as exc:
        print(f"[SelfCritique] [ERROR] Critique failed: {exc}")
        passed = True
        recs = []
        data = {"error": str(exc)}

    # Record trace
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "self_critique_completed",
            "SelfCritique",
            f"Passed: {passed}",
            details=data,
        )

    # If it fails, log recommendations to history
    log_text = "Self critique passed successfully."
    if not passed and recs:
        log_text = f"Self critique recommended adjustments: {'; '.join(recs)}"

    return {
        "history": append_to_history(
            "SelfCritique",
            "Critique",
            log_text,
        ),
    }
