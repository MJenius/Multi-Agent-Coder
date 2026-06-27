from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.config import PLANNER_MODEL_CANDIDATES, NVIDIA_CONTEXT_WINDOWS
from issue_resolver.llm_utils import invoke_with_role_fallback, calculate_max_tokens


_SYSTEM_PROMPT = r"""\
You are the Planner. Given a GitHub issue and code context, write a PLAIN-TEXT strategic plan.

OUTPUT FORMAT:
<plan>
## Analysis
- Root cause: [What's broken and why]
- Scope: [What files must change]
- Invariants to preserve: [Code contracts, interfaces, etc.]

## Strategy
1. [Step 1 - which file, what change]
2. [Step 2 - dependent change]
3. [Step 3 - validation/test strategy]

## Edge Cases
- [Potential pitfall 1]
- [Potential pitfall 2]
</plan>

CRITICAL RULES:
1. Output ONLY the <plan> tags - no explanation before or after
2. Be specific about which files and functions will change
3. Reference line numbers if known from symbol map
4. Think about edge cases and error handling
5. Consider backwards compatibility
6. DO NOT write code or test syntax - only strategy
7. Keep it concise (under 300 words) but complete
8. GENERAL vs. SPECIFIC: If an issue concerns parameter serialization, data formats, parsing, or type encoding, prioritize finding and fixing the general serialization/utility components (e.g., encoders, helper classes, or common formatters) rather than target wrappers or specific endpoint resources.

DEFENSIVE CODING FOR OPTIONAL ATTRIBUTES:
When accessing attributes that may not exist on all instances:
  - Recommend: getattr(obj, 'attribute_name', default_value)
  - Recommend: if hasattr(obj, 'attribute_name'): value = obj.attribute_name
  - Alternative: try/except AttributeError blocks for attribute access
  - Avoid: Accessing obj.attribute_name directly without checks
"""


def planner_node(state: AgentState) -> dict:
    print("[Planner] Generating fix strategy...")

    issue_text = state.get("issue", "(no issue)")
    file_context = state.get("file_context", [])
    symbol_map = state.get("symbol_map", "")
    iterations = state.get("iterations", 0)
    plan_iteration = state.get("plan_iteration", 0)

    from issue_resolver.config import PLANNER_MAX_ITERATIONS
    if plan_iteration >= PLANNER_MAX_ITERATIONS:
        print(f"[Planner] Plan refinement limit ({PLANNER_MAX_ITERATIONS}) reached")
        return {
            "plan": state.get("plan", "(plan not generated)"),
            "plan_iteration": plan_iteration + 1,
            "history": append_to_history(
                "Planner",
                "Refinement Limit",
                f"Plan refinement reached {PLANNER_MAX_ITERATIONS} iterations.",
            ),
        }

    context_parts = []

    first_model = (
        PLANNER_MODEL_CANDIDATES[0]
        if PLANNER_MODEL_CANDIDATES
        else "nvidia/nemotron-3-super-120b-a12b"
    )
    context_window = NVIDIA_CONTEXT_WINDOWS.get(first_model, 131_072)

    available_for_context = context_window - 4000

    truncated_symbol_map = symbol_map
    if symbol_map and file_context:
        context_files = set()
        for item in file_context:
            if isinstance(item, str) and "File:" in item:
                file_path = item.split("File:")[1].split("\n")[0].strip()
                context_files.add(file_path)

        if context_files:
            symbol_lines = symbol_map.split("\n")
            relevant_symbols = [
                line
                for line in symbol_lines
                if any(fname in line for fname in context_files)
            ]
            if relevant_symbols:
                truncated_symbol_map = "\n".join(relevant_symbols[:30])
            else:
                truncated_symbol_map = "\n".join(symbol_lines[:20])
        else:
            truncated_symbol_map = "\n".join(symbol_map.split("\n")[:20])
    elif symbol_map:
        truncated_symbol_map = "\n".join(symbol_map.split("\n")[:20])

    if truncated_symbol_map:
        context_parts.append(f"## Symbol Map (Top Functions/Classes)\n{truncated_symbol_map}")

    if file_context:
        truncated_file_context = file_context[:4]
        context_parts.append(
            f"## Code Context\n" + "\n\n".join(truncated_file_context)
        )

    context_str = "\n\n".join(context_parts) if context_parts else "(no context available)"

    print(f"[Planner] Context optimization: {first_model} (window={context_window})")

    prompt_content = f"""{issue_text}

## Repository Structure
{context_str}

Based on the issue and repository context above, write a detailed fix strategy.
"""

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content),
    ]

    estimated_input_tokens = (len(_SYSTEM_PROMPT) + len(prompt_content)) // 4
    max_tokens = calculate_max_tokens(first_model, estimated_input_tokens, ratio=0.4)

    try:
        resp, chosen_model = invoke_with_role_fallback(
            role="Planner",
            candidates=PLANNER_MODEL_CANDIDATES,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        print(f"[Planner] Using model: {chosen_model}")
    except Exception as exc:
        print(f"[Planner] [ERROR] LLM failed: {exc}")
        error_msg = f"Planner failed to generate strategy: {exc}"
        return {
            "errors": error_msg,
            "iterations": iterations + 1,
            "history": append_to_history("Planner", "Error", error_msg),
        }

    raw = getattr(resp, "content", "") or ""
    if not raw:
        print("[Planner] [ERROR] Empty LLM response")
        return {
            "errors": "Planner returned empty response",
            "iterations": iterations + 1,
            "history": append_to_history("Planner", "Error", "Empty response"),
        }

    plan = ""
    s, e = raw.find("<plan>"), raw.find("</plan>")
    if s != -1 and e != -1:
        plan = raw[s + 6 : e].strip()
    else:
        plan = raw.strip()

    if not plan:
        return {
            "errors": "Could not extract plan from LLM output",
            "iterations": iterations + 1,
            "history": append_to_history("Planner", "Parse Failed", raw[:300]),
        }

    print(f"[Planner] Plan generated ({len(plan)} chars)")
    return {
        "plan": plan,
        "plan_iteration": plan_iteration + 1,
        "iterations": iterations + 1,
        "history": append_to_history("Planner", "Strategy Generated", plan[:400]),
    }
