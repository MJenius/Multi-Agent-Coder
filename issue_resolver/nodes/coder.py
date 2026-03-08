"""
Coder Node -- Phase 3: LLM-driven Unified Diff generation.

Uses qwen2.5-coder:7b via ChatOllama to produce a surgical Unified Diff
that resolves the GitHub issue based on the file_context gathered by
the Researcher.
"""

from __future__ import annotations

import re

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.tools.sandbox_tools import apply_diff_in_sandbox, clean_sandbox, get_sandbox_container

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
_base_llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0,
    base_url="http://localhost:11434",
)

@tool
def lint_code(diff: str) -> str:
    """Applies your provisional diff to the sandbox, checks for syntax errors, and then reverts the sandbox.
    Use this to verify your code compiles before submitting the final diff.
    """
    sandbox = get_sandbox_container()
    if not sandbox:
        return "Error: sandbox container down."
    
    res = apply_diff_in_sandbox(diff, ".")
    if "Error" in res:
        clean_sandbox()
        return res
        
    check_csproj = sandbox.exec_run(["bash", "-c", "find . -name '*.csproj' | head -1"], workdir="/workspace")
    check_pkg = sandbox.exec_run(["test", "-f", "package.json"], workdir="/workspace")
    
    if check_csproj.output.strip():
        chk = sandbox.exec_run(["dotnet", "build"], workdir="/workspace")
    elif check_pkg.exit_code == 0:
        chk = sandbox.exec_run(["bash", "-c", "find . -name '*.js' -exec node -c {} +"], workdir="/workspace")
    else:
        chk = sandbox.exec_run(["bash", "-c", "python -m compileall ."], workdir="/workspace")
        
    output = chk.output.decode("utf-8", errors="ignore")
    clean_sandbox()
    
    if chk.exit_code == 0:
        return "Syntax OK! No compilation errors."
    return f"Syntax Errors Found:\n{output}"

_llm = _base_llm.bind_tools([lint_code])
_MAX_CODER_ROUNDS = 3

_SYSTEM_PROMPT = """\
You are a Senior Software Engineer. Your task is to resolve a GitHub issue by 
providing a functional fix in the form of a UNIFIED DIFF.

CRITICAL RULES:
1. PLAN FIRST: You MUST output a <plan>...</plan> block outlining the steps you will take to fix the issue.
2. VERIFY: You can use the `lint_code` tool to check your initial diff for syntax/compilation errors. 
   If there are errors, fix them before proceeding.
3. BE SURGICAL: Only modify the lines necessary to fix the reported bug.
4. REPLACE: Your diff MUST have a `-` (remove) line AND a `+` (add) line that changes the actual code logic.
5. NO-OP FORBIDDEN: A diff that only changes whitespace, blank lines, or docstrings is WRONG and will be REJECTED.
6. FORMAT: Output ONLY the final diff inside a ```diff ... ``` markdown block.

DIFF FORMAT RULES:
- MUST include file headers: `--- a/file_path` and `+++ b/file_path` before the hunks.
- MUST NOT include the line numbers (e.g., `1: `, `14: `) from the context in your generated diff. Remove `<line_number>: ` prefixes!
- Keep hunks SMALL: include only 1-2 context lines before and after the change.
- The @@ header line counts MUST match the body.
- Context lines start with a SPACE character.
- Removed lines start with `-`.
- Added lines start with `+`.
- Every hunk MUST be COMPLETE. Do NOT truncate or omit lines.
"""


# ---------------------------------------------------------------------------
# Helper: extract diff from markdown fenced block
# ---------------------------------------------------------------------------
def _extract_diff(llm_output: str) -> str:
    """Return the content between ```diff and ``` markers."""
    # Start marker
    start_marker = "```diff"
    start_pos = llm_output.find(start_marker)
    
    if start_pos == -1:
        # Fallback: maybe they just did ``` without diff?
        start_marker = "```"
        start_pos = llm_output.find(start_marker)

    if start_pos != -1:
        content_start = start_pos + len(start_marker)
        # Look for end marker
        end_pos = llm_output.find("```", content_start)
        if end_pos != -1:
            return llm_output[content_start:end_pos].strip()
        else:
            # Not closed? Take the rest of the output
            return llm_output[content_start:].strip()

    # No markers found -- return raw output (best effort)
    # But only if it looks like a diff (simple check)
    if "---" in llm_output and "+++" in llm_output:
        return llm_output.strip()
    
    return ""

def _extract_plan(llm_output: str) -> str:
    """Extracts the <plan> block from the output."""
    start = llm_output.find("<plan>")
    end = llm_output.find("</plan>")
    if start != -1 and end != -1:
        return llm_output[start+6:end].strip()
    return ""


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------
def coder_node(state: AgentState) -> dict:
    """Generate a Unified Diff that fixes the issue."""
    print("[Coder] Agent Coder is thinking...")

    issue_text = state.get("issue", "(no issue provided)")
    file_context = state.get("file_context", [])
    errors = state.get("errors", "")

    # -- Build the human message with all available context ----------------
    parts: list[str] = [f"## GitHub Issue\n{issue_text}"]

    if file_context:
        joined = "\n\n".join(file_context)
        parts.append(f"## Relevant Source Code\n{joined}")

    if errors:
        parts.append(
            f"## Previous Errors (use as feedback to improve your diff)\n{errors}"
        )

    parts.append(
        "First generate a <plan>, optionally use `lint_code` to verify your diff, "
        "and finally produce the Unified Diff inside a ```diff``` code block."
    )

    human_content = "\n\n".join(parts)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    history_additions: list[dict] = []
    final_diff = ""
    final_plan = ""

    for round_num in range(1, _MAX_CODER_ROUNDS + 1):
        print(f"[Coder]  |-- Round {round_num}/{_MAX_CODER_ROUNDS}")
        try:
            response = _llm.invoke(messages)
        except Exception as exc:
            print(f"[Coder] [ERROR] LLM call failed: {exc}")
            history_additions.extend(append_to_history("Coder", "Error", str(exc)))
            break

        messages.append(response)
        
        raw_output = response.content
        if raw_output:
            history_additions.extend(append_to_history("Coder", "Generation", raw_output, max_length=600))
            
            ext_plan = _extract_plan(raw_output)
            if ext_plan:
                final_plan = ext_plan
            ext_diff = _extract_diff(raw_output)
            if ext_diff:
                final_diff = ext_diff

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            print("[Coder] No more tool calls -- generating final diff.")
            break
            
        for tc in tool_calls:
            if tc["name"] == "lint_code":
                print("[Coder]    --> lint_code(...)")
                diff_arg = tc["args"].get("diff", "")
                
                log_payload = f"Tool: lint_code\nDiff Length: {len(diff_arg)}"
                history_additions.extend(append_to_history("Coder", "Tool Call", log_payload, max_length=150))
                
                res = lint_code.invoke({"diff": diff_arg})
                messages.append(ToolMessage(content=str(res), tool_call_id=tc["id"]))

    if final_diff:
        if "---" in final_diff and "+++" in final_diff:
            print("[Coder] [OK] Valid Unified Diff extracted.")
        else:
            print("[Coder] [WARN] Extracted text may not be a valid diff.")
    else:
        print("[Coder] [WARN] No diff content could be extracted.")

    print(f"[Coder] Proposed fix length: {len(final_diff)} chars.")
    return {
        "plan": final_plan,
        "proposed_fix": final_diff,
        "history": history_additions
    }
