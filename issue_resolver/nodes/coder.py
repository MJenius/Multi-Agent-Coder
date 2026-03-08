"""
Coder Node -- Phase 3: LLM-driven Unified Diff generation.

Uses qwen2.5-coder:7b via ChatOllama to produce a surgical Unified Diff
that resolves the GitHub issue based on the file_context gathered by
the Researcher.
"""

from __future__ import annotations

import re

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
_llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0,
    base_url="http://localhost:11434",
)

_SYSTEM_PROMPT = """\
You are a Senior Software Engineer. Your task is to resolve a GitHub issue by 
providing a functional fix in the form of a UNIFIED DIFF.

CRITICAL RULES:
1. BE SURGICAL: Only modify the lines necessary to fix the reported bug.
2. REPLACE the buggy line: Your diff MUST have a `-` (remove) line AND a `+` (add) line that changes the actual code logic. 
3. NO-OP FORBIDDEN: A diff that only changes whitespace, blank lines, or docstrings is WRONG and will be REJECTED.
4. Do NOT add a new return statement below an existing one — that creates unreachable dead code. Instead, REPLACE the existing return line.
5. Ignore any "line endings" errors from previous attempts — the system handles that automatically. Focus ONLY on the logic fix.
6. FORMAT: Output ONLY the diff inside a ```diff ... ``` markdown block. No extra commentary.

DIFF FORMAT RULES (VERY IMPORTANT - follow EXACTLY):
- Keep hunks SMALL: include only 1-2 context lines before and after the change.
- The @@ header line counts MUST match the body. For example, @@ -8,4 +8,4 @@ means
  4 lines of old content and 4 lines of new content follow.
- Context lines (unchanged) start with a SPACE character.
- Removed lines start with `-`.
- Added lines start with `+`.
- Every hunk MUST be COMPLETE. Do NOT truncate or omit lines.

EXAMPLE of a CORRECT diff:
```diff
diff --git a/src/utils.py b/src/utils.py
--- a/src/utils.py
+++ b/src/utils.py
@@ -8,4 +8,4 @@ def calculate_total(items):
     \"\"\"
-    return sum(item.price for item in items)
+    return sum(item.price for item in items) if items else 0
 
```

Notice: 4 lines in old (context, `-` line, context, blank) and 4 in new (context, `+` line, context, blank). Counts match. The `-` line REMOVES the old return, and the `+` line ADDS the fixed return.
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


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------
def coder_node(state: AgentState) -> dict:
    """Generate a Unified Diff that fixes the issue using qwen2.5-coder:7b."""
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
        "Now produce the Unified Diff that fixes this issue. "
        "Output ONLY the diff in a ```diff``` code block."
    )

    human_content = "\n\n".join(parts)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    # -- Call the LLM (single call, no tool loop) --------------------------
    try:
        response = _llm.invoke(messages)
        raw_output = response.content
        print(f"[Coder] LLM returned {len(raw_output)} chars.")
        
        # Log the full raw output (including reasoning)
        # We allow up to 600 chars of reasoning to be stored in history
        history_addition = append_to_history("Coder", "Reasoning & Generation", raw_output, max_length=600)
        
    except Exception as exc:
        print(f"[Coder] [ERROR] LLM call failed: {exc}")
        return {"proposed_fix": "", "history": append_to_history("Coder", "Error", str(exc))}

    # -- Extract the diff from the fenced code block ----------------------
    diff = _extract_diff(raw_output)

    if diff:
        # Quick sanity check: does it look like a unified diff?
        if "---" in diff and "+++" in diff:
            print("[Coder] [OK] Valid Unified Diff extracted.")
        else:
            print("[Coder] [WARN] Extracted text may not be a valid diff.")
    else:
        print("[Coder] [WARN] No diff content could be extracted.")

    print(f"[Coder] Proposed fix length: {len(diff)} chars.")
    return {
        "proposed_fix": diff,
        "history": history_addition
    }
