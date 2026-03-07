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


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
_llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0,
    base_url="http://localhost:11434",
)

_SYSTEM_PROMPT = """\
You are the Coder agent in a multi-agent system that resolves GitHub issues.

Your ONLY job is to produce a Unified Diff that fixes the bug described in
the issue.  You will be given:
  1. The GitHub issue text.
  2. Relevant source code snippets (with line numbers).
  3. (Optional) Error feedback from a previous failed attempt.

RULES -- follow these exactly:
- Be SURGICAL: change ONLY the lines necessary to fix the bug. Do not
  refactor, rename, or reorganise anything else.
- Output ONLY a standard Unified Diff inside a fenced code block, like:

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -start,count +start,count @@
 context line
-removed line
+added line
```

- The `@@` hunk headers MUST have accurate line numbers.  The code snippets
  you receive already include line numbers -- use them to compute the correct
  `@@ -start,count +start,count @@` values.
- Do NOT write any explanation, commentary, or conversational text.
- If multiple files need changes, include all of them in a single diff block
  with separate --- / +++ headers per file.
"""


# ---------------------------------------------------------------------------
# Helper: extract diff from markdown fenced block
# ---------------------------------------------------------------------------
def _extract_diff(llm_output: str) -> str:
    """Return the content between ```diff and ``` markers.

    Falls back to the raw output if no fenced block is found.
    """
    # Try regex first for robustness (handles optional whitespace)
    match = re.search(r"```diff\s*\n(.*?)```", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: simple string search
    start_marker = "```diff"
    end_marker = "```"
    start = llm_output.find(start_marker)
    if start != -1:
        start += len(start_marker)
        end = llm_output.find(end_marker, start)
        if end != -1:
            return llm_output[start:end].strip()

    # No markers found -- return raw output (best effort)
    return llm_output.strip()


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
    except Exception as exc:
        print(f"[Coder] [ERROR] LLM call failed: {exc}")
        return {"proposed_fix": ""}

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
    return {"proposed_fix": diff}
