from __future__ import annotations

import difflib
import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.config import CODER_MODEL_CANDIDATES, CODER_MAX_RETRIES
from issue_resolver.llm_utils import invoke_with_role_fallback, calculate_max_tokens


_SYSTEM_PROMPT = r"""\
You are a code fixing assistant. Output a minimal, surgical fix.

You MUST output your fix in ONE of these two formats:

FORMAT A — Unified Diff (PREFERRED):
```diff
--- a/path/to/file.ext
+++ b/path/to/file.ext
@@ -LINE,COUNT +LINE,COUNT @@
 context line
-old line to remove
+new line to add
 context line
```

FORMAT B — JSON Patch (FALLBACK):
```json
{
  "file": "path/to/file.ext",
  "hunks": [
    {
      "start_line": 42,
      "delete_lines": ["    old_code_line_1", "    old_code_line_2"],
      "insert_lines": ["    new_code_line_1", "    new_code_line_2"]
    }
  ]
}
```

CRITICAL RULES:
1. READ the issue carefully and identify what needs fixing
2. Keep changes MINIMAL — only modify the lines that fix the bug
3. Preserve exact indentation from the original source code
4. Output ONLY the diff or JSON patch block — no extra commentary
5. For unified diffs, include 3 lines of context around changes
6. For JSON patches, use exact line content from the source file
7. The file path must match one of the provided source files
"""

_DEBUGGING_MODE_PROMPT = r"""\

DEBUGGING MODE ACTIVATED
════════════════════════════════════════
Your previous fix attempt had the following error(s):

ERROR CONTEXT: {error_context}

STRATEGY FOR THIS RETRY:
1. The error indicates the failure is at specific line(s): {error_lines}
2. Re-examine the source code at those exact lines
3. Check if your previous diff introduced a syntax error
4. Verify your fix addresses the root cause
5. Ensure line numbers and context lines match the actual source

Remember: The test MUST pass after your fix is applied.
"""


def _strip_line_numbers(text: str) -> str:
    out = []
    for line in text.split("\n"):
        m = re.match(r"^\d+: (.*)$", line)
        out.append(m.group(1) if m else line)
    return "\n".join(out)


def _extract_file_info(file_context: list[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for snippet in file_context:
        m = re.search(r"# --- (?:\[HINTED\] )?file: (.+?) ---\n?", snippet)
        if m:
            path = m.group(1).lstrip("./")
            raw = snippet[m.end():]
            content = _strip_line_numbers(raw)
            content = re.sub(r"\n?\[TRUNCATED[^\]]*\]\s*$", "", content)
            files[path] = content
    return files


def _match_path(target: str, known: list[str]) -> str:
    t = target.lstrip("./").replace("sandbox_workspace/", "")
    if t in known:
        return t
    base = t.rsplit("/", 1)[-1] if "/" in t else t
    for k in known:
        if k.endswith("/" + base) or k == base:
            return k
    return t


def _make_diff(original: str, modified: str, path: str) -> str:
    a = original.splitlines(keepends=True)
    b = modified.splitlines(keepends=True)
    if a and not a[-1].endswith("\n"):
        a[-1] += "\n"
    if b and not b[-1].endswith("\n"):
        b[-1] += "\n"
    return "".join(difflib.unified_diff(a, b, f"a/{path}", f"b/{path}"))


def _parse_unified_diff(text: str) -> str:
    patterns = [
        r"```diff\n(.*?)```",
        r"```\n(---\s+a/.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            diff_content = match.group(1).strip()
            if "---" in diff_content and "+++" in diff_content:
                return diff_content
    
    lines = text.split("\n")
    diff_lines = []
    in_diff = False
    for line in lines:
        if line.startswith("--- a/") or line.startswith("--- "):
            in_diff = True
        if in_diff:
            if line.startswith(("---", "+++", "@@", " ", "-", "+", "\\")):
                diff_lines.append(line)
            elif not line.strip():
                diff_lines.append(line)
            else:
                if len(diff_lines) > 3:
                    break
    
    if diff_lines and any(l.startswith("---") for l in diff_lines):
        return "\n".join(diff_lines).strip()
    
    return ""


def _extract_json_block(text: str) -> str:
    json_match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    brace_depth = 0
    start_idx = -1
    for i, ch in enumerate(text):
        if ch == "{" and start_idx == -1:
            start_idx = i
            brace_depth = 1
        elif ch == "{" and start_idx != -1:
            brace_depth += 1
        elif ch == "}" and start_idx != -1:
            brace_depth -= 1
            if brace_depth == 0:
                return text[start_idx : i + 1]

    return ""


def _parse_json_patch(text: str, file_info: dict[str, str], known_paths: list[str]) -> str:
    raw_json = _extract_json_block(text)
    if not raw_json:
        return ""

    try:
        patch = json.loads(raw_json)
    except json.JSONDecodeError:
        cleaned = re.sub(r"//[^\n]*", "", raw_json)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
        try:
            patch = json.loads(cleaned)
        except json.JSONDecodeError:
            print("[Coder] [WARN] JSON patch parsing failed even after cleanup")
            return ""

    if not isinstance(patch, dict) or "file" not in patch or "hunks" not in patch:
        print("[Coder] [WARN] JSON patch missing required 'file' or 'hunks' fields")
        return ""

    target_file = _match_path(patch["file"], known_paths)
    if target_file not in file_info:
        print(f"[Coder] [WARN] JSON patch target '{target_file}' not in context")
        return ""

    original = file_info[target_file]
    original_lines = original.split("\n")
    modified_lines = list(original_lines)

    offset = 0
    for hunk in patch.get("hunks", []):
        start = hunk.get("start_line", 1) - 1 + offset
        delete_lines = hunk.get("delete_lines", [])
        insert_lines = hunk.get("insert_lines", [])
        delete_count = len(delete_lines)

        end = start + delete_count
        modified_lines[start:end] = insert_lines
        offset += len(insert_lines) - delete_count

    modified = "\n".join(modified_lines)
    if modified == original:
        print("[Coder] [WARN] JSON patch produced no changes")
        return ""

    return _make_diff(original, modified, target_file)


def _extract_issue_identifiers(issue_text: str) -> dict[str, list[str]]:
    high_priority: list[str] = []
    medium_priority: list[str] = []

    for m in re.finditer(r"`([^`\n]{2,120})`", issue_text):
        lit = m.group(1).strip()
        clean = re.sub(r'\([^)]*\)$', '', lit)
        high_priority.append(clean.lower())

    for m in re.finditer(r"'([^'\n]{2,120})'|\"([^\"\n]{2,120})\"", issue_text):
        lit = (m.group(1) or m.group(2) or "").strip()
        if lit:
            high_priority.append(lit.lower())

    for m in re.finditer(r"\b([a-z]{2}-[A-Z]{2})\b", issue_text):
        high_priority.append(m.group(1).lower())

    for m in re.finditer(r"\b([a-z]+[A-Z][a-zA-Z0-9]*|[A-Z][a-z]+[A-Z][a-zA-Z0-9]*)\b", issue_text):
        ident = m.group(1)
        if len(ident) > 3:
            medium_priority.append(ident.lower())

    for m in re.finditer(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b", issue_text):
        medium_priority.append(m.group(1).lower())

    def dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item not in seen and len(item) >= 2:
                seen.add(item)
                result.append(item)
        return result

    return {
        "high": dedupe(high_priority),
        "medium": dedupe(medium_priority),
        "low": [],
    }


def coder_node(state: AgentState) -> dict:
    print("[Coder] Generating code fix...")

    issue_text = state.get("issue", "(no issue)")
    file_context = state.get("file_context", [])
    errors = state.get("errors", "")
    iterations = state.get("iterations", 0)
    coder_retry_budget = state.get("coder_retry_budget", 3)

    from issue_resolver.config import MAX_ITERATIONS
    if iterations >= MAX_ITERATIONS:
        print(f"[Coder] [ABORT] Max iterations ({MAX_ITERATIONS}) reached")
        return {
            "errors": f"Max iterations ({MAX_ITERATIONS}) reached without successful fix",
            "iterations": iterations + 1,
            "history": append_to_history("Coder", "Aborted", "Max iterations reached"),
        }

    file_info = _extract_file_info(file_context)
    known_paths = list(file_info.keys())
    print(f"[Coder] Context files: {known_paths}")

    issue_identifiers = _extract_issue_identifiers(issue_text)
    print(f"[Coder] Extracted identifiers:")
    print(f"  High priority: {issue_identifiers.get('high', [])}")
    print(f"  Medium priority: {issue_identifiers.get('medium', [])}")

    ctx_parts = []
    for p, c in file_info.items():
        lines = c.split("\n")
        if len(lines) > 300:
            c = "\n".join(lines[:300]) + "\n[TRUNCATED]"
        ctx_parts.append(f"# === FILE: {p} ===\n{c}")
    full_context = "\n\n".join(ctx_parts) or "(no source code available)"

    base_parts = [f"## GitHub Issue\n{issue_text}"]
    if errors:
        base_parts.append(f"## Previous Errors (your last fix failed)\n{errors}")

    contribution_guidelines = state.get("contribution_guidelines", "")
    if contribution_guidelines:
        base_parts.append(
            f"## Repository Contribution Guidelines\n"
            f"Follow these guidelines when making your fix:\n{contribution_guidelines}"
        )

    all_identifiers = (
        issue_identifiers.get("high", []) + issue_identifiers.get("medium", [])
    )
    if all_identifiers:
        key_identifiers = all_identifiers[:5]
        base_parts.append(
            f"\nIMPORTANT: The issue mentions these specific identifiers: {', '.join(key_identifiers)}. "
            f"Your fix MUST address these exact identifiers from the source code."
        )

    base_parts.append(
        "\nProvide your fix as a unified diff (```diff block) or JSON patch (```json block)."
    )

    history: list[dict] = []

    estimated_prompt = "\n\n".join(base_parts) + "\n\n## Source Code\n" + full_context
    estimated_input_tokens = len(estimated_prompt) // 4
    first_model = CODER_MODEL_CANDIDATES[0] if CODER_MODEL_CANDIDATES else "meta/llama-3.3-70b-instruct"
    dynamic_max_tokens = calculate_max_tokens(first_model, estimated_input_tokens)
    print(f"[Coder] Dynamic token calc: input≈{estimated_input_tokens}, max_out={dynamic_max_tokens}")

    temperatures = [0.0]
    for i in range(CODER_MAX_RETRIES):
        temperatures.append(round(min(0.15, 0.1 * (i + 1)), 2))

    diff = ""
    plan = state.get("plan", "")
    last_failure = ""

    system_prompt = _SYSTEM_PROMPT
    if errors:
        error_context = state.get("test_error_context", "")[:200]
        error_lines = state.get("error_line_numbers", "")
        system_prompt += _DEBUGGING_MODE_PROMPT.format(
            error_context=error_context or "Test failed",
            error_lines=error_lines or "(not specified)",
        )
        print(f"[Coder] [DEBUGGING MODE ACTIVATED]")

    for attempt, temp in enumerate(temperatures):
        label = f"attempt {attempt + 1}/{len(temperatures)}, temp={temp}"
        print(f"[Coder] Calling LLM ({label})...")

        parts = base_parts.copy()

        if plan:
            parts.insert(1, f"## Fix Strategy (from Planner)\n{plan}")
            parts.insert(2, f"## Source Code\n{full_context}")
        else:
            parts.insert(1, f"## Source Code\n{full_context}")

        user_content = "\n\n".join(parts)

        msgs: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        if last_failure and attempt > 0:
            msgs.append(
                HumanMessage(
                    content=(
                        f"Your previous output could not be applied: {last_failure}\n"
                        f"Try again with the source context above. "
                        f"Output ONLY a ```diff or ```json block."
                    )
                )
            )

        try:
            resp, chosen_model = invoke_with_role_fallback(
                role="Coder",
                candidates=CODER_MODEL_CANDIDATES,
                messages=msgs,
                temperature=temp,
                max_tokens=dynamic_max_tokens,
            )
            if attempt == 0:
                print(f"[Coder] Using model: {chosen_model}")
        except Exception as exc:
            print(f"[Coder] [ERROR] LLM failed: {exc}")
            history.extend(append_to_history("Coder", "Error", str(exc)))
            last_failure = f"LLM error: {exc}"
            continue

        raw = getattr(resp, "content", "") or ""
        if not raw:
            print("[Coder] [ERROR] Empty LLM response")
            last_failure = "Empty LLM response"
            continue

        print(f"[Coder] LLM returned {len(raw)} chars")
        history.extend(append_to_history("Coder", "Generation", raw, max_length=800))

        diff = _parse_unified_diff(raw)
        if diff:
            print(f"[Coder] [OK] Parsed unified diff ({len(diff)} chars)")
            break

        diff = _parse_json_patch(raw, file_info, known_paths)
        if diff:
            print(f"[Coder] [OK] Parsed JSON patch → diff ({len(diff)} chars)")
            break

        last_failure = "Could not parse unified diff or JSON patch from LLM output"
        if attempt < len(temperatures) - 1:
            print(f"[Coder] [RETRY] {last_failure} — will retry with higher temperature")

    if not diff:
        print(f"[Coder] [ERROR] All {len(temperatures)} attempts failed")
        error_msg = (
            f"CODE FIX FAILED after {len(temperatures)} attempts.\n"
            f"Last failure: {last_failure}\n\n"
            f"REQUIRED FORMAT: ```diff or ```json block with unified diff or JSON patch."
        )
        history.extend(append_to_history("Coder", "Parse Failed", error_msg, max_length=500))
        return {"errors": error_msg, "history": history}

    print(f"[Coder] Final diff preview:\n{diff[:400]}")
    return {"plan": plan, "proposed_fix": diff, "history": history}