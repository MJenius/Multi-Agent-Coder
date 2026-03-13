"""
Coder Node -- Generates code fixes using search-and-replace.

Uses Groq-hosted coding models. The LLM identifies buggy lines and provides
a corrected version.  A unified diff is then generated programmatically
using Python's difflib -- far more reliable than asking small LLMs to
produce raw unified diffs directly.
"""

from __future__ import annotations

import difflib
import re

from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.config import CODER_MODEL_CANDIDATES, CODER_NUM_PREDICT, CODER_MAX_RETRIES
from issue_resolver.llm_utils import invoke_with_role_fallback, calculate_max_tokens

_SYSTEM_PROMPT = r"""\
You are a code fixing assistant. Output a minimal, surgical fix.

FORMAT:
<plan>Brief description of what is broken and how you will fix it.</plan>
<fix>
FILE: path/to/file.ext
SEARCH:
(exact source lines to find)
REPLACE:
(corrected lines)
</fix>

CRITICAL RULES:
1. READ the issue carefully and identify what needs fixing
2. SEARCH must be EXACT lines from the provided source code (copy-paste exactly)
3. Keep indentation and spacing EXACTLY as shown in source
4. REPLACE must be the corrected version with SAME indentation
5. Keep changes MINIMAL (1-5 lines only) - no refactoring
6. Work with what you have: even partial file context is enough to infer the fix
7. If issue title is clear (e.g., "Always use UTF-8 ECI"), find where encoding is set and fix it
8. Output ONLY <plan> and <fix> tags - no extra commentary

SAFE ATTRIBUTE ACCESS FOR OPTIONAL FIELDS:
Special handling needed for objects with optional attributes (like Stripe models):
✅ CORRECT: value = getattr(obj, "attribute_name", None)
✅ CORRECT: if hasattr(obj, "attribute_name"): value = obj.attribute_name
❌ WRONG:  value = obj.attribute_name  (raises AttributeError if missing)

When accessing optional attributes, always:
- Use getattr(obj, "attr", default_value) to safely get with fallback
- Or check with hasattr(obj, "attr") before accessing
- This prevents AttributeError when attributes don't exist on all instances

DO NOT include:
❌ Line numbers: "157: ", "158: "
❌ Markdown fences: "```", "```javascript"
❌ "..." indicating omission

EXAMPLES OF ISSUES YOU CAN FIX WITH PARTIAL CODE:
✅ "Always use UTF-8 ECI mode" + partial file → find encoder config, add ECI flag
✅ "Fix null pointer in handler" + class file → find null check, add safety check
✅ "Missing return value" + method stub → infer return statement from issue context
✅ "AttributeError on optional field" → use getattr() or hasattr() for safe access

BE BOLD: Use the issue title + context to make the surgical fix.
"""

_DEBUGGING_MODE_PROMPT = r"""\

DEBUGGING MODE ACTIVATED
════════════════════════════════════════
Your previous fix attempt had the following error(s):

ERROR CONTEXT: {error_context}

STRATEGY FOR THIS RETRY:
1. The test error indicates the failure is at specific line(s): {error_lines}
2. Re-examine the SEARCH block - it must match the EXACT source code at those line numbers
3. Do NOT change your SEARCH block randomly; align it precisely to the error location
4. If you used fuzzy matching before and it failed, make the SEARCH more specific
5. Check if your REPLACE introduces a syntax error or type mismatch
6. Verify your fix addresses the root cause, not just suppressing the symptom

Remember: The test MUST pass after your fix is applied.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_line_numbers(text: str) -> str:
    """Remove line-number prefixes added by the read_file tool."""
    out = []
    for line in text.split("\n"):
        m = re.match(r"^\d+: (.*)$", line)
        out.append(m.group(1) if m else line)
    return "\n".join(out)


def _extract_file_info(file_context: list[str]) -> dict[str, str]:
    """Parse context snippets into {filepath: original_content}."""
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


def _extract_plan(text: str) -> str:
    s, e = text.find("<plan>"), text.find("</plan>")
    return text[s + 6:e].strip() if s != -1 and e != -1 else ""


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting from LLM output, preserving code content.

    Removes:
    - Markdown headers (##, ###, # prefixes)
    - Code fence lines (``` with optional language tag)
    - Line number prefixes like "157: "
    - Leading/trailing blank lines around sections
    """
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Remove code fence lines entirely
        if re.match(r"^```", stripped):
            continue
        # Strip markdown header prefixes (## FILE: -> FILE:)
        line = re.sub(r"^\s*#{1,6}\s+", "", line)
        # Strip line number prefixes like "157: " or "157:  "
        line = re.sub(r"^\s*\d+:\s*", "", line)
        out.append(line)
    return "\n".join(out)


def _parse_fix(text: str) -> tuple[str, str, str]:
    """Extract FILE / SEARCH / REPLACE from LLM output.

    Strategy:
    1. Try to extract from <fix>...</fix> XML tags (ideal format).
    2. Otherwise, preprocess with _strip_markdown() and parse
       FILE:/SEARCH:/REPLACE: markers via line-by-line state machine.
    """
    candidates = _parse_fix_candidates(text)
    if not candidates:
        return "", "", ""
    return candidates[0]


def _parse_fix_candidates(text: str) -> list[tuple[str, str, str]]:
    """Extract all FILE/SEARCH/REPLACE candidates from LLM output."""
    out: list[tuple[str, str, str]] = []

    # --- Strategy 1: XML fix tags (can be multiple) ---
    for fix_m in re.finditer(r"<fix>(.*?)</fix>", text, re.DOTALL):
        block = fix_m.group(1)
        file_m = re.search(r"FILE:\s*(.+)", block)
        search_m = re.search(r"SEARCH:\n(.*?)REPLACE:\n", block, re.DOTALL)
        if not (file_m and search_m):
            continue
        filepath = file_m.group(1).strip()
        replace_pos = block.find("REPLACE:\n") + len("REPLACE:\n")
        replace_text = block[replace_pos:].rstrip()
        out.append((filepath, search_m.group(1).rstrip("\n"), replace_text))

    if out:
        return out

    # --- Strategy 2: Strip markdown, then parse all FILE blocks ---
    cleaned = _strip_markdown(text)
    lines = cleaned.split("\n")
    i = 0
    while i < len(lines):
        file_m = re.search(r"FILE:\s*(.+)", lines[i])
        if not file_m:
            i += 1
            continue

        filepath = file_m.group(1).strip()
        i += 1

        # Find SEARCH marker for this file block
        while i < len(lines) and not re.search(r"SEARCH:", lines[i]):
            i += 1
        if i >= len(lines):
            break
        i += 1

        search_lines: list[str] = []
        while i < len(lines) and not re.search(r"REPLACE:", lines[i]):
            # Stop if a new file starts before REPLACE
            if re.search(r"FILE:\s*", lines[i]):
                break
            search_lines.append(lines[i])
            i += 1
        if i >= len(lines) or not re.search(r"REPLACE:", lines[i]):
            continue
        i += 1

        replace_lines: list[str] = []
        while i < len(lines):
            if re.search(r"FILE:\s*", lines[i]) or re.search(r"SEARCH:", lines[i]) or lines[i].strip() == "</fix>":
                break
            replace_lines.append(lines[i])
            i += 1

        search_text = "\n".join(search_lines).strip()
        replace_text = "\n".join(replace_lines).strip()
        if filepath and search_text and replace_text:
            out.append((filepath, search_text, replace_text))

    return out


def _extract_issue_identifiers(issue_text: str) -> dict[str, list[str]]:
    """Extract code identifiers from issue text, categorized by specificity.
    
    Returns dict with keys:
    - 'high': Highly specific identifiers (locale codes, IDs, specific strings in quotes)
    - 'medium': Code-like identifiers (function names, class names, variable names)
    - 'low': General terms that might be code-related
    """
    high_priority: list[str] = []
    medium_priority: list[str] = []
    
    # Backtick literals: `foo`, `isMobilePhone(...)`, `uz-UZ` (HIGH priority)
    for m in re.finditer(r"`([^`\n]{2,120})`", issue_text):
        lit = m.group(1).strip()
        # Remove function call parentheses for matching
        clean = re.sub(r'\([^)]*\)$', '', lit)
        high_priority.append(clean.lower())
    
    # Quoted literals: 'foo' or "foo" (HIGH priority)
    for m in re.finditer(r"'([^'\n]{2,120})'|\"([^\"\n]{2,120})\"", issue_text):
        lit = (m.group(1) or m.group(2) or "").strip()
        if lit:
            high_priority.append(lit.lower())
    
    # Locale codes: uz-UZ, ar-QA (HIGH priority - highly specific pattern)
    for m in re.finditer(r"\b([a-z]{2}-[A-Z]{2})\b", issue_text):
        high_priority.append(m.group(1).lower())
    
    # camelCase or PascalCase identifiers (MEDIUM priority)
    for m in re.finditer(r"\b([a-z]+[A-Z][a-zA-Z0-9]*|[A-Z][a-z]+[A-Z][a-zA-Z0-9]*)\b", issue_text):
        ident = m.group(1)
        if len(ident) > 3:  # Filter out very short matches
            medium_priority.append(ident.lower())
    
    # snake_case identifiers (MEDIUM priority)
    for m in re.finditer(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b", issue_text):
        medium_priority.append(m.group(1).lower())
    
    # Dedupe while preserving order
    def dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item not in seen and len(item) >= 2:
                seen.add(item)
                result.append(item)
        return result
    
    return {
        'high': dedupe(high_priority),
        'medium': dedupe(medium_priority),
        'low': []
    }


def _build_focus_context(
    file_info: dict[str, str],
    issue_identifiers: dict[str, list[str]],
    max_hits_per_file: int = 6,  # Reduced from 8 to 6 for speed
) -> str:
    """Build issue-relevant code snippets by finding identifiers in source files.

    Returns plain text snippets with line numbers, prioritizing high-specificity matches.
    """
    if not file_info or not any(issue_identifiers.values()):
        return ""

    # Build search pool: prioritize high, then medium priority identifiers
    search_pool = issue_identifiers.get('high', []) + issue_identifiers.get('medium', [])
    if not search_pool:
        return ""

    chunks: list[str] = []
    for path, content in file_info.items():
        lines = content.split("\n")
        
        # Find lines matching identifiers, with priority scoring
        scored_hits: list[tuple[int, int]] = []  # (line_num, priority_score)
        for i, line in enumerate(lines, start=1):
            ll = line.lower()
            score = 0
            # High priority identifiers get score 10
            for ident in issue_identifiers.get('high', []):
                if ident in ll:
                    score += 10
            # Medium priority get score 3
            for ident in issue_identifiers.get('medium', []):
                if ident in ll:
                    score += 3
            
            if score > 0:
                scored_hits.append((i, score))
        
        if not scored_hits:
            continue
        
        # Sort by score (descending) and take top hits
        scored_hits.sort(key=lambda x: x[1], reverse=True)
        top_hits = [ln for ln, _ in scored_hits[:max_hits_per_file]]
        
        # Build snippet with minimal context (just +/- 1 line)
        # NO LINE NUMBERS - confuses LLM into including them in SEARCH blocks
        snippet_lines: list[str] = []
        seen: set[int] = set()
        for ln in top_hits:
            # Only 1 line before and after for speed (reduced from 2)
            for k in range(max(1, ln - 1), min(len(lines), ln + 1) + 1):
                if k not in seen:
                    seen.add(k)
                    snippet_lines.append((k, lines[k - 1]))
        
        # Sort by line number for readability, then format WITHOUT line numbers
        snippet_lines.sort(key=lambda x: x[0])
        formatted_lines = [line for _, line in snippet_lines]
        chunks.append(f"# Focus file: {path}\n" + "\n".join(formatted_lines))

    return "\n\n".join(chunks)


def _is_candidate_relevant(
    search: str, 
    replace: str, 
    issue_identifiers: dict[str, list[str]]
) -> bool:
    """Check whether candidate fix touches issue-specific identifiers.

    Uses prioritized matching: high-priority identifiers (quotes, backticks, locale codes)
    must match if present. Falls back to medium-priority (code identifiers) if no high-priority exists.
    """
    high = issue_identifiers.get('high', [])
    medium = issue_identifiers.get('medium', [])
    
    if not high and not medium:
        return True  # No identifiers to check against

    text = f"{search}\n{replace}".lower()
    
    # If we have high-priority identifiers, at least one MUST match
    if high:
        matched = [ident for ident in high if ident in text]
        if matched:
            print(f"[Coder] Relevance ✓: Found high-priority identifier(s): {matched[:3]}")
            return True
        else:
            print(f"[Coder] Relevance ✗: Missing high-priority identifiers {high[:3]}")
            return False
    
    # Fall back to medium-priority identifiers (need at least one match)
    if medium:
        matched = [ident for ident in medium if ident in text]
        if matched:
            print(f"[Coder] Relevance ✓: Found medium-priority identifier(s): {matched[:3]}")
            return True
        else:
            print(f"[Coder] Relevance ✗: No match for medium-priority identifiers {medium[:3]}")
            return False
    
    return True


def _match_path(target: str, known: list[str]) -> str:
    """Fuzzy-match an LLM file path to a known context path."""
    t = target.lstrip("./").replace("sandbox_workspace/", "")
    if t in known:
        return t
    base = t.rsplit("/", 1)[-1] if "/" in t else t
    for k in known:
        if k.endswith("/" + base) or k == base:
            return k
    return t


def _find_and_replace(original: str, search: str, replace: str) -> str | None:
    """Find SEARCH in original, substitute with REPLACE.  Returns None on failure.

    Strategies (tried in order):
    1. Exact substring match
    2. Outer-whitespace-stripped match
    3. Per-line whitespace-normalised match
    4. Minimal-diff extraction from large SEARCH/REPLACE blocks
    """
    # Level 1: exact substring match
    if search in original:
        return original.replace(search, replace, 1)

    # Level 2: strip outer whitespace
    ss = search.strip()
    if ss and ss in original:
        return original.replace(ss, replace.strip(), 1)

    # Level 3: per-line whitespace normalization
    orig_lines = original.split("\n")
    search_lines = search.strip().split("\n")
    n = len(search_lines)
    search_stripped = [l.strip() for l in search_lines]

    for i in range(len(orig_lines) - n + 1):
        if [l.strip() for l in orig_lines[i:i + n]] == search_stripped:
            replace_lines = replace.strip().split("\n")
            return "\n".join(orig_lines[:i] + replace_lines + orig_lines[i + n:])

    # Level 4: extract the *actually-changed* lines from large blocks
    if n > 5:
        replace_lines_raw = replace.strip().split("\n")
        replace_stripped = [l.strip() for l in replace_lines_raw]

        # Find indices where SEARCH and REPLACE differ
        min_len = min(len(search_stripped), len(replace_stripped))
        changed_overlap = [
            idx for idx in range(min_len)
            if search_stripped[idx] != replace_stripped[idx]
        ]

        # Reject pure append/remove transformations in large blocks; these are
        # often unrelated refactors from weaker models.
        if not changed_overlap:
            return None

        if changed_overlap:
            first = changed_overlap[0]
            last = changed_overlap[-1]
            # Grab context: 1 line before first diff, up to 1 line after last diff within SEARCH
            ctx_start = max(0, first - 1)
            ctx_end = min(len(search_stripped), last + 2)
            sub_search = search_stripped[ctx_start:ctx_end]

            for i in range(len(orig_lines) - len(sub_search) + 1):
                if [l.strip() for l in orig_lines[i:i + len(sub_search)]] == sub_search:
                    # Preserve original indentation for replacement lines
                    indent = re.match(r"(\s*)", orig_lines[i]).group(1)
                    repl_start = ctx_start
                    repl_end = min(len(replace_stripped), last + 2)
                    sub_replace = []
                    for rl in replace_lines_raw[repl_start:repl_end]:
                        if rl.strip():  # non-empty
                            sub_replace.append(indent + rl.strip())
                        else:
                            sub_replace.append("")
                    return "\n".join(
                        orig_lines[:i] + sub_replace + orig_lines[i + len(sub_search):]
                    )

    return None


def _make_diff(original: str, modified: str, path: str) -> str:
    """Produce a unified diff."""
    a = original.splitlines(keepends=True)
    b = modified.splitlines(keepends=True)
    if a and not a[-1].endswith("\n"):
        a[-1] += "\n"
    if b and not b[-1].endswith("\n"):
        b[-1] += "\n"
    return "".join(difflib.unified_diff(a, b, f"a/{path}", f"b/{path}"))


def _extract_diff_fallback(text: str) -> str:
    """Pull a unified diff from a fenced code block (last resort)."""
    for marker in ("```diff", "```\n---"):
        pos = text.find(marker)
        if pos != -1:
            start = text.find("\n", pos) + 1
            end = text.find("```", start)
            return (text[start:end] if end != -1 else text[start:]).strip()
    return ""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def _attempt_fix(
    raw: str,
    file_info: dict[str, str],
    known_paths: list[str],
    issue_identifiers: dict[str, list[str]],
) -> tuple[str, str, str]:
    """Try to extract a usable diff from raw LLM output.

    Returns (diff, fpath, failure_reason).  diff is empty on failure.
    """
    candidates = _parse_fix_candidates(raw)
    if not candidates:
        reason = "Could not parse any FILE/SEARCH/REPLACE candidates"
        print(f"[Coder] [WARN] {reason}")
        return "", "", reason

    # Prefer smaller, more surgical changes first.
    candidates.sort(key=lambda c: len(c[1].split("\n")))

    last_reason = ""
    for idx, (fpath, search, replace) in enumerate(candidates, start=1):
        print(
            f"[Coder] Candidate {idx}/{len(candidates)}: "
            f"file={repr(fpath)} search={len(search)}ch replace={len(replace)}ch"
        )

        if not _is_candidate_relevant(search, replace, issue_identifiers):
            last_reason = (
                "Candidate not relevant to issue identifiers; likely unrelated refactor."
            )
            print(f"[Coder] [SKIP] {last_reason}")
            continue

        matched = _match_path(fpath, known_paths)
        print(f"[Coder] Fix target: {fpath} -> {matched}")
        if matched not in file_info:
            last_reason = f"File '{matched}' not in context (known: {known_paths})"
            print(f"[Coder] [WARN] {last_reason}")
            continue

        original = file_info[matched]
        modified = _find_and_replace(original, search, replace)
        if modified is None:
            last_reason = (
                "SEARCH block not found in source file. "
                "Copy the EXACT lines from source; avoid restructuring."
            )
            print(f"[Coder] [WARN] {last_reason}")
            continue
        if modified == original:
            last_reason = "Fix is a no-op (SEARCH == REPLACE)."
            print(f"[Coder] [WARN] {last_reason}")
            continue

        diff = _make_diff(original, modified, matched)
        print(f"[Coder] [OK] Generated diff ({len(diff)} chars)")
        return diff, matched, ""

    return "", "", (last_reason or "No usable candidate produced an applicable diff")


def coder_node(state: AgentState) -> dict:
    """Generate a code fix via search-and-replace, then produce a unified diff.

    On failure, retries up to CODER_MAX_RETRIES times with increasing
    temperature to get a different LLM response.
    """
    print("[Coder] Generating code fix...")

    issue_text = state.get("issue", "(no issue)")
    file_context = state.get("file_context", [])
    errors = state.get("errors", "")
    iterations = state.get("iterations", 0)
    
    # Safety check: don't run coder if we've exceeded max iterations
    from issue_resolver.config import MAX_ITERATIONS
    if iterations >= MAX_ITERATIONS:
        print(f"[Coder] [ABORT] Max iterations ({MAX_ITERATIONS}) reached")
        return {
            "errors": f"Max iterations ({MAX_ITERATIONS}) reached without successful fix",
            "iterations": iterations + 1,
            "history": append_to_history("Coder", "Aborted", "Max iterations reached")
        }

    file_info = _extract_file_info(file_context)
    known_paths = list(file_info.keys())
    print(f"[Coder] Context files: {known_paths}")

    # Extract identifiers from the issue with priority categorization
    issue_identifiers = _extract_issue_identifiers(issue_text)
    all_identifiers = (
        issue_identifiers.get('high', []) + 
        issue_identifiers.get('medium', [])
    )
    print(f"[Coder] Extracted identifiers from issue:")
    print(f"  High priority: {issue_identifiers.get('high', [])}")
    print(f"  Medium priority: {issue_identifiers.get('medium', [])}")

    # Build focused context (small, targeted snippets)
    focus_context = _build_focus_context(file_info, issue_identifiers)
    
    # Build full context (only used on retry if focused approach fails)
    ctx_parts = []
    for p, c in file_info.items():
        lines = c.split("\n")
        # Reduced from 300 to 150 lines for faster processing
        if len(lines) > 150:
            c = "\n".join(lines[:150]) + "\n[TRUNCATED]"
        ctx_parts.append(f"# === FILE: {p} ===\n{c}")
    full_context = "\n\n".join(ctx_parts) or "(no source code available)"

    # Build base prompt parts (reused across attempts)
    base_parts = [f"## GitHub Issue\n{issue_text}"]
    if errors:
        base_parts.append(f"## Previous Errors (your last fix failed)\n{errors}")

    contribution_guidelines = state.get("contribution_guidelines", "")
    if contribution_guidelines:
        base_parts.append(
            f"## Repository Contribution Guidelines\n"
            f"Follow these guidelines when making your fix:\n{contribution_guidelines}"
        )
    
    # Add guidance about key identifiers if we found any
    if all_identifiers:
        key_identifiers = all_identifiers[:5]  # Show top 5
        base_parts.append(
            f"\nIMPORTANT: The issue mentions these specific identifiers: {', '.join(key_identifiers)}. "
            f"Your fix MUST address these exact identifiers from the source code."
        )
    
    base_parts.append("\nProvide your fix using the FILE/SEARCH/REPLACE format.")

    history: list[dict] = []

    # --- Dynamic token allocation ---
    # Estimate input tokens from base prompt (rough: 1 token per 4 characters)
    estimated_prompt = "\n\n".join(base_parts) + "\n\n## Source Code\n" + (focus_context or full_context)
    estimated_input_tokens = len(estimated_prompt) // 4
    
    # Calculate dynamic max_tokens using the first model in candidates
    first_model = CODER_MODEL_CANDIDATES[0] if CODER_MODEL_CANDIDATES else "qwen-2.5-coder-32b"
    dynamic_max_tokens = calculate_max_tokens(first_model, estimated_input_tokens)
    print(f"[Coder] Dynamic token calc: input≈{estimated_input_tokens}, max_out={dynamic_max_tokens}")

    # --- Retry loop with smart context strategy ---
    # Attempt 1: Use focused context (fast, targeted)
    # Retries: Use full context (slower, comprehensive)
    temperatures = [0.0]
    for i in range(CODER_MAX_RETRIES):
        temperatures.append(round(min(0.15, 0.1 * (i + 1)), 2))  # Max temp reduced to 0.15
    diff = ""
    plan = state.get("plan", "")  # Retrieve plan from state if available
    last_failure = ""
    
    # Build system prompt with debugging mode if we're retrying
    system_prompt = _SYSTEM_PROMPT
    if iterations > 0:
        error_category = state.get("error_category", "Unknown")
        error_context = state.get("test_error_context", "")[:200]  # Limit to 200 chars
        error_lines = state.get("error_line_numbers", "")
        
        system_prompt += _DEBUGGING_MODE_PROMPT.format(
            error_context=error_context or "Test failed",
            error_lines=error_lines or "(not specified)"
        )
        print(f"[Coder] [DEBUGGING MODE ACTIVATED] Category: {error_category}")

    for attempt, temp in enumerate(temperatures):
        is_first_attempt = (attempt == 0)
        label = f"attempt {attempt + 1}/{len(temperatures)}, temp={temp}"
        
        # Smart context selection: focused first, full on retry
        if is_first_attempt and focus_context:
            context_to_use = focus_context
            context_label = "focused"
            print(f"[Coder] Using focused context ({len(focus_context)} chars)")
        else:
            context_to_use = full_context
            context_label = "full"
            print(f"[Coder] Using full context ({len(full_context)} chars)")
        
        print(f"[Coder] Calling LLM ({label}, {context_label} context)...")

        # Build prompt with selected context
        parts = base_parts.copy()
        
        # Add plan context if available
        if plan:
            parts.insert(1, f"## Fix Strategy (from Planner)\n{plan}")
            parts.insert(2, f"## Source Code\n{context_to_use}")
        else:
            parts.insert(1, f"## Source Code\n{context_to_use}")
        
        user_content = "\n\n".join(parts)

        # Build messages — on retry, append specific failure feedback
        msgs: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        if last_failure and not is_first_attempt:
            msgs.append(HumanMessage(
                content=(
                    f"Your previous output could not be applied: {last_failure}\n"
                    f"Try again with the full source context shown above. "
                    f"Output ONLY <plan> and <fix> tags. "
                    f"SEARCH must contain EXACT lines from the source code."
                )
            ))

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

        plan = _extract_plan(raw) or plan
        if plan:
            print(f"[Coder] Plan: {plan}")

        # Try primary: search-and-replace -> programmatic diff
        diff, _, failure_reason = _attempt_fix(
            raw,
            file_info,
            known_paths,
            issue_identifiers,
        )
        if diff:
            break

        # Try fallback: raw unified diff in a fenced block
        fb = _extract_diff_fallback(raw)
        if fb and "---" in fb and "+++" in fb:
            fb_lines = fb.split("\n")
            for i, line in enumerate(fb_lines):
                if line.startswith("--- a/") or line.startswith("+++ b/"):
                    fb_lines[i] = line[:6] + _match_path(line[6:], known_paths)
            diff = "\n".join(fb_lines)
            print(f"[Coder] [OK] Fallback diff ({len(diff)} chars)")
            break

        last_failure = failure_reason or "No usable fix in LLM output"
        if attempt < len(temperatures) - 1:
            print(f"[Coder] [RETRY] {last_failure} — will retry with higher temperature")

    # All attempts exhausted
    if not diff:
        print(f"[Coder] [ERROR] All {len(temperatures)} attempts failed")
        error_msg = (
            f"CODE FIX FAILED after {len(temperatures)} attempts.\n"
            f"Last failure: {last_failure}\n\n"
            f"REQUIRED FORMAT:\n"
            f"<plan>fix description</plan>\n"
            f"<fix>\n"
            f"FILE: path/to/file\n"
            f"SEARCH:\n"
            f"(exact code lines)\n"
            f"REPLACE:\n"
            f"(corrected code)\n"
            f"</fix>\n\n"
            f"The SEARCH block MUST contain exact lines copied from the source file."
        )
        history.extend(append_to_history("Coder", "Parse Failed", error_msg, max_length=500))
        return {"errors": error_msg, "history": history}

    print(f"[Coder] Final diff preview:\n{diff[:400]}")
    return {"plan": plan, "proposed_fix": diff, "history": history}