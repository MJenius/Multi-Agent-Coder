"""
Coder Node -- Generates code fixes using search-and-replace.

Uses qwen2.5-coder:7b via ChatOllama.  The LLM identifies buggy lines and provides
a corrected version.  A unified diff is then generated programmatically
using Python's difflib -- far more reliable than asking small LLMs to
produce raw unified diffs directly.
"""

from __future__ import annotations

import difflib
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
_llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0,
    base_url="http://localhost:11434",
    num_predict=1024,
)

_SYSTEM_PROMPT = """\
You are a Senior Software Engineer. Fix the bug described in the GitHub issue.

You MUST use this EXACT output format:

<plan>1-2 sentence fix description</plan>

<fix>
FILE: path/to/file.ext
SEARCH:
(exact lines from source that need changing)
REPLACE:
(corrected lines)
</fix>

EXAMPLE:
<plan>Swap the inverted default constants so BLACK maps to filled and WHITE maps to space.</plan>

<fix>
FILE: QRCoder/AsciiQRCode.cs
SEARCH:
            WHITE_ALL = "\\u2588",
            WHITE_BLACK = "\\u2580",
            BLACK_WHITE = "\\u2584",
            BLACK_ALL = " ",
REPLACE:
            WHITE_ALL = " ",
            WHITE_BLACK = "\\u2584",
            BLACK_WHITE = "\\u2580",
            BLACK_ALL = "\\u2588",
</fix>

RULES:
1. Copy EXACT lines from the source code into SEARCH (including indentation).
2. Keep changes SURGICAL -- only the minimum lines needed.
3. Do NOT rewrite whole functions or add/remove methods.
4. Output ONLY the <plan> and <fix> blocks. Nothing else.
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


def _parse_fix(text: str) -> tuple[str, str, str]:
    """Extract FILE / SEARCH / REPLACE from LLM output."""
    fix_m = re.search(r"<fix>(.*?)</fix>", text, re.DOTALL)
    block = fix_m.group(1) if fix_m else text

    file_m = re.search(r"FILE:\s*(.+)", block)
    if not file_m:
        return "", "", ""

    search_m = re.search(r"SEARCH:\n(.*?)REPLACE:\n", block, re.DOTALL)
    if not search_m:
        return file_m.group(1).strip(), "", ""

    replace_pos = block.find("REPLACE:\n") + len("REPLACE:\n")
    replace_text = block[replace_pos:].rstrip()

    return file_m.group(1).strip(), search_m.group(1).rstrip("\n"), replace_text


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
    """Find SEARCH in original, substitute with REPLACE.  Returns None on failure."""
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
def coder_node(state: AgentState) -> dict:
    """Generate a code fix via search-and-replace, then produce a unified diff."""
    print("[Coder] Generating code fix...")

    issue_text = state.get("issue", "(no issue)")
    file_context = state.get("file_context", [])
    errors = state.get("errors", "")

    file_info = _extract_file_info(file_context)
    known_paths = list(file_info.keys())
    print(f"[Coder] Context files: {known_paths}")

    # Build clean context (line numbers stripped for less noise)
    ctx_parts = []
    for p, c in file_info.items():
        lines = c.split("\n")
        if len(lines) > 300:
            c = "\n".join(lines[:300]) + "\n[TRUNCATED]"
        ctx_parts.append(f"# === FILE: {p} ===\n{c}")
    ctx = "\n\n".join(ctx_parts) or "(no source code available)"

    parts = [f"## GitHub Issue\n{issue_text}", f"## Source Code\n{ctx}"]
    if errors:
        parts.append(f"## Previous Errors (your last fix failed)\n{errors}")
    parts.append("Provide your fix using the FILE/SEARCH/REPLACE format.")

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(parts)),
    ]
    history: list[dict] = []

    print("[Coder] Calling LLM...")
    try:
        resp = _llm.invoke(messages)
    except Exception as exc:
        print(f"[Coder] [ERROR] LLM failed: {exc}")
        history.extend(append_to_history("Coder", "Error", str(exc)))
        return {"proposed_fix": "", "history": history}

    raw = getattr(resp, "content", "") or ""
    if not raw:
        print("[Coder] [ERROR] Empty LLM response")
        history.extend(append_to_history("Coder", "Error", "Empty LLM response"))
        return {"proposed_fix": "", "history": history}

    print(f"[Coder] LLM returned {len(raw)} chars")
    history.extend(append_to_history("Coder", "Generation", raw, max_length=800))

    plan = _extract_plan(raw)
    if plan:
        print(f"[Coder] Plan: {plan}")

    # --- Primary: search-and-replace -> programmatic diff ---
    diff = ""
    fpath, search, replace = _parse_fix(raw)

    if fpath and search and replace:
        matched = _match_path(fpath, known_paths)
        print(f"[Coder] Fix target: {fpath} -> {matched}")

        if matched in file_info:
            original = file_info[matched]
            modified = _find_and_replace(original, search, replace)
            if modified and modified != original:
                diff = _make_diff(original, modified, matched)
                print(f"[Coder] [OK] Generated diff ({len(diff)} chars)")
            elif modified == original:
                print("[Coder] [WARN] Fix is a no-op (SEARCH == REPLACE)")
            else:
                print("[Coder] [WARN] SEARCH block not found in source file")
        else:
            print(f"[Coder] [WARN] File '{matched}' not in context. Known: {known_paths}")
    else:
        missing = []
        if not fpath:
            missing.append("FILE")
        if not search:
            missing.append("SEARCH")
        if not replace:
            missing.append("REPLACE")
        print(f"[Coder] [WARN] Could not parse fix format (missing: {', '.join(missing)})")

    # --- Fallback: extract raw unified diff from fenced block ---
    if not diff:
        print("[Coder] Trying fallback: raw diff extraction")
        fb = _extract_diff_fallback(raw)
        if fb and "---" in fb and "+++" in fb:
            lines = fb.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("--- a/") or line.startswith("+++ b/"):
                    lines[i] = line[:6] + _match_path(line[6:], known_paths)
            diff = "\n".join(lines)
            print(f"[Coder] [OK] Fallback diff ({len(diff)} chars)")
        else:
            print("[Coder] [ERROR] No usable fix in LLM output")

    if diff:
        print(f"[Coder] Final diff preview:\n{diff[:400]}")

    return {"plan": plan, "proposed_fix": diff, "history": history}