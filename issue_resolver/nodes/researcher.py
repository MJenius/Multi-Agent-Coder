"""
Researcher Node -- Phase 2: LLM-driven codebase exploration.

Uses llama3.2 via ChatOllama with tool-binding to:
  1. List .py files in the target repository.
  2. Search for relevant function / class names mentioned in the issue.
  3. Read the most relevant files (max 3 files, <=500 lines each).
  4. Populate file_context with discovered code snippets.
  
Phase 1 Improvements:
  - Detects direct file hints (e.g., "🎯 HINT: QRCoder/AsciiQRCode.cs")
  - Detects repository language (C#, Python, Node.js, Java)
  - Implements fallback search strategy for 0-result queries
  - Timeouts on all tool calls to prevent hangs
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.config import OLLAMA_BASE_URL, RESEARCHER_MODEL
from issue_resolver.tools import (
    REPO_TOOLS, 
    list_files, 
    search_code, 
    read_file,
    generate_repo_map,
    get_symbol_definition
)


# ---------------------------------------------------------------------------
# LLM setup -- tool-augmented model
# ---------------------------------------------------------------------------
_base_llm = ChatOllama(
    model=RESEARCHER_MODEL,
    temperature=0,
    base_url=OLLAMA_BASE_URL,
)

_llm = _base_llm.bind_tools(REPO_TOOLS)

# Map tool names --> callables for dispatching
_TOOL_MAP = {
    "list_files": list_files,
    "search_code": search_code,
    "read_file": read_file,
    "generate_repo_map": generate_repo_map,
    "get_symbol_definition": get_symbol_definition,
}

_SYSTEM_PROMPT = """\
You are the Researcher agent. Find the source code relevant to a GitHub issue FAST.

Available tools:
  read_file(file_path)               - Read a file (truncated at 500 lines). USE THIS FIRST if you know the file.
  search_code(query, directory)      - Grep for a string across code files.
  get_symbol_definition(symbol, dir) - Find where a function/class is defined.
  generate_repo_map(directory)       - Get a tree view of the repo structure. ONLY if you don't know where to look.
  list_files(directory)              - List code files in a specific folder.

SPEED RULES (CRITICAL):
──────────────────────
1. If the issue mentions a SPECIFIC FILE PATH → call read_file() IMMEDIATELY. Do NOT map first.
2. If the issue has a HINT (🎯) → follow the hint directly with read_file().
3. Only call generate_repo_map() if you have NO idea where the relevant code is.
4. Target searches to SPECIFIC folders (e.g., './QRCoder', './src'), never search root '.'.
5. Maximum 3 files read. Stop as soon as you have the relevant code.
6. After reading the target file, STOP. Do not explore further unless the code is clearly wrong file.

CONSTRAINTS:
- NEVER read more than 3 files total.
- NEVER use list_files on root directory for large repos.
- Prefer search_code with specific folder paths over broad searches.
- When done, simply state what you found. No need for additional exploration.
"""


# ---------------------------------------------------------------------------
# Constants -- memory guards
# ---------------------------------------------------------------------------
_MAX_TOOL_ROUNDS = 5   # max LLM <-> tool iterations
_MAX_FILES_READ = 3    # cap on read_file calls
_MAX_TOTAL_LINES = 500  # soft cap across all files read


# ---------------------------------------------------------------------------
# PHASE 1 IMPROVEMENTS: Helper Functions
# ---------------------------------------------------------------------------

# Supported code file extensions for hint extraction
_CODE_EXTENSIONS = r'(?:cs|py|js|ts|tsx|xaml|java|go|cpp|h|jsx|csproj|sln|rb|rs|swift|kt)'

def _extract_hints_from_issue(issue_text: str) -> list[str]:
    """Extract ONLY file paths from issue description. Never returns sentences.
    
    Strategy: Find all substrings that look like file paths (e.g., QRCoder/AsciiQRCode.cs)
    and return just those paths. Sentences containing paths are NOT returned.
    
    Args:
        issue_text: GitHub issue body text.
    
    Returns:
        List of file paths found (relative paths like "QRCoder/AsciiQRCode.cs").
    """
    # Single robust pattern: match file paths with optional directory components
    # Matches: "QRCoder/AsciiQRCode.cs", "src/main.py", "Button.tsx", "./utils/helper.js"
    # Does NOT match full sentences or prose text
    path_pattern = rf'(?:\./)?([A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-.]+)*\.(?:{_CODE_EXTENSIONS}))'
    all_matches = re.findall(path_pattern, issue_text)
    
    # Deduplicate while preserving order, normalize leading ./
    seen = set()
    unique_paths = []
    for match in all_matches:
        normalized = match.lstrip('./')
        if normalized not in seen:
            unique_paths.append(normalized)
            seen.add(normalized)
    
    # Prefer full paths (with /) over bare filenames
    # If we have both "AsciiQRCode.cs" and "QRCoder/AsciiQRCode.cs", keep only the full path
    final_hints = []
    for path in unique_paths:
        if '/' not in path:
            # Bare filename -- only keep if no full path version exists
            has_full_path = any(path in p and '/' in p for p in unique_paths)
            if not has_full_path:
                final_hints.append(path)
        else:
            final_hints.append(path)
    
    return final_hints


def _detect_language(repo_path: str) -> str:
    """Detect primary programming language of the repository (fast, no deep scan).
    
    Checks only for known marker files and top-level/immediate child extensions.
    Never uses recursive globs (those can hang on 2500+ file repos).
    
    Args:
        repo_path: Path to repository root.
    
    Returns:
        Language identifier: "csharp", "python", "nodejs", "java", "unknown"
    """
    repo = Path(repo_path).resolve()
    
    # Fast check: known marker FILES (no recursive glob needed)
    marker_files = {
        "csharp": ["*.sln", "*.csproj"],
        "python": ["setup.py", "requirements.txt", "pyproject.toml", "setup.cfg"],
        "nodejs": ["package.json", "package-lock.json"],
        "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    }
    
    for lang, patterns in marker_files.items():
        for pattern in patterns:
            if "*" in pattern:
                # Glob only in root dir (non-recursive)
                if list(repo.glob(pattern)):
                    return lang
            else:
                if (repo / pattern).exists():
                    return lang
    
    # Quick fallback: check root + 1 level deep for common extensions
    ext_counts = {"csharp": 0, "python": 0, "nodejs": 0, "java": 0}
    ext_map = {".cs": "csharp", ".py": "python", ".js": "nodejs", ".ts": "nodejs", ".java": "java"}
    
    try:
        for item in repo.iterdir():
            if item.is_file() and item.suffix in ext_map:
                ext_counts[ext_map[item.suffix]] += 1
            elif item.is_dir() and item.name not in {'bin', 'obj', '.git', 'node_modules', '__pycache__', '.vs'}:
                try:
                    for child in item.iterdir():
                        if child.is_file() and child.suffix in ext_map:
                            ext_counts[ext_map[child.suffix]] += 1
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError):
        pass
    
    max_lang = max(ext_counts, key=ext_counts.get)
    return max_lang if ext_counts[max_lang] > 0 else "unknown"


def _try_search_variations(
    base_query: str,
    directory: str,
    language: str = "unknown",
    max_attempts: int = 3
) -> tuple[str, bool]:
    """Try progressively broader searches if base query fails.
    
    Implements fallback strategy:
      1. Original query: "ASCIIQRCode"
      2. CamelCase → snake_case: "ascii_qr_code"
      3. Partial tokens: "ASCII", "QRCode", etc.
    
    Args:
        base_query: Original search term.
        directory: Directory to search in.
        language: Detected language ("csharp", "python", etc.) for smarter fallbacks.
        max_attempts: Maximum number of search variations to try.
    
    Returns:
        Tuple of (result_text, found_matches). found_matches=True if matches > 0.
    """
    queries = [base_query]
    
    # Add snake_case variant
    snake_case = re.sub(r'([A-Z])', r'_\1', base_query).lower().lstrip('_')
    if snake_case != base_query.lower() and snake_case not in queries:
        queries.append(snake_case)
    
    # Add lowercase variant
    if base_query.lower() not in queries:
        queries.append(base_query.lower())
    
    # Add partial tokens (split on CamelCase or underscores)
    tokens = re.split(r'([A-Z][a-z]+|[a-z]+|_)', base_query)
    tokens = [t for t in tokens if t and t != '_']
    for token in tokens[:max_attempts]:
        if token.lower() not in queries:
            queries.append(token.lower())
    
    print(f"[Researcher] Search variations for '{base_query}': {queries[:max_attempts]}")
    
    for attempt, query in enumerate(queries[:max_attempts], 1):
        try:
            result = search_code.invoke({"query": query, "directory": directory})
            match_count = len([l for l in result.split('\n') if l.strip() and ':' in l and not l.startswith('[')])
            
            if match_count > 0:
                print(f"[Researcher] ✅ Found {match_count} match(es) for '{query}' (attempt {attempt})")
                return result, True
            else:
                print(f"[Researcher] ❌ No matches for '{query}' (attempt {attempt})")
        except Exception as e:
            print(f"[Researcher] Error searching for '{query}': {e}")
    
    return f"No matches found after {len(queries)} search variations of '{base_query}'.", False


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------
def researcher_node(state: AgentState) -> dict:
    """Run the Researcher agent: analyse the issue, call tools, return context."""
    print("[Researcher] Starting codebase exploration...")

    repo_path = state.get("repo_path", ".")
    issue_text = state.get("issue", "(no issue provided)")
    errors = state.get("errors", "")

    human_str = f"GitHub Issue:\n{issue_text}\n\nRepository path: {repo_path}\n\n"
    if errors:
        human_str += f"Supervisor Feedback/Errors:\n{errors}\n\n"
    human_str += "Please explore the repository and find the relevant code."

    # Seed the conversation with system prompt + issue
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_str),
    ]

    snippets: list[str] = list(state.get("file_context", []))
    files_read = 0
    total_lines = 0

    # Initialize history for this round
    history_additions: list[dict] = []
    
    # Store initial issue into history
    history_additions.extend(append_to_history("Researcher", "Strategic Targeting", f"Repo: {repo_path}\nIssue: {issue_text}"))

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: PRE-SCAN -- Detect language and extract hints
    # ─────────────────────────────────────────────────────────────────────────
    language = _detect_language(repo_path)
    print(f"[Researcher] Detected language: {language}")
    history_additions.extend(append_to_history("Researcher", "Language Detection", f"Language: {language}"))
    
    hint_files = _extract_hints_from_issue(issue_text)
    if hint_files:
        print(f"[Researcher] Found {len(hint_files)} direct hint(s): {hint_files}")
        history_additions.extend(append_to_history("Researcher", "Hint Extraction", f"Hints: {hint_files}"))
        
        # Read hinted files immediately (BEFORE LLM tool loop)
        print(f"[Researcher] Reading {len(hint_files)} hint file(s)...")
        for idx, hint_file in enumerate(hint_files[:_MAX_FILES_READ], 1):
            if files_read >= _MAX_FILES_READ:
                print(f"[Researcher] Reached max file limit; skipping remaining hints.")
                break
            
            # Normalize path: strip leading repo path if hint includes it
            normalized_hint = hint_file.lstrip('./')
            repo_name = Path(repo_path).name
            if normalized_hint.startswith(repo_name + '/'):
                # Hint includes repo folder (e.g., "issue_resolver/graph.py")
                # Strip it since repo_path already points there
                normalized_hint = normalized_hint[len(repo_name) + 1:]
            
            # Build absolute path
            safe_path_resolved = (Path(repo_path) / normalized_hint).resolve()
            
            try:
                result = read_file.invoke({"file_path": str(safe_path_resolved)})
                if result.startswith("Error"):
                    print(f"[Researcher]    ❌ Failed to read {hint_file}: {result[:100]}")
                    history_additions.extend(
                        append_to_history("Researcher", "Hint Read", f"Failed: {hint_file} ({result[:80]})")
                    )
                else:
                    lines_in_file = result.count("\n")
                    print(f"[Researcher]    ✅ Read {lines_in_file} lines from {hint_file}")
                    
                    # Store as a context snippet
                    snippet = f"# --- [HINTED] file: {hint_file} ---\n{result}"
                    snippets.append(snippet)
                    files_read += 1
                    total_lines += lines_in_file + 1
                    
                    history_additions.extend(
                        append_to_history("Researcher", "Hint Read", f"✅ {hint_file} ({lines_in_file} lines)")
                    )
            except Exception as e:
                print(f"[Researcher]    ❌ Exception reading {hint_file}: {e}")
                history_additions.extend(
                    append_to_history("Researcher", "Hint Read", f"Exception: {str(e)[:80]}")
                )
    
    # Early exit if hints provided ANY relevant content
    # Key insight: if we have hint files with actual code, the LLM loop just wastes time
    # calling generate_repo_map and re-discovering what we already have
    if snippets and files_read >= 1:
        print(f"[Researcher] ✅ Hints provided {files_read} file(s), {total_lines} lines. Skipping LLM search.")
        print(f"[Researcher] Done -- collected {len(snippets)} snippet(s), "
              f"{files_read} file(s) read, ~{total_lines} lines.")
        history_additions.extend(
            append_to_history("Researcher", "Targeting Complete", f"Collected {len(snippets)} snippets (from hints). Read {files_read} files.")
        )
        return {
            "file_context": snippets,
            "history": history_additions
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: LLM-DRIVEN SEARCH (Fall back to this only if hints didn't work)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Reduce rounds when using LLM fallback -- we need speed
    max_rounds = 3  # Enough for: repo_map → search → read_file
    
    # Inject context about what we already know (from hints that partially worked)
    if snippets:
        context_note = (
            f"\n\nIMPORTANT: The system already read {files_read} file(s) from hints "
            f"({total_lines} lines). These files are already in context. "
            f"Focus ONLY on finding additional relevant files, do NOT re-read files you already have. "
            f"You need at most {_MAX_FILES_READ - files_read} more file(s)."
        )
        messages[-1] = HumanMessage(content=messages[-1].content + context_note)
    
    for round_num in range(1, max_rounds + 1):
        print(f"[Researcher]  |-- Round {round_num}/{max_rounds}")

        # ── Ask the LLM ────────────────────────────────────────────
        try:
            response = _llm.invoke(messages)
        except Exception as exc:
            print(f"[Researcher] [ERROR] LLM call failed: {exc}")
            break

        messages.append(response)

        # ── If there are no tool calls, the agent is done ──────────
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            print("[Researcher] No more tool calls -- wrapping up.")
            break

        # ── Execute each tool call ─────────────────────────────────
        for tc in tool_calls:
            fn_name = tc["name"]
            fn_args = tc["args"]
            call_id = tc["id"]

            print(f"[Researcher]    --> {fn_name}({fn_args})")
            
            # Formulate the payload - only store filename and brief summary in state
            # but sys_logger inside log handles the printing. 
            log_payload = f"Tool: {fn_name}\nArgs: {json.dumps(fn_args)}"
            history_additions.extend(append_to_history("Researcher", "Tool Call", log_payload, max_length=150))

            # Enforce memory guards for read_file
            if fn_name == "read_file":
                if files_read >= _MAX_FILES_READ:
                    result = (
                        f"[BLOCKED] Already read {_MAX_FILES_READ} files -- "
                        "limit reached. Please summarise with what you have."
                    )
                    messages.append(ToolMessage(content=result, tool_call_id=call_id))
                    continue
                if total_lines >= _MAX_TOTAL_LINES:
                    result = (
                        f"[BLOCKED] Already read {total_lines} lines -- "
                        "line budget exhausted."
                    )
                    messages.append(ToolMessage(content=result, tool_call_id=call_id))
                    continue

            # Dispatch the tool
            tool_fn = _TOOL_MAP.get(fn_name)
            if tool_fn is None:
                result = f"Unknown tool '{fn_name}'."
            else:
                try:
                    result = tool_fn.invoke(fn_args)
                except Exception as exc:
                    result = f"Tool error: {exc}"

            # Track read_file usage
            if fn_name == "read_file" and not result.startswith(("Error", "[BLOCKED")):
                files_read += 1
                total_lines += result.count("\n") + 1
                # Store as a context snippet
                file_label = fn_args.get("file_path", "unknown")
                snippet = f"# --- file: {file_label} ---\n{result}"
                snippets.append(snippet)

            messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    print(f"[Researcher] Done -- collected {len(snippets)} snippet(s), "
          f"{files_read} file(s) read, ~{total_lines} lines.")

    history_additions.extend(append_to_history("Researcher", "Targeting Complete", f"Collected {len(snippets)} snippets. Read {files_read} files."))

    return {
        "file_context": snippets,
        "history": history_additions
    }
