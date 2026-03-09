"""
Researcher Node -- Phase 2: LLM-driven codebase exploration.

Uses llama3.2 via ChatOllama with tool-binding to:
  1. List .py files in the target repository.
  2. Search for relevant function / class names mentioned in the issue.
  3. Read the most relevant files (max 3 files, <=500 lines each).
  4. Populate file_context with discovered code snippets.
"""

from __future__ import annotations

import json

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
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
    model="llama3.2:latest",
    temperature=0,
    base_url="http://localhost:11434",
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
You are the Researcher agent in a multi-agent system that resolves GitHub issues.

Your job is to explore a LOCAL code repository and find the source code
relevant to the issue using a SURGICAL, MAP-REDUCE STRATEGY called "Strategic Targeting".

You have the following tools:
  generate_repo_map(directory, max_depth=2)      - Get a HIGH-LEVEL tree view of the repo structure.
  get_symbol_definition(symbol, dir)             - Find where a function or class is defined.
  list_files(directory)                          - Recursively list code files (for specific folders).
  search_code(query, directory)                  - Grep for a string across all code files.
  read_file(file_path)                           - Read a file (truncated at 500 lines).

STRATEGIC TARGETING APPROACH (3 Phases):
────────────────────────────────────────────────

PHASE 1: MAP THE STRUCTURE (5 seconds)
1. Call `generate_repo_map('.')` with default max_depth=2 FIRST.
   - This shows only 2 levels deep: top-level folders and immediate children.
   - Skip overwhelming detail; focus on folder layout.

PHASE 2: IDENTIFY THE TARGET (Surgical Focus)
2. Based on the map and the issue, identify the MOST LIKELY folder.
   - If you see 'QRCoder', 'src', 'handlers', 'models', etc., these are targets.
   - For C# projects (like QRCoder), expect: QRCoder/*.cs files
   - If the issue mentions "ASCII" and "QRCode", the bug is likely in AsciiQRCode.cs
   - Use `get_symbol_definition(symbol, dir)` if you know the exact class/function.
   - Or use `search_code(query, dir)` to grep ONLY in the target folder (not root).

PHASE 3: DRILL DOWN TO SOURCE (Precision Search)
3. Once you identify a promising folder:
   - Call `list_files('folder_name')` to see all code files in that SPECIFIC folder.
   - Focus ONLY on the most relevant files (e.g., AsciiQRCode.cs for ASCII-related bugs).
   - Then use `read_file` to examine the top 3 most relevant files.

CRITICAL CONSTRAINTS & RULES:
──────────────────────────────
- NEVER use `list_files` on the root directory if the repository has >50 files.
- ALWAYS start with `generate_repo_map('.')` first.
- Target your searches to SPECIFIC folders, not the root directory.
- NEVER read more than 3 files total.
- SPEED: The goal is to find the bug in under 3 tool calls if possible.
- When calling `list_files`, prefer specific subdirectories like './QRCoder' or './src' over '.'.
- If the issue mentions a specific file (e.g., AsciiQRCode.cs), search for it directly and read it ASAP.

IMMEDIATE REFLEX FOR DIRECT FILE PATHS (CRITICAL):
───────────────────────────────────────────────────
IF the issue description contains or mentions a specific relative file path:
  - Example paths: './QRCoder/AsciiQRCode.cs', 'src/handlers/auth.py', 'components/Button.tsx'
  - AS SOON AS YOU SEE THE PATH (even while mapping), call `read_file()` on it immediately.
  - You do NOT need to wait for the full map; direct paths are golden hints.
  - Pattern: If the issue says "bug is in X.cs" or "look at Y.py", that's a direct file hint.
  
  HOW TO DETECT A DIRECT FILE HINT:
    1. Look for mentions of files with extensions (.cs, .py, .ts, .js, .xaml, .tsx, etc.)
    2. Look for paths with slashes or backslashes (component/name.ext or component\\name.ext)
    3. Non-exhaustive examples of direct hints: "QRCoder/AsciiQRCode.cs", "handlers/request.py", "./utils/math.ts"

HINT REFLEX (Critical Optimization):
────────────────────────────────────
IF A 'HINT' OR '🎯' MARKER IS PROVIDED in the issue description:
  1. IMMEDIATELY call `read_file` on that file path AFTER generating the repo map.
  2. DO NOT waste time mapping the entire project if you already have a lead.
  3. HINT markers to watch for (case-insensitive):
     - "HINT:" (followed by a file path or description)
     - "🎯 TARGET:" (direct target)
     - "🎯 FOCUS:" (where to focus)
     - "bug likely in" or "bug is in" (strong directional hint)
     - "check this file" or "look at this file"
  4. Example: If the hint says "Focus on QRCoder/AsciiQRCode.cs", call:
     → generate_repo_map('.')        [quick overview]
     → read_file('./sandbox_workspace/QRCoder/AsciiQRCode.cs')  [read target immediately]
  5. This reflexive targeting saves tokens and gets you to the code 5x faster.

REFUSE EMPTY RESULTS (Fallback Strategy):
──────────────────────────────────────────
DO NOT report "Collected 0 snippets" or "nothing found" if initial searches fail. Instead:

  A) If `search_code(query, directory)` returns ZERO matches for a logic block:
     - DO NOT assume the code doesn't exist; try different file extensions.
     - FALLBACK PATTERN for different languages:
       * Looking for a class/function in C# but found nothing in .cs files?
         → Try searching again in .xaml files (UI markup, could contain code-behind references)
         → Try .csproj files (project configuration, might contain build logic)
       * Looking in Python but found nothing in .py files?
         → Try .pyx (Cython) or .pxd (Cython declarations)
       * Looking in JavaScript/TypeScript?
         → Try both .ts and .tsx (TypeScript React), .js and .jsx (JavaScript React)
     - USE `list_files()` on target folder to see ALL files, then pick the most relevant.
  
  B) If `generate_repo_map` or any tool produces minimal output:
     - The tool did NOT fail; the ignore filters are WORKING to skip build artifacts.
     - DO NOT interpret small output as "nothing found" — it means the repo is clean.
     - Proceed to search SPECIFIC target folders with `search_code()`.

  C) Explicitly try multiple search variations:
     - Original query: "AsciiQRCode"
     - Fallback queries: "AsciiQR", "Ascii", "QRCode", "qr_code" (snake_case variant)
     - Different folders: Try QRCoder/, then utils/, then root.

  D) If still stuck after 2 search attempts:
     - Call `list_files('target_folder')` to visually scan for likely files.
     - Manually skim through file names to find probable suspects.
     - Do NOT give up on 0 matches — it usually means your search query was too specific.
"""


# ---------------------------------------------------------------------------
# Constants -- memory guards
# ---------------------------------------------------------------------------
_MAX_TOOL_ROUNDS = 5   # max LLM <-> tool iterations
_MAX_FILES_READ = 3    # cap on read_file calls
_MAX_TOTAL_LINES = 500  # soft cap across all files read


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

    for round_num in range(1, _MAX_TOOL_ROUNDS + 1):
        print(f"[Researcher]  |-- Round {round_num}/{_MAX_TOOL_ROUNDS}")

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
