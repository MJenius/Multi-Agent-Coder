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
relevant to the issue using a Map-Reduce strategy.  

You have the following tools:
  generate_repo_map(directory)      - Get a high-level tree view of the repo + README.
  get_symbol_definition(symbol, dir) - Find where a function or class is defined.
  list_files(directory)             - Recursively list code files.
  search_code(query, directory)     - Grep for a string across all code files.
  read_file(file_path)              - Read a file (truncated at 500 lines).

Map-Reduce Strategy:
1. MAP phase: ALWAYS call `generate_repo_map('.')` FIRST to understand the structure.
2. TARGET phase: Based on the map and the issue, look for Integration Points.
   - Use `get_symbol_definition` if you know exactly what is crashing.
   - Or use `search_code` to grep for relevant strings.
3. REDUCE (READ) phase: Call `read_file` to collect context on the most relevant files.

CRITICAL CONSTRAINTS:
- NEVER use 'list_files' on a specific file. 
- To see what is inside a file, you MUST use 'read_file'.
- Do NOT read more than 3 files total.
- Prefer targeted reads over reading everything.
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
    history_additions.extend(append_to_history("Researcher", "Started Exploration", f"Repo: {repo_path}\nIssue: {issue_text}"))

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

    history_additions.extend(append_to_history("Researcher", "Finished Exploration", f"Collected {len(snippets)} snippets. Read {files_read} files."))

    return {
        "file_context": snippets,
        "history": history_additions
    }
