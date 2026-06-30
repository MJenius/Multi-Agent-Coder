"""Debugger Node — performs root cause analysis on verification failures.

Runs when checks fail, inspects traces and knowledge graph calls deterministically,
parses errors, and enriches context to instruct the next coding attempts.
"""

from __future__ import annotations

import json
import re
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.llm_utils import invoke_with_role_fallback
from issue_resolver.core.prompt_registry import get_prompt_registry
import issue_resolver.runtime_context as runtime_context


# Register prompt template on import
_DEFAULT_PROMPT = """\
You are an expert Debugger. You are given a repository issue description, a proposed patch that failed verification, the compilation/lint/test execution error messages, and a structured diagnostics report derived from the repository graph and LSP.

Your job is to analyze the root cause of the failure and output a bug diagnostics report.
Analyze specifically why the patch failed: check if there was a SEARCH block mismatch (search block not found/matched), a compilation/syntax error, a linter violation, or a test failure.

You must output a single JSON block wrapped in a ```json markdown block:
{
  "failure_type": "One of: search_mismatch, compilation_error, linter_error, test_failure, wrong_file, other",
  "root_cause": "Detailed explanation of why the proposed patch failed",
  "knowledge_graph_findings": "Related functions or classes that might be connected",
  "strategy_adjustments": "Specific adjustments recommended for the next coding attempt"
}
"""

get_prompt_registry().register("debugger", "1.0", _DEFAULT_PROMPT)


def _parse_errors(errors_text: str) -> list[dict[str, Any]]:
    """Parse traceback or compiler error lines to extract files, lines, and functions."""
    matches = []
    
    # Python traceback pattern: File "path/to/file.py", line 123, in func_name
    tb_pattern = re.compile(
        r'File\s+"([^"]+\.(?:py|js|ts|go|rs|java|cpp|c|h|jsx|tsx|vue|rb))",\s+line\s+(\d+)(?:,\s+in\s+(\w+))?',
        re.IGNORECASE
    )
    for m in tb_pattern.finditer(errors_text):
        matches.append({
            "file": m.group(1).replace("\\", "/"),
            "line": int(m.group(2)),
            "function": m.group(3) or ""
        })
        
    # Compiler / Linter warning format: filename.py:123: or filename.py:123:45:
    compile_pattern = re.compile(
        r'(?:^|\n)([^:\s\n]+\.(?:py|js|ts|go|rs|java|cpp|c|h|jsx|tsx|vue|rb)):(\d+)(?::(\d+))?:?',
        re.IGNORECASE
    )
    for m in compile_pattern.finditer(errors_text):
        file_path = m.group(1).replace("\\", "/")
        line_num = int(m.group(2))
        # Avoid duplicates
        if not any(match["file"] == file_path and match["line"] == line_num for match in matches):
            matches.append({
                "file": file_path,
                "line": line_num,
                "function": ""
            })
            
    # Generic exception message extraction: e.g. "TypeError: ..."
    exception_pattern = re.compile(r'(\b\w+(?:Error|Exception|Warning|Fault)\b:.*)', re.IGNORECASE)
    msg_match = exception_pattern.search(errors_text)
    if msg_match:
        for match in matches:
            match["exception_message"] = msg_match.group(1).strip()
            
    return matches


def debugger_node(state: AgentState) -> dict:
    """Diagnose build/test failure using deterministic parsing, graph lookups, and LLM reasoning."""
    print("[Debugger] Diagnosing validation failure...")
    issue = state.get("issue", "")
    proposed_fix = state.get("proposed_fix", "")
    errors = state.get("errors", "")
    
    # 1. Parse errors deterministically
    parsed_errors = _parse_errors(errors)
    
    graph = runtime_context.get_knowledge_graph()
    lsp_bridge = runtime_context.get_lsp_bridge()
    lsp_available = lsp_bridge is not None and lsp_bridge.is_available

    # 2. Query knowledge graph for symbols related to parsed error locations
    graph_context_parts = []
    
    if graph and parsed_errors:
        for match in parsed_errors:
            file_path = match["file"]
            line_num = match["line"]
            norm_path = file_path.replace("\\", "/").lstrip("./")
            
            # Find function/class covering this line in the graph
            matched_sym = None
            for fn in graph.functions.values():
                fn_norm_path = fn.file_path.replace("\\", "/").lstrip("./")
                if fn_norm_path == norm_path or norm_path.endswith(fn_norm_path):
                    if fn.line_number <= line_num <= fn.end_line:
                        matched_sym = fn
                        match["function"] = fn.name
                        break
            
            if matched_sym:
                symbol_name = matched_sym.name
                callers = graph.find_callers_structured(symbol_name)
                tests = graph.get_tests_for(matched_sym.file_path)
                
                info = (
                    f"- Error at `{norm_path}` line {line_num} in function `{symbol_name}`.\n"
                    f"  - Callers: {', '.join(c['name'] for c in callers[:3]) or 'none'}\n"
                    f"  - Tests for file: {', '.join(tests[:2]) or 'none'}"
                )
                
                # Fetch LSP references if available
                if lsp_available:
                    try:
                        from issue_resolver.intelligence.lsp_tools import lsp_find_references
                        refs = lsp_find_references(symbol_name, matched_sym.file_path, matched_sym.line_number)
                        if refs:
                            info += f"\n  - LSP References: {', '.join(f'{r.get(chr(102), chr(105))}:{r.get(chr(108), 0)}' for r in refs[:3])}"
                    except Exception:
                        pass
                
                graph_context_parts.append(info)
            else:
                graph_context_parts.append(f"- Error at `{norm_path}` line {line_num} (no enclosing symbol found in graph).")

    # 3. Compute blast radius
    blast_radius = []
    if graph:
        structured_plan = state.get("structured_plan", {})
        files_to_edit = structured_plan.get("files_to_edit", [])
        if files_to_edit:
            blast_radius = graph.get_affected_modules(files_to_edit)

    graph_context = ""
    if graph_context_parts:
        graph_context += "### Repository Proximity Context:\n" + "\n".join(graph_context_parts)
    if blast_radius:
        graph_context += f"\n\n### Transitive Blast Radius of proposed files:\n- {', '.join(blast_radius[:10])}"

    debug_input = f"""Issue Description:
{issue}

Proposed Patch that Failed:
{proposed_fix}

Verification Errors:
{errors}

{graph_context}
"""

    prompt = get_prompt_registry().get("debugger")
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=debug_input),
    ]

    repo_path = state.get("repo_path", ".")

    try:
        resp, model_name = invoke_with_role_fallback(
            role="debugger",
            candidates=["deepseek-ai/deepseek-v4-flash"],
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        raw = getattr(resp, "content", "") or ""
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        json_str = m.group(1).strip() if m else raw.strip()
        diagnostics = json.loads(json_str)
    except Exception as exc:
        print(f"[Debugger] [ERROR] Diagnostics execution failed: {exc}")
        diagnostics = {
            "failure_type": "other",
            "root_cause": f"Verification error details: {errors}",
            "knowledge_graph_findings": "None",
            "strategy_adjustments": "Retry the implementation step with additional verification checks.",
        }

    # 4. Refresh/update file context with latest contents from disk
    import os
    from issue_resolver.tools.repo_tools import read_file
    
    updated_file_context = list(state.get("file_context", []))
    files_to_refresh = set()
    
    # Refresh target files from plan
    structured_plan = state.get("structured_plan", {})
    for f in structured_plan.get("files_to_edit", []):
        files_to_refresh.add(f.replace("\\", "/").lstrip("./"))
        
    # Refresh files with errors
    for err in parsed_errors:
        if "file" in err:
            files_to_refresh.add(err["file"].replace("\\", "/").lstrip("./"))
            
    # Refresh files with search block mismatches
    mismatched_files = re.findall(r"Block \d+ on (\S+) failed: Search block content not matched", errors)
    for f in mismatched_files:
        files_to_refresh.add(f.replace("\\", "/").lstrip("./"))
        
    for rel_path in files_to_refresh:
        file_abs = os.path.abspath(os.path.join(repo_path, rel_path))
        if os.path.exists(file_abs):
            content = read_file.invoke({"file_path": file_abs})
            if not content.startswith("Error"):
                snippet_header = f"=== File: {rel_path} ==="
                new_snippet = f"{snippet_header}\n{content}"
                
                # Update in context if already exists, else append
                found = False
                for idx, snip in enumerate(updated_file_context):
                    if snip.startswith(snippet_header):
                        updated_file_context[idx] = new_snippet
                        found = True
                        break
                if not found:
                    updated_file_context.append(new_snippet)

    # Record trace
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "debugging_completed",
            "Debugger",
            "Completed root cause diagnosis",
            details={
                "parsed_errors": parsed_errors,
                "blast_radius": blast_radius,
                "diagnostics": diagnostics,
            },
        )

    # Append recommendations to plan
    plan_adjusted = (
        f"{state.get('plan', '')}\n\n"
        f"## Debugger Diagnostics ({diagnostics.get('failure_type', 'unknown')}):\n"
        f"- Root Cause: {diagnostics.get('root_cause')}\n"
        f"- Recommended Adjustments: {diagnostics.get('strategy_adjustments')}"
    )

    print(f"[Debugger] Root cause diagnosed ({diagnostics.get('failure_type')}): {diagnostics.get('root_cause')[:200]}...")

    return {
        "plan": plan_adjusted,
        "file_context": updated_file_context,
        "history": append_to_history(
            "Debugger",
            "Diagnose",
            f"Diagnosed failure ({diagnostics.get('failure_type', 'unknown')}): {diagnostics.get('root_cause')[:150]}",
        ),
    }
