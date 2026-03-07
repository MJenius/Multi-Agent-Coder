"""
Phase 2 -- Smoke Test Runner

Feeds a sample GitHub issue into the compiled graph and
traces the Supervisor's routing decisions step by step.
The Researcher now uses real tools to explore the target repo.

Usage:
    python -m issue_resolver.main
"""

from __future__ import annotations

from issue_resolver.graph import build_graph
from issue_resolver.state import AgentState
from issue_resolver.utils.logger import get_token_estimate
import json


SAMPLE_ISSUE = """\
Title: calculate_total crashes on empty list

When `calculate_total([])` is called, it raises a TypeError because
`sum()` receives a generator over an empty list that has no `.price`
attribute.  Expected behaviour: return 0 for an empty list.

Steps to reproduce:
1. Call `calculate_total([])`
2. Observe: TypeError: 'NoneType' object has no attribute 'price'

Environment: Python 3.12, Ubuntu 24.04
"""


def main() -> None:
    print("=" * 60)
    print("  Multi-Agent Issue Resolver -- Phase 2 Smoke Test")
    print("=" * 60)

    # Build & compile the graph
    app = build_graph()

    # Prepare the initial state
    # Default repo path -- override with env var or CLI arg in the future
    repo_path = "./sandbox_workspace"

    initial_state: AgentState = {
        "issue": SAMPLE_ISSUE,
        "repo_path": repo_path,
        "file_context": [],
        "proposed_fix": "",
        "errors": "",
        "next_step": "",
        "iterations": 0,
        "history": [],
    }

    print(f"\n[REPO] Target repository: {repo_path}")

    print(f"\n[ISSUE]\n{SAMPLE_ISSUE}")
    print("-" * 60)

    # Stream the graph execution step by step for visualization and accumulation
    print("\n[START] Starting graph execution...\n")

    full_state = initial_state.copy()
    for step in app.stream(initial_state):
        # Each step in 'updates' mode is {node_name: state_update}
        for node_name, state_update in step.items():
            print(f"  -- Node '{node_name}' returned: {list(state_update.keys())}")
            # Manually merge updates into full_state (simplified LangGraph-like merge)
            for key, val in state_update.items():
                if key == "history":
                    full_state["history"] = full_state.get("history", []) + val
                elif key == "file_context":
                    # Simple merge for file context items (ensure uniqueness)
                    # Note: in a real LangGraph setup, the State would handle this via define_reducer
                    current_context = full_state.get("file_context", [])
                    for item in val:
                        if item not in current_context:
                            current_context.append(item)
                    full_state["file_context"] = current_context
                else:
                    full_state[key] = val
    final_state = full_state # Preserve for report

    print("\n" + "=" * 60)
    print("  [OK] Graph completed!")
    print("=" * 60)

    # Print the final state summary and generate resolution_report.json
    if final_state:
        print("\n[RESULT] Final state snapshot:")
        
        # State summary printing
        for key, value in final_state.items():
            if key == "history":
                continue # Skip console dumping full history
            if isinstance(value, list):
                print(f"  {key}: [{len(value)} item(s)]")
            elif isinstance(value, str) and len(value) > 80:
                print(f"  {key}: {value[:80]}...")
            else:
                print(f"  {key}: {value}")
        
        # Generate resolution_report.json
        print("\n[REPORT] Generating resolution_report.json...")
        
        history = final_state.get("history", [])
        total_iterations = final_state.get("iterations", 0)
        final_errors = final_state.get("errors", "")
        final_proposed_fix = final_state.get("proposed_fix", "")
        
        # Extract metadata from history
        files_read = []
        failed_diffs = []
        total_chars = 0
        
        for entry in history:
            node = entry.get("node")
            action = entry.get("action")
            content = entry.get("content", "")
            
            total_chars += len(content)
            
            if node == "Researcher" and action == "Tool Call":
                if '"read_file"' in content:
                    files_read.append(content)
                    
            if node == "Reviewer" and (action == "Apply Patch Failed" or action == "Test Execution"):
                if "Error" in content or "Traceback" in content or "FAIL" in content or "failed" in content.lower():
                    failed_diffs.append({
                        "node": node,
                        "action": action,
                        "traceback_snippet": content[:500]
                    })
        
        # Determine Success Path boolean
        # Success if it reached END and no errors remain
        is_resolved = (final_proposed_fix and not final_errors and final_state.get("next_step") == "end")
        
        report = {
            "is_resolved": is_resolved,
            "total_iterations": total_iterations,
            "total_character_estimate": total_chars,
            "total_token_estimate": get_token_estimate(str(total_chars)), 
            "files_read_summary": files_read,
            "failed_diffs_and_tracebacks": failed_diffs,
            "final_successful_diff": final_proposed_fix if is_resolved else None,
            "final_errors": final_errors if not is_resolved else None,
            "full_history_trace": history
        }
        
        with open("resolution_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        print("[OK] Exported resolution_report.json")


if __name__ == "__main__":
    main()
