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

    # Stream the graph execution step by step
    print("\n[START] Starting graph execution...\n")

    final_state = None
    for step in app.stream(initial_state):
        # Each step is a dict of {node_name: state_update}
        for node_name, state_update in step.items():
            print(f"  -- Node '{node_name}' returned: {list(state_update.keys())}")
        final_state = step

    print("\n" + "=" * 60)
    print("  [OK] Graph completed!")
    print("=" * 60)

    # Print the final state summary and generate resolution_report.json
    if final_state:
        # Get the last node's output
        last_node = list(final_state.keys())[-1]
        last_output = final_state[last_node]
        print(f"\n[RESULT] Final state snapshot (from '{last_node}'):")
        
        # State summary printing
        for key, value in last_output.items():
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
        
        history = last_output.get("history", [])
        total_iterations = last_output.get("iterations", 0)
        final_errors = last_output.get("errors", "")
        final_proposed_fix = last_output.get("proposed_fix", "")
        
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
                    
            if node == "Reviewer" and action == "Apply Patch Failed" or action == "Test Execution":
                if "Error" in content or "Traceback" in content or "FAIL" in content or "failed" in content.lower():
                    failed_diffs.append({
                        "node": node,
                        "action": action,
                        "traceback_snippet": content[:500]
                    })
        
        # Determine Success Path boolean
        is_resolved = False
        if final_proposed_fix and not final_errors:
            is_resolved = True
        elif len(history) > 0 and history[-1].get("action") == "Failure Summary":
            is_resolved = False

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
