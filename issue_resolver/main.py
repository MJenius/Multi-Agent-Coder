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

    # Print the final state summary
    if final_state:
        # Get the last node's output
        last_node = list(final_state.keys())[-1]
        last_output = final_state[last_node]
        print(f"\n[RESULT] Final state snapshot (from '{last_node}'):")
        for key, value in last_output.items():
            if isinstance(value, list):
                print(f"  {key}: [{len(value)} item(s)]")
                # Show file_context snippets
                if key == "file_context":
                    for i, snippet in enumerate(value, 1):
                        preview = snippet[:200] + "..." if len(snippet) > 200 else snippet
                        print(f"    [{i}] {preview}")
            elif isinstance(value, str) and len(value) > 80:
                print(f"  {key}: {value[:80]}...")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
