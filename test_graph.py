"""Minimal test to verify graph compilation and trace execution."""
import sys
from issue_resolver.graph import build_graph
from issue_resolver.state import AgentState

def main():
    app = build_graph()
    
    initial_state: AgentState = {
        "issue": "calculate_total crashes on empty list",
        "repo_path": "./sandbox_workspace",
        "file_context": [],
        "plan": "",
        "proposed_fix": "",
        "errors": "",
        "next_step": "",
        "iterations": 0,
        "is_resolved": False,
        "history": [],
    }
    
    print("Starting graph...")
    step_count = 0
    for step in app.stream(initial_state):
        step_count += 1
        for node_name, state_update in step.items():
            keys = list(state_update.keys())
            print(f"Step {step_count} - Node: {node_name} - Keys: {keys}")
            if "next_step" in state_update:
                print(f"  -> next_step = {state_update['next_step']}")
            if "iterations" in state_update:
                print(f"  -> iterations = {state_update['iterations']}")
    
    print(f"\nDone! Total steps: {step_count}")

if __name__ == "__main__":
    main()
