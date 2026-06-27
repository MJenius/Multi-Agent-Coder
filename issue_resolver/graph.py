from __future__ import annotations

from langgraph.graph import StateGraph, END

from issue_resolver.state import AgentState
from issue_resolver.nodes import (
    setup_node,
    supervisor_node,
    researcher_node,
    planner_node,
    testgen_node,
    test_validator_node,
    coder_node,
    reviewer_node,
    failure_handler_node,
)


def _route_supervisor(state: AgentState) -> str:
    next_step = state.get("next_step", "end")
    valid_routes = {
        "researcher",
        "planner",
        "test_generator",
        "test_validator",
        "coder",
        "failure_handler",
    }
    if next_step in valid_routes:
        return next_step
    return "end"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("setup", setup_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("planner", planner_node)
    graph.add_node("test_generator", testgen_node)
    graph.add_node("test_validator", test_validator_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("failure_handler", failure_handler_node)

    graph.set_entry_point("setup")
    graph.add_edge("setup", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "researcher": "researcher",
            "planner": "planner",
            "test_generator": "test_generator",
            "test_validator": "test_validator",
            "coder": "coder",
            "failure_handler": "failure_handler",
            "end": END,
        },
    )

    graph.add_edge("researcher", "supervisor")
    graph.add_edge("planner", "supervisor")
    graph.add_edge("test_generator", "supervisor")
    graph.add_edge("test_validator", "supervisor")
    graph.add_edge("coder", "reviewer")
    graph.add_edge("reviewer", "supervisor")
    graph.add_edge("failure_handler", END)

    compiled = graph.compile()
    print("[OK] Graph compiled successfully!")
    return compiled


app = build_graph()
