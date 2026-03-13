"""
Graph Builder -- Compiles the LangGraph StateGraph for the issue resolver.

Topology
--------
                 ┌──────────────┐
        ┌───────►│  Researcher  ├───────┐
        │        └──────────────┘       │
  ┌─────┴──────┐                  ┌─────▼──────┐
  │ Supervisor │◄─────────────────┤ Supervisor │  (re-enters after every agent)
  └─────┬──────┘                  └─────┬──────┘
        │        ┌──────────────┐       │
        ├───────►│   Planner    ├───────┘
        │        └──────────────┘
        │        ┌──────────────┐
        ├───────►│  TestGen     ├───────┐
        │        └──────────────┘       │
        │        ┌──────────────┐       │
        ├───────►│    Coder     ├───────┤
        │        └──────────────┘       │
        │        ┌──────────────┐       │
        └───────►│   Reviewer   ├───────┘
                 └──────────────┘

The Supervisor is the entry point. After each specialist node runs,
control returns to the Supervisor for re-evaluation. The Supervisor
sets `next_step` which the conditional edge reads to route the graph.

Key Flows:
- Setup → Supervisor → Researcher → Supervisor
- Supervisor → Planner → Supervisor
- Supervisor → TestGen → Supervisor
- Supervisor → Coder → Reviewer → Supervisor
- Supervisor → END (when resolved)
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from issue_resolver.state import AgentState
from issue_resolver.nodes import (
    setup_node,
    supervisor_node,
    researcher_node,
    planner_node,
    testgen_node,
    coder_node,
    reviewer_node,
)

def _route_supervisor(state: AgentState) -> str:
    """Read the Supervisor's routing decision from state."""
    next_step = state.get("next_step", "end")
    if next_step == "researcher":
        return "researcher"
    elif next_step == "planner":
        return "planner"
    elif next_step == "test_generator":
        return "test_generator"
    elif next_step == "coder":
        return "coder"
    else:
        return "end"


def build_graph() -> StateGraph:
    """Construct and compile the full agent graph."""

    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────
    graph.add_node("setup", setup_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("planner", planner_node)
    graph.add_node("test_generator", testgen_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)

    # ── Entry point ─────────────────────────────────────────────────
    graph.set_entry_point("setup")
    graph.add_edge("setup", "supervisor")

    # ── Conditional edges from Supervisor ───────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "researcher": "researcher",
            "planner": "planner",
            "test_generator": "test_generator",
            "coder": "coder",
            "end": END,
        },
    )

    # ── After each specialist, loop back to Supervisor ──────────────
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("planner", "supervisor")
    graph.add_edge("test_generator", "supervisor")
    # Coder proposes a fix, Reviewer tests it
    graph.add_edge("coder", "reviewer")
    # Reviewer returns errors (or lack thereof), loop back to Supervisor
    graph.add_edge("reviewer", "supervisor")

    # ── Compile ─────────────────────────────────────────────────────
    compiled = graph.compile()
    print("[OK] Graph compiled successfully!")
    return compiled

app = build_graph()
