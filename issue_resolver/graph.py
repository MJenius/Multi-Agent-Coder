"""
Graph Builder — Compiles the LangGraph StateGraph for the issue resolver.

Topology
--------
                 ┌──────────────┐
        ┌───────►│  Researcher  ├───────┐
        │        └──────────────┘       │
  ┌─────┴──────┐                  ┌─────▼──────┐
  │ Supervisor │◄─────────────────┤ Supervisor │  (re-enters after every agent)
  └─────┬──────┘                  └─────┬──────┘
        │        ┌──────────────┐       │
        ├───────►│    Coder     ├───────┘
        │        └──────────────┘
        │
        └───────► END

The Supervisor is the entry point.  After each specialist node runs,
control returns to the Supervisor for re-evaluation.  The Supervisor
sets `next_step` which the conditional edge reads to route the graph.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from issue_resolver.state import AgentState
from issue_resolver.nodes import supervisor_node, researcher_node, coder_node


def _route_supervisor(state: AgentState) -> str:
    """Read the Supervisor's routing decision from state."""
    next_step = state.get("next_step", "end")
    if next_step == "researcher":
        return "researcher"
    elif next_step == "coder":
        return "coder"
    else:
        return "end"


def build_graph() -> StateGraph:
    """Construct and compile the full agent graph."""

    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("coder", coder_node)

    # ── Entry point ─────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── Conditional edges from Supervisor ───────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "researcher": "researcher",
            "coder": "coder",
            "end": END,
        },
    )

    # ── After each specialist, loop back to Supervisor ──────────────
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("coder", "supervisor")

    # ── Compile ─────────────────────────────────────────────────────
    compiled = graph.compile()
    print("✅ Graph compiled successfully!")
    return compiled
