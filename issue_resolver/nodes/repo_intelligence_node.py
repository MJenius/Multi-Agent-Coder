"""Repo Intelligence Node — builds the Repository Knowledge Graph and index.

Executes during workspace discovery setup. Builds the static AST knowledge
graph and the embedding index, storing them as runtime context singletons.
"""

from __future__ import annotations

import time

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.intelligence.knowledge_graph import RepoKnowledgeGraph
from issue_resolver.intelligence.graph_builder import GraphBuilder
from issue_resolver.intelligence.embeddings import RepoEmbeddingIndex
import issue_resolver.runtime_context as runtime_context


def repo_intelligence_node(state: AgentState) -> dict:
    """Build knowledge graph and semantic index for the repo."""
    repo_path = state.get("repo_path", ".")
    print(f"[RepoIntelligence] Building knowledge graph for repository at {repo_path}...")

    t0 = time.monotonic()

    # 1. Build knowledge graph
    builder = GraphBuilder(repo_path)
    graph = builder.build()
    runtime_context.set_knowledge_graph(graph)

    # 2. Build embedding index
    embedding_index = RepoEmbeddingIndex()
    embedding_index.build_from_graph(graph)

    duration_ms = (time.monotonic() - t0) * 1000
    print(f"[RepoIntelligence] Graph and index built in {duration_ms:.1f}ms")

    # Serialize compact summary for state storage
    summary = graph.to_summary_dict()

    # Record trace event if running
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "repo_intelligence_built",
            "RepoIntelligence",
            f"Built knowledge graph in {duration_ms:.1f}ms",
            duration_ms=duration_ms,
            details={
                "total_modules": len(graph.modules),
                "total_classes": len(graph.classes),
                "total_functions": len(graph.functions),
            },
        )

    return {
        "repo_intelligence": summary,
        "history": append_to_history(
            "RepoIntelligence",
            "Index",
            f"Indexed {len(graph.modules)} modules, {len(graph.classes)} classes, {len(graph.functions)} functions in {duration_ms/1000:.1f}s",
        ),
    }
