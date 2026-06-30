"""Repo Intelligence Node — builds the Repository Knowledge Graph, indexes, and LSP bridge.

Executes during workspace discovery setup. Builds the static AST knowledge
graph, the embedding index, the hybrid retriever, and initialises the LSP
bridge — storing them all as runtime context singletons so that every
downstream node queries intelligence centrally.
"""

from __future__ import annotations

import time

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.intelligence.knowledge_graph import RepoKnowledgeGraph
from issue_resolver.intelligence.graph_builder import GraphBuilder
from issue_resolver.intelligence.embeddings import RepoEmbeddingIndex
from issue_resolver.intelligence.hybrid_retriever import HybridRetriever
import issue_resolver.runtime_context as runtime_context


def repo_intelligence_node(state: AgentState) -> dict:
    """Build knowledge graph, semantic index, hybrid retriever, and LSP bridge."""
    repo_path = state.get("repo_path", ".")
    print(f"[RepoIntelligence] Building knowledge graph for repository at {repo_path}...")

    t0 = time.monotonic()

    # 1. Build knowledge graph
    builder = GraphBuilder(repo_path)
    graph = builder.build()
    runtime_context.set_knowledge_graph(graph)

    # 2. Build embedding index and store centrally
    embedding_index = RepoEmbeddingIndex()
    embedding_index.build_from_graph(graph)
    runtime_context.set_embedding_index(embedding_index)

    # 3. Build hybrid retriever and store centrally
    retriever = HybridRetriever(graph, embedding_index)
    runtime_context.set_hybrid_retriever(retriever)

    # 4. Initialise LSP bridge (best-effort, graceful degradation)
    lsp_available = False
    try:
        from issue_resolver.intelligence.lsp_bridge import LSPBridge
        # Detect primary language from graph
        lang_counts: dict[str, int] = {}
        for mod in graph.modules.values():
            if mod.language and not mod.is_config:
                lang_counts[mod.language] = lang_counts.get(mod.language, 0) + 1
        primary_lang = max(lang_counts, key=lang_counts.get) if lang_counts else ""

        if primary_lang:
            lsp = LSPBridge(repo_path, primary_lang)
            if lsp.is_available:
                runtime_context.set_lsp_bridge(lsp)
                lsp_available = True
                print(f"[RepoIntelligence] LSP bridge initialised for {primary_lang}")
            else:
                print(f"[RepoIntelligence] LSP bridge not available for {primary_lang} (server not installed)")
    except ImportError:
        print("[RepoIntelligence] LSP bridge module not available, skipping")
    except Exception as exc:
        print(f"[RepoIntelligence] LSP bridge initialisation failed: {exc}")

    duration_ms = (time.monotonic() - t0) * 1000
    print(f"[RepoIntelligence] Graph, index, and retriever built in {duration_ms:.1f}ms")

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
                "lsp_available": lsp_available,
            },
        )

    return {
        "repo_intelligence": summary,
        "history": append_to_history(
            "RepoIntelligence",
            "Index",
            f"Indexed {len(graph.modules)} modules, {len(graph.classes)} classes, "
            f"{len(graph.functions)} functions in {duration_ms/1000:.1f}s. "
            f"LSP: {'available' if lsp_available else 'unavailable'}",
        ),
    }
