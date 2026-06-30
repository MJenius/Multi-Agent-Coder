"""Context Curator Node — retrieves, scores, and ranks file context for issues.

Reuses the centralized graph, embedding index, and hybrid retriever from
``runtime_context``.  If context confidence is low, it dynamically expands the
retrieval radius to pull in additional files, import neighbors, and test files.
"""

from __future__ import annotations

import time

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.intelligence.hybrid_retriever import HybridRetriever
from issue_resolver.intelligence.embeddings import RepoEmbeddingIndex
import issue_resolver.runtime_context as runtime_context
from issue_resolver.tools.repo_tools import read_file


def context_curator_node(state: AgentState) -> dict:
    """Retrieve and curate high-confidence context for issue planning."""
    issue = state.get("issue", "")
    repo_path = state.get("repo_path", ".")
    print("[ContextCurator] Curating context for planning...")

    t0 = time.monotonic()

    # Get graph and indexes centrally from runtime_context
    graph = runtime_context.get_knowledge_graph()
    embedding_index = runtime_context.get_embedding_index()
    retriever = runtime_context.get_hybrid_retriever()

    if retriever is None:
        # Fallback setup if not pre-built
        if graph is None:
            from issue_resolver.intelligence.graph_builder import GraphBuilder
            graph = GraphBuilder(repo_path).build()
            runtime_context.set_knowledge_graph(graph)

        if embedding_index is None:
            embedding_index = RepoEmbeddingIndex()
            embedding_index.build_from_graph(graph)
            runtime_context.set_embedding_index(embedding_index)

        retriever = HybridRetriever(graph, embedding_index)
        runtime_context.set_hybrid_retriever(retriever)

    # 1. Run hybrid retrieval
    report = retriever.retrieve_with_report(issue, top_k=12)
    scored_results = report.results

    # 2. Check if we need adaptive context expansion
    localization_result = state.get("localization_result", {})
    needs_more_context = report.needs_more_context or localization_result.get("needs_researcher_fallback", False)

    if needs_more_context:
        print("[ContextCurator] Low-confidence context detected. Running adaptive expansion...")
        # Increase top_k to 20 for wider coverage
        report_expanded = retriever.retrieve_with_report(issue, top_k=20)
        scored_results = report_expanded.results

    duration_ms = (time.monotonic() - t0) * 1000

    # 3. Score and check confidence
    file_context_paths = []
    context_confidence = {}
    confidence_counts = {"high": 0, "medium": 0, "low": 0}

    # Load contents and construct context list
    for res in scored_results:
        file_context_paths.append(res.path)
        context_confidence[res.path] = res.confidence.value
        confidence_counts[res.confidence.value] += 1

    print(
        f"[ContextCurator] Retrieved {len(file_context_paths)} files in {duration_ms:.1f}ms "
        f"({confidence_counts['high']} high, {confidence_counts['medium']} medium, {confidence_counts['low']} low confidence)"
    )

    # If no high or medium confidence files found, fall back to keyword code search
    if confidence_counts["high"] == 0 and confidence_counts["medium"] == 0:
        print("[ContextCurator] Warning: low-confidence context detected. Performing fallback keyword search...")
        from issue_resolver.tools.repo_tools import search_code
        keywords = retriever._extract_identifiers(issue)
        if keywords:
            fallback_res = search_code.invoke({"query": keywords[0], "directory": repo_path})
            import re
            found_paths = re.findall(r"^(?:file:///)?([^\s:]+)(?::\d+)?", fallback_res, re.MULTILINE)
            for path in found_paths:
                clean_path = path.replace("\\", "/").lstrip("./")
                if graph and clean_path in graph.modules and clean_path not in file_context_paths:
                    file_context_paths.append(clean_path)
                    context_confidence[clean_path] = "medium"
                    confidence_counts["medium"] += 1

    # Record trace event
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record_retrieval(
            agent="ContextCurator",
            files=file_context_paths,
            confidence_scores={
                path: float(next(r.score for r in scored_results if r.path == path))
                for path in file_context_paths
                if any(r.path == path for r in scored_results)
            },
        )

    # Load contents for the final file_context
    loaded_snippets = []
    # Pull up to 12 files if expanded context was requested, otherwise top 8
    max_snippets = 12 if needs_more_context else 8
    for path in file_context_paths[:max_snippets]:
        content = read_file.invoke({"file_path": f"{repo_path}/{path}"})
        if not content.startswith("Error"):
            loaded_snippets.append(f"=== File: {path} ===\n{content}")

    return {
        "file_context": loaded_snippets,
        "context_confidence": context_confidence,
        "history": append_to_history(
            "ContextCurator",
            "Retrieve",
            f"Curated {len(file_context_paths)} files ({confidence_counts['high']} high, "
            f"{confidence_counts['medium']} medium confidence). Expanded: {needs_more_context}.",
        ),
    }
