"""Context Curator Node — retrieves, scores, and ranks file context for issues.

Combines semantic, AST, dependency, symbol, and keyword overlap via the
HybridRetriever. Each file receives a confidence score. Low-confidence files
trigger warning flags.
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

    # Get graph and build embedding index if missing
    graph = runtime_context.get_knowledge_graph()
    if graph is None:
        from issue_resolver.intelligence.graph_builder import GraphBuilder
        graph = GraphBuilder(repo_path).build()
        runtime_context.set_knowledge_graph(graph)

    embedding_index = RepoEmbeddingIndex()
    embedding_index.build_from_graph(graph)

    # 1. Run hybrid retrieval
    retriever = HybridRetriever(graph, embedding_index)
    scored_results = retriever.retrieve(issue, top_k=12)

    duration_ms = (time.monotonic() - t0) * 1000

    # 2. Score and check confidence
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
        # Pick first few keywords from issue
        keywords = retriever._extract_identifiers(issue)
        if keywords:
            fallback_res = search_code.invoke({"query": keywords[0], "directory": repo_path})
            # Add files from search code output if any are found
            import re
            found_paths = re.findall(r"^(?:file:///)?([^\s:]+)(?::\d+)?", fallback_res, re.MULTILINE)
            for path in found_paths:
                clean_path = path.replace("\\", "/").lstrip("./")
                if clean_path in graph.modules and clean_path not in file_context_paths:
                    file_context_paths.append(clean_path)
                    context_confidence[clean_path] = "medium"
                    confidence_counts["medium"] += 1

    # Record trace event if running
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record_retrieval(
            agent="ContextCurator",
            files=file_context_paths,
            confidence_scores={path: float(next(r.score for r in scored_results if r.path == path)) for path in file_context_paths if any(r.path == path for r in scored_results)},
        )

    # Load contents for the final file_context
    loaded_snippets = []
    for path in file_context_paths[:8]:  # Limit loaded contents to top 8 to stay under token budget
        content = read_file.invoke({"file_path": f"{repo_path}/{path}"})
        if not content.startswith("Error"):
            loaded_snippets.append(f"=== File: {path} ===\n{content}")

    return {
        "file_context": loaded_snippets,
        "context_confidence": context_confidence,
        "history": append_to_history(
            "ContextCurator",
            "Retrieve",
            f"Curated {len(file_context_paths)} files ({confidence_counts['high']} high, {confidence_counts['medium']} medium confidence).",
        ),
    }
