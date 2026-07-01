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
    """Retrieve and curate context, using shared localization primarily, expanding if confidence is low."""
    issue = state.get("issue", "")
    repo_path = state.get("repo_path", ".")
    print("[ContextCurator] Curating context for planning...")

    # Get graph and indexes centrally from runtime_context
    graph = runtime_context.get_knowledge_graph()
    embedding_index = runtime_context.get_embedding_index()
    retriever = runtime_context.get_hybrid_retriever()
    
    # 1. Read shared localization output
    localization = state.get("localization_result", {})
    localization_confidence = state.get("localization_confidence", 0.0)
    
    primary_files = localization.get("primary_files", [])
    file_context_paths = [f["path"] for f in primary_files]
    context_confidence = {f["path"]: f["confidence"] for f in primary_files}
    
    # 2. Check if we need validation / context expansion (Requirement 1 & 5)
    # If confidence is low or needs fallback or no files found, expand context
    needs_expansion = (localization_confidence < 0.7 or localization.get("needs_researcher_fallback", False) or not primary_files)
    
    expansion_reasons = []
    if needs_expansion:
        print("[ContextCurator] Diagnostics: Low-confidence/empty localization. Performing validation and context expansion...")
        
        # Build retriever centrally if needed
        if retriever is None and graph is not None:
            if embedding_index is None:
                from issue_resolver.intelligence.embeddings import RepoEmbeddingIndex
                embedding_index = RepoEmbeddingIndex()
                embedding_index.build_from_graph(graph)
                runtime_context.set_embedding_index(embedding_index)
            from issue_resolver.intelligence.hybrid_retriever import HybridRetriever
            retriever = HybridRetriever(graph, embedding_index)
            runtime_context.set_hybrid_retriever(retriever)
            
        if retriever is not None:
            report_expanded = retriever.retrieve_with_report(issue, top_k=20)
            expanded_results = report_expanded.results
            
            # Merge expanded results into our paths
            for res in expanded_results:
                if res.path not in file_context_paths:
                    file_context_paths.append(res.path)
                    context_confidence[res.path] = res.confidence.value
                    expansion_reasons.append(f"expanded candidate: {res.path}")
                    
        # Fallback keyword code search if we still have low confidence/no files
        if len(file_context_paths) < 3:
            from issue_resolver.tools.repo_tools import search_code
            import re
            keywords = retriever._extract_identifiers(issue) if retriever else []
            if keywords:
                fallback_res = search_code.invoke({"query": keywords[0], "directory": repo_path})
                found_paths = re.findall(r"^(?:file:///)?([^\s:]+)(?::\d+)?", fallback_res, re.MULTILINE)
                for path in found_paths:
                    clean_path = path.replace("\\", "/").lstrip("./")
                    if graph and clean_path in graph.modules and clean_path not in file_context_paths:
                        file_context_paths.append(clean_path)
                        context_confidence[clean_path] = "low"
                        expansion_reasons.append(f"keyword fallback: {clean_path}")

    # Load contents for the final file_context
    loaded_snippets = []
    # Pull up to 12 files if expanded context was used, otherwise top 8
    max_snippets = 12 if needs_expansion else 8
    selected_paths = file_context_paths[:max_snippets]
    
    # 3. Log diagnostics (Requirement 6)
    print(f"[ContextCurator] Diagnostics: Selected {len(selected_paths)} files for context:")
    for path in selected_paths:
        origin = "shared localization"
        for r in expansion_reasons:
            if path in r:
                origin = "validation expansion"
                break
        conf = context_confidence.get(path, "unknown")
        print(f"  - `{path}` (origin={origin}, confidence={conf})")
        
    for path in selected_paths:
        content = read_file.invoke({"file_path": f"{repo_path}/{path}"})
        if not content.startswith("Error"):
            loaded_snippets.append(f"=== File: {path} ===\n{content}")
            
    # Record trace event
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record_retrieval(
            agent="ContextCurator",
            files=selected_paths,
            confidence_scores={
                path: 0.8 if context_confidence.get(path) == "high" else (0.5 if context_confidence.get(path) == "medium" else 0.2)
                for path in selected_paths
            },
        )
        
    return {
        "file_context": loaded_snippets,
        "context_confidence": context_confidence,
        "history": append_to_history(
            "ContextCurator",
            "Retrieve",
            f"Curated {len(selected_paths)} files. Expanded: {needs_expansion} (added {len(expansion_reasons)} files).",
        ),
    }
