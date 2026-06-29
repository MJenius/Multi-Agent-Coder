"""Multi-signal hybrid retriever.

Combines five signals for context retrieval:
1. Semantic similarity (embeddings)
2. AST relationships (knowledge graph distance)
3. Dependency distance (import hops)
4. Symbol overlap (exact/fuzzy identifier match)
5. Issue keyword matching (ripgrep / regex)

Every retrieved file receives a confidence score:
- High (≥0.75): strong multi-signal match
- Medium (0.45–0.75): partial match, planner should verify
- Low (<0.45): weak match, retrieve more context before editing
"""

from __future__ import annotations

import re
from typing import Any

from issue_resolver.core.interfaces import Confidence, ScoredResult
from issue_resolver.intelligence.knowledge_graph import RepoKnowledgeGraph
from issue_resolver.intelligence.embeddings import RepoEmbeddingIndex


# Default signal weights (configurable via model_routing.json)
_DEFAULT_WEIGHTS = {
    "semantic": 0.30,
    "ast_relationship": 0.20,
    "dependency_distance": 0.15,
    "symbol_overlap": 0.20,
    "keyword_match": 0.15,
}


class HybridRetriever:
    """Multi-signal retrieval engine.

    Usage::

        retriever = HybridRetriever(graph, embedding_index)
        results = retriever.retrieve("calculate_total crashes on empty list", top_k=10)
        for result in results:
            print(f"{result.path}: {result.score:.2f} ({result.confidence.value})")
    """

    def __init__(
        self,
        graph: RepoKnowledgeGraph,
        embedding_index: RepoEmbeddingIndex,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.graph = graph
        self.embedding_index = embedding_index
        self.weights = weights or dict(_DEFAULT_WEIGHTS)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        hint_files: list[str] | None = None,
    ) -> list[ScoredResult]:
        """Retrieve the top-k most relevant files for *query*.

        Returns ``ScoredResult`` objects with composite scores and
        per-signal breakdowns.
        """
        all_paths = list(self.graph.modules.keys())
        if not all_paths:
            return []

        # Resolve symbols in query to anchor files in Repository Graph
        keywords = self._extract_identifiers(query)
        symbol_files = []
        for kw in keywords:
            cls = self.graph.get_class(kw)
            if cls:
                symbol_files.append(cls.file_path)
            fn = self.graph.get_function(kw)
            if fn:
                symbol_files.append(fn.file_path)

        anchors = list(hint_files) if hint_files else []
        for sf in symbol_files:
            if sf not in anchors:
                anchors.append(sf)

        # Compute per-signal scores for every file
        scores: dict[str, dict[str, float]] = {p: {} for p in all_paths}

        # Signal 1: Semantic similarity
        semantic_results = self.embedding_index.search_files(query, top_k=len(all_paths))
        semantic_map = {path: score for path, score in semantic_results}
        max_semantic = max(semantic_map.values()) if semantic_map else 1.0
        for path in all_paths:
            raw = semantic_map.get(path, 0.0)
            scores[path]["semantic"] = raw / max_semantic if max_semantic > 0 else 0.0

        # Signal 2: AST relationship (proximity to anchors)
        if anchors:
            for path in all_paths:
                ast_score = self._compute_ast_score(path, anchors)
                scores[path]["ast_relationship"] = ast_score
        else:
            for path in all_paths:
                scores[path]["ast_relationship"] = 0.0

        # Signal 3: Dependency distance
        if anchors:
            for path in all_paths:
                dep_score = self._compute_dependency_score(path, anchors)
                scores[path]["dependency_distance"] = dep_score
        else:
            for path in all_paths:
                scores[path]["dependency_distance"] = 0.0

        # Signal 4: Symbol overlap
        for path in all_paths:
            symbol_score = self._compute_symbol_overlap(path, keywords)
            scores[path]["symbol_overlap"] = symbol_score

        # Signal 5: Keyword match (filename + content heuristic)
        for path in all_paths:
            kw_score = self._compute_keyword_score(path, keywords)
            scores[path]["keyword_match"] = kw_score

        # Compute weighted composite score
        results: list[ScoredResult] = []
        for path in all_paths:
            signal_scores = scores[path]
            composite = sum(
                signal_scores.get(signal, 0.0) * self.weights.get(signal, 0.0)
                for signal in self.weights
            )

            # Boost hint files
            if hint_files and path in hint_files:
                composite = max(composite, 0.85)

            if composite > 0.01:  # Skip zero-score files
                results.append(ScoredResult(
                    path=path,
                    content="",  # Content loaded on demand
                    score=min(composite, 1.0),
                    signal_scores=signal_scores,
                ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ----- signal computation -----

    def _compute_ast_score(self, path: str, hint_files: list[str]) -> float:
        """Score based on graph distance to hint files."""
        if path in hint_files:
            return 1.0
        module = self.graph.get_module(path)
        if not module:
            return 0.0

        # Check if this file shares classes/functions with hint files
        for hint in hint_files:
            hint_mod = self.graph.get_module(hint)
            if not hint_mod:
                continue
            # Same directory bonus
            if path.rsplit("/", 1)[0] == hint.rsplit("/", 1)[0]:
                return 0.6
            # Shared symbols
            shared = set(module.classes) & set(hint_mod.classes)
            shared |= set(module.functions) & set(hint_mod.functions)
            if shared:
                return 0.8
        return 0.0

    def _compute_dependency_score(self, path: str, hint_files: list[str]) -> float:
        """Score based on import distance to hint files."""
        if path in hint_files:
            return 1.0
        for hint in hint_files:
            # Direct import
            deps = self.graph.get_dependencies(hint)
            if path in deps or any(path.endswith(d.replace(".", "/") + ".py") for d in deps):
                return 1.0
            # Reverse: hint imports from this file
            dependents = self.graph.get_dependents(path)
            if hint in dependents:
                return 0.8
        return 0.0

    def _compute_symbol_overlap(self, path: str, keywords: list[str]) -> float:
        """Score based on how many issue identifiers appear in this file's symbols."""
        module = self.graph.get_module(path)
        if not module or not keywords:
            return 0.0

        all_symbols = set(s.lower() for s in module.classes + module.functions + module.exports)
        matches = sum(1 for kw in keywords if kw.lower() in all_symbols)
        # Also check fuzzy (substring)
        fuzzy_matches = sum(
            1 for kw in keywords
            if any(kw.lower() in sym for sym in all_symbols)
        )
        total = matches * 2 + fuzzy_matches
        return min(total / max(len(keywords), 1), 1.0)

    def _compute_keyword_score(self, path: str, keywords: list[str]) -> float:
        """Score based on keyword presence in filename/path."""
        if not keywords:
            return 0.0
        path_lower = path.lower()
        matches = sum(1 for kw in keywords if kw.lower() in path_lower)
        return min(matches / max(len(keywords), 1), 1.0)

    def _extract_identifiers(self, text: str) -> list[str]:
        """Extract likely code identifiers from issue text."""
        # Backtick-quoted identifiers
        backtick = re.findall(r"`([^`\n]{2,60})`", text)
        identifiers = [m.split("(")[0].strip() for m in backtick]

        # camelCase and snake_case
        identifiers.extend(re.findall(r"\b([a-z][a-zA-Z0-9]*_[a-zA-Z0-9_]+)\b", text))
        identifiers.extend(re.findall(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", text))

        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for ident in identifiers:
            lower = ident.lower()
            if lower not in seen and len(ident) >= 3:
                seen.add(lower)
                unique.append(ident)
        return unique[:20]
