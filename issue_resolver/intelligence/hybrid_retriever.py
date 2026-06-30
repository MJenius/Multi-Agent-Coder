"""Multi-signal hybrid retriever.

Combines eight signals for context retrieval:
1. Exact symbol match (class/function name hit in the knowledge graph)
2. Semantic similarity (embeddings)
3. Dependency distance (import hops)
4. AST relationships (knowledge graph proximity)
5. Import relationships (direct import edge)
6. Test proximity (is this a test for a matched file?)
7. File ownership (same directory as anchor files)
8. Historical fixes (git log proximity — placeholder)

Every retrieved file receives a confidence score:
- High (≥0.75): strong multi-signal match
- Medium (0.45–0.75): partial match, planner should verify
- Low (<0.45): weak match, retrieve more context before editing
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from issue_resolver.core.interfaces import Confidence, ScoredResult
from issue_resolver.intelligence.knowledge_graph import RepoKnowledgeGraph
from issue_resolver.intelligence.embeddings import RepoEmbeddingIndex


# Default signal weights — exact symbol match is the strongest signal
_DEFAULT_WEIGHTS = {
    "exact_symbol_match": 0.25,
    "semantic": 0.15,
    "dependency_distance": 0.15,
    "ast_relationship": 0.10,
    "import_relationship": 0.10,
    "test_proximity": 0.10,
    "file_ownership": 0.05,
    "historical_fixes": 0.10,
}


@dataclass
class RetrievalReport:
    """Extended retrieval output with diagnostic information."""

    results: list[ScoredResult]
    max_confidence: float = 0.0
    needs_more_context: bool = False
    signal_summary: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.results:
            self.max_confidence = max(r.score for r in self.results)
        self.needs_more_context = self.max_confidence < 0.45


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
        report = self.retrieve_with_report(query, top_k=top_k, hint_files=hint_files)
        return report.results

    def retrieve_with_report(
        self,
        query: str,
        top_k: int = 10,
        hint_files: list[str] | None = None,
    ) -> RetrievalReport:
        """Retrieve with full diagnostic report.

        The report includes ``needs_more_context`` flag and per-signal
        contribution summary.
        """
        all_paths = list(self.graph.modules.keys())
        if not all_paths:
            return RetrievalReport(results=[])

        # Resolve symbols in query to anchor files in Repository Graph
        keywords = self._extract_identifiers(query)
        symbol_files: list[str] = []
        matched_symbols: set[str] = set()
        for kw in keywords:
            cls = self.graph.get_class(kw)
            if cls:
                symbol_files.append(cls.file_path)
                matched_symbols.add(kw)
            fn = self.graph.get_function(kw)
            if fn:
                symbol_files.append(fn.file_path)
                matched_symbols.add(kw)

        anchors = list(hint_files) if hint_files else []
        for sf in symbol_files:
            if sf not in anchors:
                anchors.append(sf)

        # Determine test file set for proximity scoring
        test_files_for_anchors: set[str] = set()
        for anchor in anchors:
            test_files_for_anchors.update(self.graph.get_tests_for(anchor))

        # Compute per-signal scores for every file
        scores: dict[str, dict[str, float]] = {p: {} for p in all_paths}

        # Signal 1: Exact symbol match (strongest signal)
        for path in all_paths:
            scores[path]["exact_symbol_match"] = self._compute_exact_symbol_match(
                path, matched_symbols,
            )

        # Signal 2: Semantic similarity
        semantic_results = self.embedding_index.search_files(query, top_k=len(all_paths))
        semantic_map = {path: score for path, score in semantic_results}
        max_semantic = max(semantic_map.values()) if semantic_map else 1.0
        for path in all_paths:
            raw = semantic_map.get(path, 0.0)
            scores[path]["semantic"] = raw / max_semantic if max_semantic > 0 else 0.0

        # Signal 3: Dependency distance
        if anchors:
            for path in all_paths:
                dep_score = self._compute_dependency_score(path, anchors)
                scores[path]["dependency_distance"] = dep_score
        else:
            for path in all_paths:
                scores[path]["dependency_distance"] = 0.0

        # Signal 4: AST relationship (proximity to anchors)
        if anchors:
            for path in all_paths:
                ast_score = self._compute_ast_score(path, anchors)
                scores[path]["ast_relationship"] = ast_score
        else:
            for path in all_paths:
                scores[path]["ast_relationship"] = 0.0

        # Signal 5: Import relationship (direct import edge)
        if anchors:
            for path in all_paths:
                scores[path]["import_relationship"] = self._compute_import_relationship(
                    path, anchors,
                )
        else:
            for path in all_paths:
                scores[path]["import_relationship"] = 0.0

        # Signal 6: Test proximity
        for path in all_paths:
            scores[path]["test_proximity"] = self._compute_test_proximity(
                path, test_files_for_anchors, anchors,
            )

        # Signal 7: File ownership (same directory)
        if anchors:
            for path in all_paths:
                scores[path]["file_ownership"] = self._compute_file_ownership(
                    path, anchors,
                )
        else:
            for path in all_paths:
                scores[path]["file_ownership"] = 0.0

        # Signal 8: Historical fixes (placeholder — git log proximity)
        for path in all_paths:
            scores[path]["historical_fixes"] = 0.0

        # Compute weighted composite score
        results: list[ScoredResult] = []
        signal_totals: dict[str, float] = {s: 0.0 for s in self.weights}

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
                for signal in self.weights:
                    signal_totals[signal] += signal_scores.get(signal, 0.0)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        top_results = results[:top_k]

        # Build signal summary (average contribution of each signal)
        n = len(top_results) or 1
        signal_summary = {s: total / n for s, total in signal_totals.items()}

        return RetrievalReport(
            results=top_results,
            signal_summary=signal_summary,
        )

    # ----- signal computation -----

    def _compute_exact_symbol_match(
        self, path: str, matched_symbols: set[str],
    ) -> float:
        """Score 1.0 if this file defines a symbol that was matched from the query."""
        if not matched_symbols:
            return 0.0
        module = self.graph.get_module(path)
        if not module:
            return 0.0
        all_symbols = set(s.lower() for s in module.classes + module.functions)
        matches = sum(1 for s in matched_symbols if s.lower() in all_symbols)
        if matches == 0:
            return 0.0
        return min(matches / len(matched_symbols), 1.0)

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
            # Shared symbols
            shared = set(module.classes) & set(hint_mod.classes)
            shared |= set(module.functions) & set(hint_mod.functions)
            if shared:
                return 0.8
            # Same directory bonus
            if path.rsplit("/", 1)[0] == hint.rsplit("/", 1)[0]:
                return 0.5
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

    def _compute_import_relationship(
        self, path: str, anchors: list[str],
    ) -> float:
        """Score based on direct import edge existence."""
        if path in anchors:
            return 1.0
        for anchor in anchors:
            deps = self.graph.get_dependencies(anchor)
            if path in deps:
                return 1.0
            deps_rev = self.graph.get_dependencies(path)
            if anchor in deps_rev:
                return 0.8
        return 0.0

    def _compute_test_proximity(
        self,
        path: str,
        test_files_for_anchors: set[str],
        anchors: list[str],
    ) -> float:
        """Score based on whether this file is a test for an anchor."""
        if path in test_files_for_anchors:
            return 1.0
        # Check if this file is tested by any of the test files
        module = self.graph.get_module(path)
        if module and module.is_test:
            for anchor in anchors:
                if path in self.graph.get_tests_for(anchor):
                    return 1.0
        return 0.0

    def _compute_file_ownership(
        self, path: str, anchors: list[str],
    ) -> float:
        """Score based on shared directory with anchor files."""
        if path in anchors:
            return 1.0
        path_dir = path.rsplit("/", 1)[0] if "/" in path else ""
        for anchor in anchors:
            anchor_dir = anchor.rsplit("/", 1)[0] if "/" in anchor else ""
            if path_dir and path_dir == anchor_dir:
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

        # PascalCase class names
        identifiers.extend(re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", text))

        # camelCase and snake_case
        identifiers.extend(re.findall(r"\b([a-z][a-zA-Z0-9]*_[a-zA-Z0-9_]+)\b", text))
        identifiers.extend(re.findall(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", text))

        # Dotted names (module.Class.method)
        identifiers.extend(re.findall(r"\b([a-zA-Z_]\w+\.[a-zA-Z_]\w+(?:\.[a-zA-Z_]\w+)*)\b", text))

        # File paths
        identifiers.extend(re.findall(r"([a-zA-Z_]\w*(?:/[a-zA-Z_]\w*)*\.(?:py|js|ts|go|rs|java|cpp|c|h))", text))

        # Error class names (e.g. TypeError, ValueError)
        identifiers.extend(re.findall(r"\b([A-Z][a-z]+(?:Error|Exception|Warning|Fault))\b", text))

        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for ident in identifiers:
            lower = ident.lower()
            if lower not in seen and len(ident) >= 3:
                seen.add(lower)
                unique.append(ident)
        return unique[:30]
