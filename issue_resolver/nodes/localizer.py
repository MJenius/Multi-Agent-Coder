"""Repository Localizer Node — deterministic entity extraction and graph-based code localization.

This is the **primary** localization path.  It extracts entities from
the issue text using regex, looks them up in the Repository Knowledge
Graph and (when available) the LSP bridge, then scores and ranks all
located files.

When localization confidence is low (< 0.4) or graph miss rate is
high (> 40%), the ``needs_researcher_fallback`` flag is set so the
Researcher can be invoked as a secondary path.

No LLM calls are made here — this is pure graph traversal + regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from issue_resolver.core.interfaces import Confidence
from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
import issue_resolver.runtime_context as runtime_context


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LocatedSymbol:
    """A symbol found in the repository graph or via LSP."""

    name: str
    kind: str                   # "class", "function", "method", "parameter"
    file_path: str
    line_number: int
    end_line: int
    parent_class: str           # empty for top-level
    confidence: float           # how confident we are this is the right symbol
    source: str                 # "graph", "lsp", "ripgrep"


@dataclass
class ScoredFile:
    """A file scored by the Localizer."""

    path: str
    score: float
    confidence: Confidence
    reasons: list[str] = field(default_factory=list)


@dataclass
class LocalizationResult:
    """Complete output of the Localizer pipeline."""

    primary_files: list[ScoredFile]
    symbols: list[LocatedSymbol]
    references: list[str]
    test_files: list[str]
    dependency_neighbors: list[str]
    confidence: float                    # overall localization confidence (0.0–1.0)
    entities_extracted: list[str]
    graph_hits: int
    graph_misses: int
    missed_entities: list[str]
    lsp_available: bool
    needs_researcher_fallback: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_files": [
                {"path": f.path, "score": f.score, "confidence": f.confidence.value, "reasons": f.reasons}
                for f in self.primary_files
            ],
            "symbols": [
                {
                    "name": s.name, "kind": s.kind, "file_path": s.file_path,
                    "line_number": s.line_number, "end_line": s.end_line,
                    "parent_class": s.parent_class, "confidence": s.confidence,
                    "source": s.source,
                }
                for s in self.symbols
            ],
            "references": self.references,
            "test_files": self.test_files,
            "dependency_neighbors": self.dependency_neighbors,
            "confidence": self.confidence,
            "entities_extracted": self.entities_extracted,
            "graph_hits": self.graph_hits,
            "graph_misses": self.graph_misses,
            "missed_entities": self.missed_entities,
            "lsp_available": self.lsp_available,
            "needs_researcher_fallback": self.needs_researcher_fallback,
        }


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def _extract_entities_with_sources(issue_text: str) -> dict[str, set[str]]:
    """Extract likely code entities from the issue text, categorised by source/signal type."""
    entity_sources: dict[str, set[str]] = {}

    # 1. Backtick-quoted identifiers (highest signal)
    backtick = re.findall(r"`([^`\n]{2,80})`", issue_text)
    for m in backtick:
        clean = m.split("(")[0].strip()
        if clean and not clean.startswith("http"):
            entity_sources.setdefault(clean, set()).add("backtick")

    # 2. Code blocks — extract function/class definitions
    code_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)\n```", issue_text, re.DOTALL)
    for block in code_blocks:
        for val in re.findall(r"(?:def|class)\s+(\w+)", block):
            entity_sources.setdefault(val, set()).add("code_block")
        for val in re.findall(r"(?:function|const|let|var|export)\s+(\w+)", block):
            entity_sources.setdefault(val, set()).add("code_block")

    # 3. File paths
    file_paths = re.findall(
        r"([a-zA-Z_]\w*(?:/[a-zA-Z_]\w*)*\.(?:py|js|ts|go|rs|java|cpp|c|h|jsx|tsx|vue|rb))",
        issue_text,
    )
    for path in file_paths:
        entity_sources.setdefault(path, set()).add("file_path")

    # 4. PascalCase class names (e.g. TableTitle, HttpClient)
    pascal = re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", issue_text)
    for val in pascal:
        entity_sources.setdefault(val, set()).add("pascal")

    # 5. snake_case identifiers
    snake = re.findall(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b", issue_text)
    for val in snake:
        entity_sources.setdefault(val, set()).add("snake")

    # 6. camelCase identifiers
    camel = re.findall(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", issue_text)
    for val in camel:
        entity_sources.setdefault(val, set()).add("camel")

    # 7. Dotted names (module.Class.method)
    dotted = re.findall(
        r"\b([a-zA-Z_]\w+\.[a-zA-Z_]\w+(?:\.[a-zA-Z_]\w+)*)\b",
        issue_text,
    )
    for val in dotted:
        entity_sources.setdefault(val, set()).add("dotted")

    # 8. Error class names
    errors = re.findall(r"\b([A-Z][a-z]+(?:Error|Exception|Warning|Fault))\b", issue_text)
    for val in errors:
        entity_sources.setdefault(val, set()).add("error_class")

    # 9. Traceback file references: File "foo/bar.py", line 42
    trace_files = re.findall(r'File "([^"]+)"', issue_text)
    for path in trace_files:
        entity_sources.setdefault(path, set()).add("traceback_file")

    # 10. Traceback function names: in calculate_total
    trace_funcs = re.findall(r"^\s+in (\w+)\s*$", issue_text, re.MULTILINE)
    for func in trace_funcs:
        entity_sources.setdefault(func, set()).add("traceback_func")

    # Deduplicate and filter noise
    filtered_sources: dict[str, set[str]] = {}
    noise = {"the", "this", "that", "with", "from", "import", "and", "for", "not", "but",
             "def", "class", "return", "none", "true", "false", "self", "cls"}
    for ent, sources in entity_sources.items():
        lower = ent.lower()
        if len(ent) >= 2 and lower not in noise:
            filtered_sources[ent] = sources

    # Prioritize entities: traceback > backtick > file_path > error_class > others
    def entity_priority(item):
        ent_str, sources_set = item
        priority = 0
        if "traceback_file" in sources_set:
            priority += 100
        if "traceback_func" in sources_set:
            priority += 80
        if "backtick" in sources_set:
            priority += 60
        if "file_path" in sources_set:
            priority += 50
        if "error_class" in sources_set:
            priority += 40
        if "code_block" in sources_set:
            priority += 30
        if "dotted" in sources_set:
            priority += 20
        return priority

    sorted_entities = sorted(filtered_sources.items(), key=entity_priority, reverse=True)
    return dict(sorted_entities[:50])


def _extract_entities(issue_text: str) -> list[str]:
    """Helper for backward compatibility returning list of extracted entities."""
    return list(_extract_entities_with_sources(issue_text).keys())


def _run_localization_pass(
    entities: list[str],
    entity_sources: dict[str, set[str]],
    graph,
    lsp_bridge,
    lsp_available: bool,
    repo_path: str,
    expand: bool = False
) -> tuple[dict[str, float], dict[str, list[str]], list[LocatedSymbol], list[str], set[str], set[str], int, int, list[str]]:
    symbols: list[LocatedSymbol] = []
    file_scores: dict[str, float] = {}
    file_reasons: dict[str, list[str]] = {}
    graph_hits = 0
    graph_misses = 0
    missed_entities: list[str] = []
    all_references: list[str] = []
    test_files: set[str] = set()
    dependency_neighbors: set[str] = set()

    def get_multiplier(entity: str) -> float:
        sources = entity_sources.get(entity, set())
        if "traceback_file" in sources or "traceback_func" in sources:
            return 2.5
        if "backtick" in sources:
            return 2.0
        if "file_path" in sources:
            return 1.8
        if "code_block" in sources:
            return 1.5
        if "error_class" in sources:
            return 1.3
        return 1.0

    for entity in entities:
        found = False
        mult = get_multiplier(entity)

        # Try as dotted name: split module.Class.method
        parts = entity.split(".")
        for part in parts:
            if len(part) < 2:
                continue

            # Class/method definition lookup
            ctx = graph.get_symbol_context(part)
            defn = ctx["definition"]
            if defn:
                found = True
                symbols.append(LocatedSymbol(
                    name=part,
                    kind=defn["kind"],
                    file_path=defn["file"],
                    line_number=defn["line"],
                    end_line=defn["end_line"],
                    parent_class=defn.get("parent_class", ""),
                    confidence=0.9,
                    source="graph",
                ))

                # Score the file
                fp = defn["file"]
                file_scores[fp] = file_scores.get(fp, 0.0) + (0.6 * mult)
                file_reasons.setdefault(fp, []).append(f"defines {part}")

                # Step 3: Callers and callees
                for caller in ctx["callers"]:
                    caller_file = caller.get("file", "")
                    if caller_file:
                        file_scores[caller_file] = file_scores.get(caller_file, 0.0) + (0.2 * mult)
                        file_reasons.setdefault(caller_file, []).append(f"calls {part}")
                        all_references.append(f"{caller['name']} → {part}")

                for callee in ctx["callees"]:
                    callee_file = callee.get("file", "")
                    if callee_file:
                        file_scores[callee_file] = file_scores.get(callee_file, 0.0) + (0.12 * mult)
                        file_reasons.setdefault(callee_file, []).append(f"called by {part}")

                # Step 4: Inheritance chain
                for impl in ctx.get("implementations", []):
                    impl_file = impl.get("file", "")
                    if impl_file:
                        file_scores[impl_file] = file_scores.get(impl_file, 0.0) + (0.2 * mult)
                        file_reasons.setdefault(impl_file, []).append(f"implements {part}")

                # Step 5: Tests
                for test_path in ctx["tests"]:
                    test_files.add(test_path)
                    file_scores[test_path] = file_scores.get(test_path, 0.0) + (0.08 * mult)
                    file_reasons.setdefault(test_path, []).append(f"tests {part}")

                # Step 6: Dependencies
                for dep in ctx["imports"]:
                    dependency_neighbors.add(dep)
                for dep in ctx["dependents"]:
                    dependency_neighbors.add(dep)

        # Try as file path
        if not found and ("/" in entity or entity.endswith((".py", ".js", ".ts"))):
            normalised = entity.replace("\\", "/").lstrip("./")
            if normalised in (graph.modules or {}):
                found = True
                file_scores[normalised] = file_scores.get(normalised, 0.0) + (0.8 * mult)
                file_reasons.setdefault(normalised, []).append("mentioned as file path")

        # Fuzzy symbol search as last resort
        if not found:
            limit = 8 if expand else 3
            fuzzy_results = graph.query_symbols(entity, limit=limit)
            for r in fuzzy_results:
                if r["name"].lower() == entity.lower() or (expand and entity.lower() in r["name"].lower()):
                    found = True
                    symbols.append(LocatedSymbol(
                        name=r["name"],
                        kind=r["kind"],
                        file_path=r["file"],
                        line_number=r.get("line", 0),
                        end_line=0,
                        parent_class=r.get("parent_class", ""),
                        confidence=0.7,
                        source="graph_fuzzy",
                    ))
                    fp = r["file"]
                    file_scores[fp] = file_scores.get(fp, 0.0) + (0.4 * mult)
                    file_reasons.setdefault(fp, []).append(f"fuzzy match for {entity}")

                    # Expanded fuzzy traversal
                    if expand:
                        try:
                            ctx = graph.get_symbol_context(r["name"])
                            for caller in ctx["callers"]:
                                caller_file = caller.get("file", "")
                                if caller_file:
                                    file_scores[caller_file] = file_scores.get(caller_file, 0.0) + (0.1 * mult)
                                    file_reasons.setdefault(caller_file, []).append(f"calls fuzzy {r['name']}")
                            for callee in ctx["callees"]:
                                callee_file = callee.get("file", "")
                                if callee_file:
                                    file_scores[callee_file] = file_scores.get(callee_file, 0.0) + (0.05 * mult)
                            for test_path in ctx["tests"]:
                                test_files.add(test_path)
                        except Exception:
                            pass
                    break

        # Fallback localization via smart_search (ripgrep) if expand is True
        if not found and expand:
            from issue_resolver.utils.ripgrep_search import smart_search
            if len(entity) >= 3:
                rg_results = smart_search(entity, repo_path)
                if rg_results:
                    found = True
                    for match in rg_results:
                        matched_file = match["file"].replace("\\", "/").replace(repo_path.replace("\\", "/").rstrip("/") + "/", "").lstrip("./")
                        file_scores[matched_file] = file_scores.get(matched_file, 0.0) + (0.35 * mult)
                        file_reasons.setdefault(matched_file, []).append(f"ripgrep fallback match for {entity}")
                        symbols.append(LocatedSymbol(
                            name=entity,
                            kind="function" if "def " in match["content"] or "function" in match["content"] else "generic",
                            file_path=matched_file,
                            line_number=match["line"],
                            end_line=0,
                            parent_class="",
                            confidence=0.6,
                            source="ripgrep",
                        ))

        # Step 2b: LSP-enhanced lookup
        if found and lsp_available and symbols:
            last_sym = symbols[-1]
            try:
                from issue_resolver.intelligence.lsp_tools import lsp_find_references
                lsp_refs = lsp_find_references(
                    last_sym.name, last_sym.file_path,
                    last_sym.line_number,
                )
                for ref in lsp_refs:
                    ref_file = ref.get("file", "")
                    if ref_file and ref_file not in file_scores:
                        file_scores[ref_file] = file_scores.get(ref_file, 0.0) + (0.12 * mult)
                        file_reasons.setdefault(ref_file, []).append(f"LSP reference to {last_sym.name}")
            except Exception:
                pass

        if found:
            graph_hits += 1
        else:
            graph_misses += 1
            missed_entities.append(entity)

    return file_scores, file_reasons, symbols, all_references, test_files, dependency_neighbors, graph_hits, graph_misses, missed_entities


def _localize(issue_text: str) -> LocalizationResult:
    """Run the full deterministic localization pipeline."""
    graph = runtime_context.get_knowledge_graph()
    lsp_bridge = runtime_context.get_lsp_bridge()
    lsp_available = lsp_bridge is not None and lsp_bridge.is_available
    
    # Resolve repo path centrally
    env_config = runtime_context.get_environment_config() or {}
    repo_path = env_config.get("repo_root", ".")

    # Step 1: Extract entities
    entity_sources = _extract_entities_with_sources(issue_text)
    entities = list(entity_sources.keys())
    if not entities:
        return LocalizationResult(
            primary_files=[], symbols=[], references=[], test_files=[],
            dependency_neighbors=[], confidence=0.0, entities_extracted=[],
            graph_hits=0, graph_misses=0, missed_entities=[],
            lsp_available=lsp_available, needs_researcher_fallback=True,
        )

    if graph is None:
        return LocalizationResult(
            primary_files=[], symbols=[], references=[], test_files=[],
            dependency_neighbors=[], confidence=0.0,
            entities_extracted=entities,
            graph_hits=0, graph_misses=len(entities), missed_entities=entities,
            lsp_available=lsp_available, needs_researcher_fallback=True,
        )

    # Pass 1: standard localization lookup
    file_scores, file_reasons, symbols, all_references, test_files, dependency_neighbors, graph_hits, graph_misses, missed_entities = \
        _run_localization_pass(entities, entity_sources, graph, lsp_bridge, lsp_available, repo_path, expand=False)

    total_entities = max(len(entities), 1)
    hit_rate = graph_hits / total_entities
    
    def compute_overall_confidence(hit_rate_val, symbols_list, file_scores_dict):
        exact_hit_count = sum(1 for s in symbols_list if s.confidence >= 0.9)
        traceback_hit_count = sum(1 for s in symbols_list if s.source in ("graph", "graph_fuzzy") and any("traceback" in src for src in entity_sources.get(s.name, [])))
        backtick_hit_count = sum(1 for s in symbols_list if any("backtick" in src for src in entity_sources.get(s.name, [])))
        file_path_hit_count = sum(1 for fp in file_scores_dict if any(fp.endswith(ent) or ent in fp for ent in entities if "file_path" in entity_sources.get(ent, [])))
        
        base_confidence = hit_rate_val * 0.4
        if exact_hit_count > 0:
            base_confidence += 0.25
        if traceback_hit_count > 0:
            base_confidence += 0.25
        if backtick_hit_count > 0:
            base_confidence += 0.15
        if file_path_hit_count > 0:
            base_confidence += 0.15
        return min(1.0, max(0.0, base_confidence))

    overall_confidence = compute_overall_confidence(hit_rate, symbols, file_scores)

    # Adaptive expansion pass if confidence below 0.70
    if overall_confidence < 0.70:
        print(f"[Localizer] Confidence {overall_confidence:.2f} < 0.70. Running expanded pass with graph traversal expansion and fallback localization...")
        file_scores, file_reasons, symbols, all_references, test_files, dependency_neighbors, graph_hits, graph_misses, missed_entities = \
            _run_localization_pass(entities, entity_sources, graph, lsp_bridge, lsp_available, repo_path, expand=True)
            
        hit_rate = graph_hits / total_entities
        overall_confidence = compute_overall_confidence(hit_rate, symbols, file_scores)
        print(f"[Localizer] Expanded pass complete. Confidence updated to: {overall_confidence:.2f}")

    if not file_scores:
        return LocalizationResult(
            primary_files=[], symbols=symbols, references=all_references,
            test_files=sorted(test_files),
            dependency_neighbors=sorted(dependency_neighbors),
            confidence=overall_confidence, entities_extracted=entities,
            graph_hits=graph_hits, graph_misses=graph_misses,
            missed_entities=missed_entities,
            lsp_available=lsp_available, needs_researcher_fallback=True,
        )

    # Normalise scores
    max_score = max(file_scores.values())
    normalised_scores = {
        p: s / max_score if max_score > 0 else 0.0
        for p, s in file_scores.items()
    }

    from issue_resolver.core.interfaces import Confidence
    primary_files = [
        ScoredFile(
            path=path,
            score=score,
            confidence=Confidence.from_score(score),
            reasons=file_reasons.get(path, []),
        )
        for path, score in sorted(normalised_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Needs fallback if overall confidence below 0.70
    needs_fallback = (
        overall_confidence < 0.70
        or len(primary_files) == 0
    )

    return LocalizationResult(
        primary_files=primary_files,
        symbols=symbols,
        references=all_references,
        test_files=sorted(test_files),
        dependency_neighbors=sorted(dependency_neighbors),
        confidence=overall_confidence,
        entities_extracted=entities,
        graph_hits=graph_hits,
        graph_misses=graph_misses,
        missed_entities=missed_entities,
        lsp_available=lsp_available,
        needs_researcher_fallback=needs_fallback,
    )


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def localizer_node(state: AgentState) -> dict:
    """Deterministic repository localization (primary path).

    Extracts entities from the issue text, looks them up in the
    knowledge graph and LSP, and returns structured localization results.
    """
    issue_text = state.get("issue", "")
    print("[Localizer] Running deterministic entity extraction and graph lookup...")

    result = _localize(issue_text)

    print(
        f"[Localizer] Extracted {len(result.entities_extracted)} entities, "
        f"{result.graph_hits} graph hits, {result.graph_misses} graph misses, "
        f"confidence={result.confidence:.2f}, "
        f"needs_fallback={result.needs_researcher_fallback}"
    )

    if result.primary_files:
        top_files = [f"{f.path} ({f.confidence.value})" for f in result.primary_files[:5]]
        print(f"[Localizer] Top files: {', '.join(top_files)}")

    if result.symbols:
        top_symbols = [f"{s.name} ({s.kind})" for s in result.symbols[:5]]
        print(f"[Localizer] Symbols found: {', '.join(top_symbols)}")

    # Record trace
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "localization_completed",
            "Localizer",
            f"Localized {result.graph_hits}/{len(result.entities_extracted)} entities "
            f"(confidence={result.confidence:.2f})",
            details={
                "entities_extracted": result.entities_extracted[:10],
                "graph_hits": result.graph_hits,
                "graph_misses": result.graph_misses,
                "primary_files": [f.path for f in result.primary_files[:10]],
                "confidence": result.confidence,
                "lsp_available": result.lsp_available,
                "needs_researcher_fallback": result.needs_researcher_fallback,
            },
        )

    return {
        "localization_result": result.to_dict(),
        "localization_confidence": result.confidence,
        "history": append_to_history(
            "Localizer",
            "Localize",
            f"Extracted {len(result.entities_extracted)} entities, "
            f"{result.graph_hits} graph hits, "
            f"confidence={result.confidence:.2f}, "
            f"fallback={'needed' if result.needs_researcher_fallback else 'not needed'}",
        ),
    }
