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


def _extract_entities(issue_text: str) -> list[str]:
    """Extract likely code entities from the issue text using regex.

    Captures:
      - Backtick-quoted identifiers: `calculate_total`
      - Code block identifiers: function/class names in ```python blocks
      - File paths: src/utils.py, foo/bar.js
      - camelCase, snake_case, PascalCase tokens
      - Dotted names: module.Class.method
      - Error class names: TypeError, ValueError
      - String literals from tracebacks
    """
    entities: list[str] = []

    # 1. Backtick-quoted identifiers (highest signal)
    backtick = re.findall(r"`([^`\n]{2,80})`", issue_text)
    for m in backtick:
        # Strip function call parens: `foo()` → foo
        clean = m.split("(")[0].strip()
        if clean and not clean.startswith("http"):
            entities.append(clean)

    # 2. Code blocks — extract function/class definitions
    code_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)\n```", issue_text, re.DOTALL)
    for block in code_blocks:
        # Python: def foo(...), class Bar
        entities.extend(re.findall(r"(?:def|class)\s+(\w+)", block))
        # JS/TS: function foo, const foo = , export function foo
        entities.extend(re.findall(r"(?:function|const|let|var|export)\s+(\w+)", block))

    # 3. File paths
    entities.extend(re.findall(
        r"([a-zA-Z_]\w*(?:/[a-zA-Z_]\w*)*\.(?:py|js|ts|go|rs|java|cpp|c|h|jsx|tsx|vue|rb))",
        issue_text,
    ))

    # 4. PascalCase class names (e.g. TableTitle, HttpClient)
    entities.extend(re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", issue_text))

    # 5. snake_case identifiers
    entities.extend(re.findall(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b", issue_text))

    # 6. camelCase identifiers
    entities.extend(re.findall(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", issue_text))

    # 7. Dotted names (module.Class.method)
    entities.extend(re.findall(
        r"\b([a-zA-Z_]\w+\.[a-zA-Z_]\w+(?:\.[a-zA-Z_]\w+)*)\b",
        issue_text,
    ))

    # 8. Error class names
    entities.extend(re.findall(r"\b([A-Z][a-z]+(?:Error|Exception|Warning|Fault))\b", issue_text))

    # 9. Traceback file references: File "foo/bar.py", line 42
    entities.extend(re.findall(r'File "([^"]+)"', issue_text))

    # 10. Traceback function names: in calculate_total
    entities.extend(re.findall(r"^\s+in (\w+)\s*$", issue_text, re.MULTILINE))

    # Deduplicate preserving order, filter noise
    seen: set[str] = set()
    unique: list[str] = []
    noise = {"the", "this", "that", "with", "from", "import", "and", "for", "not", "but",
             "def", "class", "return", "none", "true", "false", "self", "cls"}
    for ent in entities:
        lower = ent.lower()
        if lower not in seen and len(ent) >= 2 and lower not in noise:
            seen.add(lower)
            unique.append(ent)
    return unique[:50]


# ---------------------------------------------------------------------------
# Localization pipeline
# ---------------------------------------------------------------------------


def _localize(issue_text: str) -> LocalizationResult:
    """Run the full deterministic localization pipeline."""
    graph = runtime_context.get_knowledge_graph()
    lsp_bridge = runtime_context.get_lsp_bridge()
    lsp_available = lsp_bridge is not None and lsp_bridge.is_available

    # Step 1: Extract entities
    entities = _extract_entities(issue_text)
    if not entities:
        return LocalizationResult(
            primary_files=[], symbols=[], references=[], test_files=[],
            dependency_neighbors=[], confidence=0.0, entities_extracted=[],
            graph_hits=0, graph_misses=0, missed_entities=[],
            lsp_available=lsp_available, needs_researcher_fallback=True,
        )

    # Step 2: Graph lookup
    symbols: list[LocatedSymbol] = []
    file_scores: dict[str, float] = {}
    file_reasons: dict[str, list[str]] = {}
    graph_hits = 0
    graph_misses = 0
    missed_entities: list[str] = []
    all_references: list[str] = []
    test_files: set[str] = set()
    dependency_neighbors: set[str] = set()

    if graph is None:
        return LocalizationResult(
            primary_files=[], symbols=[], references=[], test_files=[],
            dependency_neighbors=[], confidence=0.0,
            entities_extracted=entities,
            graph_hits=0, graph_misses=len(entities), missed_entities=entities,
            lsp_available=lsp_available, needs_researcher_fallback=True,
        )

    for entity in entities:
        found = False

        # Try as dotted name: split module.Class.method
        parts = entity.split(".")
        for part in parts:
            if len(part) < 2:
                continue

            # Step 2a: Class lookup
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
                file_scores[fp] = file_scores.get(fp, 0.0) + 0.4
                file_reasons.setdefault(fp, []).append(f"defines {part}")

                # Step 3: Callers and callees
                for caller in ctx["callers"]:
                    caller_file = caller.get("file", "")
                    if caller_file:
                        file_scores[caller_file] = file_scores.get(caller_file, 0.0) + 0.15
                        file_reasons.setdefault(caller_file, []).append(f"calls {part}")
                        all_references.append(f"{caller['name']} → {part}")

                for callee in ctx["callees"]:
                    callee_file = callee.get("file", "")
                    if callee_file:
                        file_scores[callee_file] = file_scores.get(callee_file, 0.0) + 0.1
                        file_reasons.setdefault(callee_file, []).append(f"called by {part}")

                # Step 4: Inheritance chain
                for impl in ctx.get("implementations", []):
                    impl_file = impl.get("file", "")
                    if impl_file:
                        file_scores[impl_file] = file_scores.get(impl_file, 0.0) + 0.15
                        file_reasons.setdefault(impl_file, []).append(f"implements {part}")

                # Step 5: Tests
                for test_path in ctx["tests"]:
                    test_files.add(test_path)
                    file_scores[test_path] = file_scores.get(test_path, 0.0) + 0.05
                    file_reasons.setdefault(test_path, []).append(f"tests {part}")

                # Step 6: Dependencies
                for dep in ctx["imports"]:
                    dependency_neighbors.add(dep)
                for dep in ctx["dependents"]:
                    dependency_neighbors.add(dep)

        # Try as file path
        if not found and "/" in entity or entity.endswith((".py", ".js", ".ts")):
            normalised = entity.replace("\\", "/").lstrip("./")
            if normalised in (graph.modules or {}):
                found = True
                file_scores[normalised] = file_scores.get(normalised, 0.0) + 0.5
                file_reasons.setdefault(normalised, []).append("mentioned as file path")

        # Fuzzy symbol search as last resort
        if not found:
            fuzzy_results = graph.query_symbols(entity, limit=3)
            for r in fuzzy_results:
                if r["name"].lower() == entity.lower():
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
                    file_scores[fp] = file_scores.get(fp, 0.0) + 0.25
                    file_reasons.setdefault(fp, []).append(f"fuzzy match for {entity}")
                    break

        # Step 2b: LSP-enhanced lookup (when available and graph found a location)
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
                        file_scores[ref_file] = file_scores.get(ref_file, 0.0) + 0.1
                        file_reasons.setdefault(ref_file, []).append(f"LSP reference to {last_sym.name}")
            except Exception:
                pass

        if found:
            graph_hits += 1
        else:
            graph_misses += 1
            missed_entities.append(entity)

    # Step 7: Score and rank files
    if not file_scores:
        return LocalizationResult(
            primary_files=[], symbols=symbols, references=all_references,
            test_files=sorted(test_files),
            dependency_neighbors=sorted(dependency_neighbors),
            confidence=0.0, entities_extracted=entities,
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

    primary_files = [
        ScoredFile(
            path=path,
            score=score,
            confidence=Confidence.from_score(score),
            reasons=file_reasons.get(path, []),
        )
        for path, score in sorted(normalised_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Step 8: Compute overall confidence
    total_entities = max(len(entities), 1)
    hit_rate = graph_hits / total_entities
    max_file_score = primary_files[0].score if primary_files else 0.0
    overall_confidence = (hit_rate * 0.6) + (max_file_score * 0.4)

    # Step 9: Determine if researcher fallback is needed
    needs_fallback = (
        overall_confidence < 0.4
        or (graph_misses / total_entities) > 0.4
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
