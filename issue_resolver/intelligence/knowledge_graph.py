"""Repository Knowledge Graph — the central intelligence data structure.

All downstream agents query this graph instead of repeatedly searching
the repository independently.  The graph contains modules, classes,
functions, imports, inheritance, call graph, dependency graph,
entrypoints, tests, configuration, and package relationships.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Graph node types
# ---------------------------------------------------------------------------


@dataclass
class ModuleNode:
    """A file / module in the repository."""

    path: str                        # relative to repo root
    language: str = ""               # python, javascript, csharp, ...
    size_bytes: int = 0
    line_count: int = 0
    imports: list[str] = field(default_factory=list)     # modules this imports
    exports: list[str] = field(default_factory=list)     # symbols this exports
    classes: list[str] = field(default_factory=list)     # class names defined here
    functions: list[str] = field(default_factory=list)   # top-level function names
    is_test: bool = False
    is_config: bool = False
    is_entrypoint: bool = False
    docstring: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "language": self.language,
            "size_bytes": self.size_bytes, "line_count": self.line_count,
            "imports": self.imports, "exports": self.exports,
            "classes": self.classes, "functions": self.functions,
            "is_test": self.is_test, "is_config": self.is_config,
            "is_entrypoint": self.is_entrypoint, "docstring": self.docstring,
        }


@dataclass
class ClassNode:
    """A class defined in the repository."""

    name: str
    file_path: str
    line_number: int = 0
    end_line: int = 0
    bases: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.file_path}::{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "file_path": self.file_path,
            "line_number": self.line_number, "end_line": self.end_line,
            "bases": self.bases, "methods": self.methods,
            "attributes": self.attributes, "docstring": self.docstring,
            "decorators": self.decorators,
        }


@dataclass
class FunctionNode:
    """A function or method defined in the repository."""

    name: str
    file_path: str
    line_number: int = 0
    end_line: int = 0
    parameters: list[str] = field(default_factory=list)
    return_type: str = ""
    parent_class: str = ""          # empty for top-level functions
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)        # functions this calls
    is_method: bool = False

    @property
    def qualified_name(self) -> str:
        if self.parent_class:
            return f"{self.file_path}::{self.parent_class}.{self.name}"
        return f"{self.file_path}::{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "file_path": self.file_path,
            "line_number": self.line_number, "end_line": self.end_line,
            "parameters": self.parameters, "return_type": self.return_type,
            "parent_class": self.parent_class, "docstring": self.docstring,
            "decorators": self.decorators, "calls": self.calls,
            "is_method": self.is_method,
        }


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------


@dataclass
class ImportEdge:
    """Module A imports from Module B."""

    source: str  # importing module path
    target: str  # imported module path
    symbols: list[str] = field(default_factory=list)


@dataclass
class InheritanceEdge:
    """Class A inherits from Class B."""

    child: str   # child class qualified name
    parent: str  # parent class qualified name


@dataclass
class CallEdge:
    """Function A calls Function B."""

    caller: str  # caller qualified name
    callee: str  # callee qualified name


@dataclass
class DependencyEdge:
    """Package-level dependency (from requirements/package.json/etc.)."""

    package: str
    version_spec: str = ""
    is_dev: bool = False


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------


class RepoKnowledgeGraph:
    """First-class repository knowledge graph.

    All downstream agents query this graph for repository understanding
    instead of independently searching files.

    Example usage::

        graph = RepoKnowledgeGraph()
        # ... populated by GraphBuilder ...

        module = graph.get_module("src/utils.py")
        callers = graph.get_callers("calculate_total")
        tests = graph.get_tests_for("src/utils.py")
        blast = graph.get_affected_modules(["src/utils.py"])
    """

    def __init__(self) -> None:
        self.modules: dict[str, ModuleNode] = {}
        self.classes: dict[str, ClassNode] = {}      # keyed by qualified_name
        self.functions: dict[str, FunctionNode] = {}  # keyed by qualified_name
        self.import_edges: list[ImportEdge] = []
        self.inheritance_edges: list[InheritanceEdge] = []
        self.call_edges: list[CallEdge] = []
        self.dependencies: list[DependencyEdge] = []

        # Caches built lazily
        self._caller_index: dict[str, list[str]] | None = None
        self._callee_index: dict[str, list[str]] | None = None
        self._import_from_index: dict[str, list[str]] | None = None
        self._import_to_index: dict[str, list[str]] | None = None

    # ----- module queries -----

    def get_module(self, path: str) -> ModuleNode | None:
        normalised = path.replace("\\", "/").lstrip("./")
        return self.modules.get(normalised)

    def get_all_modules(self) -> list[ModuleNode]:
        return list(self.modules.values())

    def get_test_modules(self) -> list[ModuleNode]:
        return [m for m in self.modules.values() if m.is_test]

    def get_config_files(self) -> list[str]:
        return [m.path for m in self.modules.values() if m.is_config]

    def get_entrypoints(self) -> list[str]:
        return [m.path for m in self.modules.values() if m.is_entrypoint]

    # ----- class queries -----

    def get_class(self, name: str) -> ClassNode | None:
        # Try qualified name first, then simple name
        if name in self.classes:
            return self.classes[name]
        for qn, cls in self.classes.items():
            if cls.name == name:
                return cls
        return None

    def get_classes_in_file(self, path: str) -> list[ClassNode]:
        normalised = path.replace("\\", "/").lstrip("./")
        return [c for c in self.classes.values() if c.file_path == normalised]

    def get_inheritance_chain(self, class_name: str) -> list[str]:
        """Return ordered list of base classes (MRO-like)."""
        chain: list[str] = []
        visited: set[str] = set()
        queue = [class_name]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            chain.append(current)
            for edge in self.inheritance_edges:
                if edge.child == current or edge.child.endswith(f"::{current}"):
                    parent = edge.parent
                    if parent not in visited:
                        queue.append(parent)
        return chain

    # ----- function queries -----

    def get_function(self, name: str) -> FunctionNode | None:
        if name in self.functions:
            return self.functions[name]
        for qn, fn in self.functions.items():
            if fn.name == name:
                return fn
        return None

    def get_functions_in_file(self, path: str) -> list[FunctionNode]:
        normalised = path.replace("\\", "/").lstrip("./")
        return [f for f in self.functions.values() if f.file_path == normalised]

    # ----- call graph queries -----

    def _build_call_indices(self) -> None:
        if self._caller_index is not None:
            return
        self._caller_index = {}
        self._callee_index = {}
        for edge in self.call_edges:
            self._callee_index.setdefault(edge.caller, []).append(edge.callee)
            self._caller_index.setdefault(edge.callee, []).append(edge.caller)

    def get_callers(self, symbol: str) -> list[str]:
        """Return symbols that call *symbol*."""
        self._build_call_indices()
        assert self._caller_index is not None
        # Try exact match then suffix match
        if symbol in self._caller_index:
            return self._caller_index[symbol]
        for key, callers in self._caller_index.items():
            if key.endswith(f"::{symbol}") or key.endswith(f".{symbol}"):
                return callers
        return []

    def get_callees(self, symbol: str) -> list[str]:
        """Return symbols that *symbol* calls."""
        self._build_call_indices()
        assert self._callee_index is not None
        if symbol in self._callee_index:
            return self._callee_index[symbol]
        for key, callees in self._callee_index.items():
            if key.endswith(f"::{symbol}") or key.endswith(f".{symbol}"):
                return callees
        return []

    # ----- import/dependency queries -----

    def _build_import_indices(self) -> None:
        if self._import_from_index is not None:
            return
        self._import_from_index = {}  # module -> what it imports
        self._import_to_index = {}    # module -> what imports it
        for edge in self.import_edges:
            self._import_from_index.setdefault(edge.source, []).append(edge.target)
            self._import_to_index.setdefault(edge.target, []).append(edge.source)

    def get_dependencies(self, module_path: str) -> list[str]:
        """Modules that *module_path* imports."""
        self._build_import_indices()
        assert self._import_from_index is not None
        normalised = module_path.replace("\\", "/").lstrip("./")
        return self._import_from_index.get(normalised, [])

    def get_dependents(self, module_path: str) -> list[str]:
        """Modules that import *module_path*."""
        self._build_import_indices()
        assert self._import_to_index is not None
        normalised = module_path.replace("\\", "/").lstrip("./")
        return self._import_to_index.get(normalised, [])

    def get_package_relationships(self) -> list[DependencyEdge]:
        return list(self.dependencies)

    # ----- blast radius -----

    def get_affected_modules(self, changed_files: list[str]) -> list[str]:
        """Estimate the blast radius of changes to *changed_files*.

        Returns all modules transitively dependent on the changed files.
        """
        self._build_import_indices()
        assert self._import_to_index is not None

        affected: set[str] = set()
        queue = list({f.replace("\\", "/").lstrip("./") for f in changed_files})
        while queue:
            current = queue.pop(0)
            if current in affected:
                continue
            affected.add(current)
            for dep in self._import_to_index.get(current, []):
                if dep not in affected:
                    queue.append(dep)
        return sorted(affected)

    # ----- test mapping -----

    def get_tests_for(self, module_path: str) -> list[str]:
        """Find test files that likely test *module_path*."""
        normalised = module_path.replace("\\", "/").lstrip("./")
        stem = Path(normalised).stem
        results: list[str] = []

        for mod in self.modules.values():
            if not mod.is_test:
                continue
            test_stem = Path(mod.path).stem
            # Convention: test_foo.py tests foo.py
            if test_stem == f"test_{stem}" or test_stem == f"{stem}_test":
                results.append(mod.path)
            # Check if test imports the module
            if normalised in mod.imports:
                results.append(mod.path)

        return sorted(set(results))

    # ----- symbol search -----

    def query_symbols(self, pattern: str, limit: int = 20) -> list[dict[str, Any]]:
        """Fuzzy symbol search across classes and functions."""
        pattern_lower = pattern.lower()
        results: list[dict[str, Any]] = []

        for cls in self.classes.values():
            if pattern_lower in cls.name.lower():
                results.append({
                    "kind": "class", "name": cls.name,
                    "file": cls.file_path, "line": cls.line_number,
                })

        for fn in self.functions.values():
            if pattern_lower in fn.name.lower():
                results.append({
                    "kind": "function", "name": fn.name,
                    "file": fn.file_path, "line": fn.line_number,
                    "parent_class": fn.parent_class,
                })

        # Sort by relevance: exact prefix match > contains
        results.sort(key=lambda r: (
            0 if r["name"].lower().startswith(pattern_lower) else 1,
            len(r["name"]),
        ))
        return results[:limit]

    # ----- serialisation -----

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a compact summary suitable for LLM context injection."""
        return {
            "total_modules": len(self.modules),
            "total_classes": len(self.classes),
            "total_functions": len(self.functions),
            "total_import_edges": len(self.import_edges),
            "total_inheritance_edges": len(self.inheritance_edges),
            "total_call_edges": len(self.call_edges),
            "total_dependencies": len(self.dependencies),
            "entrypoints": self.get_entrypoints(),
            "test_modules": [m.path for m in self.get_test_modules()],
            "config_files": self.get_config_files(),
            "modules": {
                path: {
                    "language": mod.language,
                    "classes": mod.classes,
                    "functions": mod.functions,
                    "is_test": mod.is_test,
                }
                for path, mod in sorted(self.modules.items())
            },
        }

    def to_text_summary(self, max_lines: int = 200) -> str:
        """Return a human-readable summary for LLM prompts."""
        lines: list[str] = [
            f"Repository Knowledge Graph: {len(self.modules)} modules, "
            f"{len(self.classes)} classes, {len(self.functions)} functions",
            "",
        ]

        entries = sorted(self.modules.items())
        if self.get_entrypoints():
            lines.append(f"Entrypoints: {', '.join(self.get_entrypoints())}")
        if self.get_test_modules():
            lines.append(f"Test modules: {', '.join(m.path for m in self.get_test_modules())}")
        if self.get_config_files():
            lines.append(f"Config files: {', '.join(self.get_config_files())}")
        lines.append("")

        for path, mod in entries:
            if len(lines) >= max_lines:
                lines.append(f"... [{len(entries) - len(lines)} more modules]")
                break
            parts = [f"  {path} ({mod.language})"]
            if mod.classes:
                parts.append(f"    classes: {', '.join(mod.classes)}")
            if mod.functions:
                parts.append(f"    functions: {', '.join(mod.functions[:10])}")
                if len(mod.functions) > 10:
                    parts[-1] += f" +{len(mod.functions) - 10} more"
            lines.extend(parts)

        return "\n".join(lines)
