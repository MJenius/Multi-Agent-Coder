"""Build a ``RepoKnowledgeGraph`` from a repository on disk.

Phase 1 — static analysis using Python ``ast`` (no new deps for Python,
          tree-sitter for JS/TS/C# if available, regex fallback).
Phase 2 — dependency analysis from manifests.
Phase 3 — caching via ``MemoryStore``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from issue_resolver.intelligence.knowledge_graph import (
    CallEdge,
    ClassNode,
    DependencyEdge,
    FunctionNode,
    ImportEdge,
    InheritanceEdge,
    ModuleNode,
    RepoKnowledgeGraph,
)

# Directories to always skip (matches repo_tools.IGNORE_DIRS)
_IGNORE_DIRS = {
    ".git", ".gitignore", ".github",
    ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".tox", ".coverage", "node_modules", ".npm",
    "bin", "obj", ".vs", "packages",
    "build", "dist", "target", "htmlcov",
    ".idea", ".vscode", ".ruff_cache",
}

_LANG_MAP = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
}

_CONFIG_NAMES = {
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "package.json", "tsconfig.json", ".eslintrc", ".eslintrc.js", ".eslintrc.json",
    ".prettierrc", ".prettierrc.json", "webpack.config.js", "vite.config.ts",
    "Makefile", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env", ".env.example", "Cargo.toml", "go.mod", "go.sum",
    "Gemfile", "build.gradle", "pom.xml",
    ".gitignore", ".editorconfig",
}

_TEST_PATTERNS = [
    re.compile(r"test_[^/\\]+\.py$"),
    re.compile(r"[^/\\]+_test\.py$"),
    re.compile(r"[^/\\]+\.test\.[jt]sx?$"),
    re.compile(r"[^/\\]+\.spec\.[jt]sx?$"),
    re.compile(r"[^/\\]+Tests?\.cs$"),
    re.compile(r"[^/\\]+Test\.java$"),
]


def _is_test_file(path: str) -> bool:
    return any(p.search(path) for p in _TEST_PATTERNS)


def _is_entrypoint(path: str, content: str, language: str) -> bool:
    name = Path(path).name
    if language == "python":
        if name in ("main.py", "app.py", "manage.py", "cli.py"):
            return True
        if 'if __name__' in content and '__main__' in content:
            return True
    if language in ("javascript", "typescript"):
        if name in ("index.js", "index.ts", "main.js", "main.ts", "app.js", "app.ts", "server.js"):
            return True
    if language == "go" and "func main()" in content:
        return True
    return False


# ---------------------------------------------------------------------------
# Python AST analysis
# ---------------------------------------------------------------------------


class _PythonVisitor(ast.NodeVisitor):
    """Extract classes, functions, imports, and calls from Python AST."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.classes: list[ClassNode] = []
        self.functions: list[FunctionNode] = []
        self.imports: list[str] = []           # module paths imported
        self.import_edges: list[ImportEdge] = []
        self.inheritance_edges: list[InheritanceEdge] = []
        self.call_edges: list[CallEdge] = []
        self._current_class: str = ""
        self._current_func: str = ""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module:
            self.imports.append(module)
            symbols = [a.name for a in node.names]
            self.import_edges.append(ImportEdge(
                source=self.file_path,
                target=module,
                symbols=symbols,
            ))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                bases.append("?")

        methods = []
        attributes: list[str] = []
        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                pass

        docstring = ast.get_docstring(node) or ""

        cls_node = ClassNode(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            bases=bases,
            docstring=docstring[:200],
            decorators=decorators,
        )

        # Traverse methods
        old_class = self._current_class
        self._current_class = node.name

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
                self._visit_function(item, parent_class=node.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)

        cls_node.methods = methods
        cls_node.attributes = attributes
        self.classes.append(cls_node)

        # Inheritance edges
        for base in bases:
            self.inheritance_edges.append(InheritanceEdge(
                child=f"{self.file_path}::{node.name}",
                parent=base,
            ))

        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._current_class:
            self._visit_function(node)
        # (class methods are handled in visit_ClassDef)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_class: str = "",
    ) -> None:
        params = []
        for arg in node.args.args:
            params.append(arg.arg)

        return_type = ""
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except Exception:
                pass

        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                pass

        docstring = ast.get_docstring(node) or ""

        fn_node = FunctionNode(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            parameters=params,
            return_type=return_type,
            parent_class=parent_class,
            docstring=docstring[:200],
            decorators=decorators,
            is_method=bool(parent_class),
        )

        # Extract calls
        calls: list[str] = []
        old_func = self._current_func
        if parent_class:
            self._current_func = f"{self.file_path}::{parent_class}.{node.name}"
        else:
            self._current_func = f"{self.file_path}::{node.name}"

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = ""
                if isinstance(child.func, ast.Name):
                    call_name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    call_name = child.func.attr
                if call_name and call_name not in calls:
                    calls.append(call_name)
                    self.call_edges.append(CallEdge(
                        caller=self._current_func,
                        callee=call_name,
                    ))

        fn_node.calls = calls
        self.functions.append(fn_node)
        self._current_func = old_func


def _analyse_python_file(file_path: str, content: str) -> dict[str, Any]:
    """Parse a Python file and return extracted information."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"error": "syntax_error"}

    visitor = _PythonVisitor(file_path)
    visitor.visit(tree)

    exports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exports.append(elt.value)

    if not exports:
        exports = [c.name for c in visitor.classes] + [
            f.name for f in visitor.functions if not f.is_method and not f.name.startswith("_")
        ]

    return {
        "classes": visitor.classes,
        "functions": visitor.functions,
        "imports": visitor.imports,
        "import_edges": visitor.import_edges,
        "inheritance_edges": visitor.inheritance_edges,
        "call_edges": visitor.call_edges,
        "exports": exports,
        "docstring": ast.get_docstring(tree) or "",
    }


# ---------------------------------------------------------------------------
# Regex fallback for non-Python languages
# ---------------------------------------------------------------------------


def _analyse_generic_file(file_path: str, content: str, language: str) -> dict[str, Any]:
    """Regex-based extraction for JS/TS/C#/Go/etc.  Less precise than AST."""
    classes: list[ClassNode] = []
    functions: list[FunctionNode] = []
    imports: list[str] = []

    lines = content.split("\n")

    if language in ("javascript", "typescript"):
        # Imports
        for line in lines:
            m = re.match(r"""^\s*import\s+.*?from\s+['"]([^'"]+)""", line)
            if m:
                imports.append(m.group(1))
            m = re.match(r"""^\s*(?:const|let|var)\s+.*?=\s*require\s*\(\s*['"]([^'"]+)""", line)
            if m:
                imports.append(m.group(1))

        # Classes
        for i, line in enumerate(lines):
            m = re.match(r"^\s*(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?", line)
            if m:
                classes.append(ClassNode(
                    name=m.group(1), file_path=file_path, line_number=i + 1,
                    bases=[m.group(2)] if m.group(2) else [],
                ))

        # Functions
        for i, line in enumerate(lines):
            m = re.match(
                r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
                line,
            )
            if m:
                functions.append(FunctionNode(
                    name=m.group(1), file_path=file_path, line_number=i + 1,
                    parameters=[p.strip().split(":")[0].strip() for p in m.group(2).split(",") if p.strip()],
                ))
            # Arrow functions assigned to const
            m = re.match(
                r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(",
                line,
            )
            if m:
                functions.append(FunctionNode(
                    name=m.group(1), file_path=file_path, line_number=i + 1,
                ))

    elif language == "csharp":
        for line in lines:
            m = re.match(r"^\s*using\s+([\w.]+)\s*;", line)
            if m:
                imports.append(m.group(1))
        for i, line in enumerate(lines):
            m = re.match(r"^\s*(?:public|internal|private|protected)?\s*(?:abstract|sealed|static)?\s*class\s+(\w+)(?:\s*:\s*(\w+))?", line)
            if m:
                classes.append(ClassNode(
                    name=m.group(1), file_path=file_path, line_number=i + 1,
                    bases=[m.group(2)] if m.group(2) else [],
                ))
        for i, line in enumerate(lines):
            m = re.match(
                r"^\s*(?:public|private|protected|internal|static|async|virtual|override|abstract)*\s*"
                r"(?:[\w<>\[\]?]+)\s+(\w+)\s*\(([^)]*)\)",
                line,
            )
            if m and m.group(1) not in ("if", "while", "for", "switch", "catch"):
                functions.append(FunctionNode(
                    name=m.group(1), file_path=file_path, line_number=i + 1,
                ))

    elif language == "go":
        for i, line in enumerate(lines):
            m = re.match(r'^\s*import\s+"([^"]+)"', line)
            if m:
                imports.append(m.group(1))
            m = re.match(r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", line)
            if m:
                functions.append(FunctionNode(
                    name=m.group(1), file_path=file_path, line_number=i + 1,
                ))

    return {
        "classes": classes,
        "functions": functions,
        "imports": imports,
        "import_edges": [ImportEdge(source=file_path, target=imp) for imp in imports],
        "inheritance_edges": [
            InheritanceEdge(child=f"{file_path}::{c.name}", parent=b)
            for c in classes for b in c.bases
        ],
        "call_edges": [],
        "exports": [c.name for c in classes] + [f.name for f in functions],
        "docstring": "",
    }


# ---------------------------------------------------------------------------
# Dependency analysis
# ---------------------------------------------------------------------------


def _parse_python_deps(repo_root: Path) -> list[DependencyEdge]:
    deps: list[DependencyEdge] = []

    # pyproject.toml
    pyproject = repo_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            content = pyproject.read_text(encoding="utf-8", errors="replace")
            # Simple regex parsing (avoids toml dependency)
            in_deps = False
            in_dev = False
            for line in content.split("\n"):
                if re.match(r"\[project\]", line) or "dependencies" in line:
                    in_deps = "dependencies" in line
                    in_dev = "dev" in line or "optional" in line
                if in_deps:
                    m = re.match(r'\s*"([a-zA-Z0-9_-]+)', line)
                    if m:
                        deps.append(DependencyEdge(package=m.group(1), is_dev=in_dev))
                if line.strip() == "]":
                    in_deps = False
        except OSError:
            pass

    # requirements.txt
    req_txt = repo_root / "requirements.txt"
    if req_txt.is_file():
        try:
            for line in req_txt.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    m = re.match(r"([a-zA-Z0-9_-]+)", line)
                    if m:
                        version = line[len(m.group(1)):].strip()
                        deps.append(DependencyEdge(package=m.group(1), version_spec=version))
        except OSError:
            pass

    return deps


def _parse_node_deps(repo_root: Path) -> list[DependencyEdge]:
    deps: list[DependencyEdge] = []
    pkg_json = repo_root / "package.json"
    if not pkg_json.is_file():
        return deps
    try:
        data = json.loads(pkg_json.read_text(encoding="utf-8", errors="replace"))
        for name, version in data.get("dependencies", {}).items():
            deps.append(DependencyEdge(package=name, version_spec=version))
        for name, version in data.get("devDependencies", {}).items():
            deps.append(DependencyEdge(package=name, version_spec=version, is_dev=True))
    except (json.JSONDecodeError, OSError):
        pass
    return deps


def _parse_dotnet_deps(repo_root: Path) -> list[DependencyEdge]:
    deps: list[DependencyEdge] = []
    for csproj in repo_root.rglob("*.csproj"):
        try:
            content = csproj.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r'<PackageReference\s+Include="([^"]+)"(?:\s+Version="([^"]*)")?', content):
                deps.append(DependencyEdge(package=m.group(1), version_spec=m.group(2) or ""))
        except OSError:
            pass
    return deps


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


class GraphBuilder:
    """Constructs a ``RepoKnowledgeGraph`` from a repository path.

    Usage::

        builder = GraphBuilder(repo_path="./sandbox_workspace")
        graph = builder.build()
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_root = Path(repo_path).resolve()
        self.graph = RepoKnowledgeGraph()

    def build(self) -> RepoKnowledgeGraph:
        """Run all analysis phases and return the populated graph."""
        print(f"[GraphBuilder] Building knowledge graph for {self.repo_root}")

        # Phase 1: Walk and analyse files
        self._walk_and_analyse()

        # Phase 2: Parse dependencies
        self._parse_dependencies()

        print(
            f"[GraphBuilder] Complete: {len(self.graph.modules)} modules, "
            f"{len(self.graph.classes)} classes, {len(self.graph.functions)} functions, "
            f"{len(self.graph.import_edges)} imports, {len(self.graph.call_edges)} calls"
        )
        return self.graph

    def _should_skip(self, path: Path) -> bool:
        parts = path.relative_to(self.repo_root).parts
        return any(part in _IGNORE_DIRS or part.startswith(".") for part in parts)

    def _walk_and_analyse(self) -> None:
        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            # Prune ignored directories
            dirnames[:] = [
                d for d in dirnames
                if d not in _IGNORE_DIRS and not d.startswith(".")
            ]

            for filename in sorted(filenames):
                file_path = Path(dirpath) / filename
                if self._should_skip(file_path):
                    continue

                suffix = file_path.suffix.lower()
                language = _LANG_MAP.get(suffix, "")
                rel_path = file_path.relative_to(self.repo_root).as_posix()

                # Config files
                is_config = filename in _CONFIG_NAMES or suffix in (".toml", ".yaml", ".yml", ".ini", ".cfg")
                is_test = _is_test_file(rel_path)

                if not language and not is_config:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                line_count = content.count("\n") + 1
                is_entry = _is_entrypoint(rel_path, content, language)

                # Create module node
                module = ModuleNode(
                    path=rel_path,
                    language=language,
                    size_bytes=file_path.stat().st_size,
                    line_count=line_count,
                    is_test=is_test,
                    is_config=is_config,
                    is_entrypoint=is_entry,
                )

                # Language-specific analysis
                if language == "python":
                    analysis = _analyse_python_file(rel_path, content)
                elif language:
                    analysis = _analyse_generic_file(rel_path, content, language)
                else:
                    analysis = {}

                if "error" not in analysis:
                    module.imports = analysis.get("imports", [])
                    module.exports = analysis.get("exports", [])
                    module.classes = [c.name for c in analysis.get("classes", [])]
                    module.functions = [f.name for f in analysis.get("functions", []) if not f.is_method]
                    module.docstring = analysis.get("docstring", "")[:200]

                    # Add nodes and edges
                    for cls in analysis.get("classes", []):
                        self.graph.classes[cls.qualified_name] = cls
                    for fn in analysis.get("functions", []):
                        self.graph.functions[fn.qualified_name] = fn
                    self.graph.import_edges.extend(analysis.get("import_edges", []))
                    self.graph.inheritance_edges.extend(analysis.get("inheritance_edges", []))
                    self.graph.call_edges.extend(analysis.get("call_edges", []))

                self.graph.modules[rel_path] = module

    def _parse_dependencies(self) -> None:
        """Parse package-level dependencies from manifest files."""
        self.graph.dependencies.extend(_parse_python_deps(self.repo_root))
        self.graph.dependencies.extend(_parse_node_deps(self.repo_root))
        self.graph.dependencies.extend(_parse_dotnet_deps(self.repo_root))

    def get_repo_hash(self) -> str:
        """Compute a hash of the repo state for cache invalidation."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.repo_root),
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Fallback: hash of file listing
        file_list = sorted(str(p) for p in self.repo_root.rglob("*") if p.is_file())
        return hashlib.sha256("\n".join(file_list[:500]).encode()).hexdigest()[:16]
