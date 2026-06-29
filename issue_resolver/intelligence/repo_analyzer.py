"""Deep repository analysis beyond the knowledge graph.

Detects framework, architecture pattern, coding conventions, naming
conventions, testing style, formatter, linter, CI system, and
complexity estimate.  Uses static detection first, then LLM-assisted
analysis when static detection is insufficient.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from issue_resolver.intelligence.knowledge_graph import RepoKnowledgeGraph


@dataclass
class RepoProfile:
    """Structured output of deep repository analysis."""

    # Core identity
    primary_language: str = ""
    secondary_languages: list[str] = field(default_factory=list)
    framework: str = ""
    architecture_pattern: str = ""

    # Conventions
    naming_style: str = ""        # camelCase, snake_case, PascalCase
    docstring_style: str = ""     # google, numpy, sphinx, jsdoc
    import_ordering: str = ""     # stdlib-first, grouped, alphabetical

    # Tooling
    formatter: str = ""           # black, prettier, gofmt, rustfmt
    linter: str = ""              # ruff, eslint, golint, clippy
    type_checker: str = ""        # mypy, tsc, pyright
    ci_system: str = ""           # github_actions, gitlab_ci, jenkins, circleci
    package_manager: str = ""     # pip/uv, npm/yarn/pnpm, cargo, go mod

    # Testing
    test_framework: str = ""      # pytest, unittest, jest, vitest, xunit
    test_convention: str = ""     # test_*.py, *.test.ts, etc.
    test_directory: str = ""
    uses_fixtures: bool = False
    uses_mocks: bool = False

    # Build
    build_command: str = ""
    test_command: str = ""
    dev_command: str = ""

    # Complexity
    total_files: int = 0
    total_lines: int = 0
    total_symbols: int = 0
    max_module_lines: int = 0
    dependency_depth: int = 0

    # Risk areas
    complex_modules: list[str] = field(default_factory=list)
    high_churn_files: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_language": self.primary_language,
            "secondary_languages": self.secondary_languages,
            "framework": self.framework,
            "architecture_pattern": self.architecture_pattern,
            "naming_style": self.naming_style,
            "docstring_style": self.docstring_style,
            "import_ordering": self.import_ordering,
            "formatter": self.formatter,
            "linter": self.linter,
            "type_checker": self.type_checker,
            "ci_system": self.ci_system,
            "package_manager": self.package_manager,
            "test_framework": self.test_framework,
            "test_convention": self.test_convention,
            "test_directory": self.test_directory,
            "uses_fixtures": self.uses_fixtures,
            "uses_mocks": self.uses_mocks,
            "build_command": self.build_command,
            "test_command": self.test_command,
            "dev_command": self.dev_command,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "total_symbols": self.total_symbols,
            "max_module_lines": self.max_module_lines,
            "dependency_depth": self.dependency_depth,
            "complex_modules": self.complex_modules,
            "high_churn_files": self.high_churn_files,
            "metadata": self.metadata,
        }

    def to_prompt_context(self) -> str:
        """Format as context for LLM prompts."""
        lines = ["## Repository Profile"]
        if self.primary_language:
            lines.append(f"- Language: {self.primary_language}")
        if self.framework:
            lines.append(f"- Framework: {self.framework}")
        if self.architecture_pattern:
            lines.append(f"- Architecture: {self.architecture_pattern}")
        if self.naming_style:
            lines.append(f"- Naming: {self.naming_style}")
        if self.formatter:
            lines.append(f"- Formatter: {self.formatter}")
        if self.linter:
            lines.append(f"- Linter: {self.linter}")
        if self.test_framework:
            lines.append(f"- Test framework: {self.test_framework}")
        if self.test_command:
            lines.append(f"- Test command: {self.test_command}")
        if self.ci_system:
            lines.append(f"- CI: {self.ci_system}")
        lines.append(f"- Size: {self.total_files} files, {self.total_lines} lines, {self.total_symbols} symbols")
        if self.complex_modules:
            lines.append(f"- Complex modules: {', '.join(self.complex_modules[:5])}")
        return "\n".join(lines)


class RepoAnalyzer:
    """Analyses a repository to produce a ``RepoProfile``.

    First attempts static detection.  Falls back to LLM analysis
    only when static detection is insufficient.

    Usage::

        analyser = RepoAnalyzer(repo_root, graph)
        profile = analyser.analyse()
    """

    def __init__(self, repo_root: str, graph: RepoKnowledgeGraph) -> None:
        self.root = Path(repo_root).resolve()
        self.graph = graph
        self.profile = RepoProfile()

    def analyse(self) -> RepoProfile:
        """Run all analysis stages."""
        self._detect_languages()
        self._detect_framework()
        self._detect_tooling()
        self._detect_testing()
        self._detect_ci()
        self._detect_conventions()
        self._detect_commands()
        self._compute_complexity()
        return self.profile

    def _detect_languages(self) -> None:
        lang_counts: dict[str, int] = {}
        for mod in self.graph.modules.values():
            if mod.language and not mod.is_config:
                lang_counts[mod.language] = lang_counts.get(mod.language, 0) + 1

        if lang_counts:
            sorted_langs = sorted(lang_counts.items(), key=lambda x: -x[1])
            self.profile.primary_language = sorted_langs[0][0]
            self.profile.secondary_languages = [l for l, _ in sorted_langs[1:3]]

    def _detect_framework(self) -> None:
        deps = {d.package.lower() for d in self.graph.dependencies}
        metadata_deps = self._extract_dependencies_from_metadata()
        all_deps = deps | metadata_deps

        # Django special detection: check for manage.py or settings.py
        has_django_files = (self.root / "manage.py").is_file() or any(
            "settings.py" in Path(m.path).name for m in self.graph.modules.values()
        )

        # Python frameworks
        if "django" in all_deps or has_django_files:
            self.profile.framework = "Django"
            self.profile.architecture_pattern = "MVC (MTV)"
        elif "fastapi" in all_deps:
            self.profile.framework = "FastAPI"
            self.profile.architecture_pattern = "REST API"
        elif "flask" in all_deps:
            self.profile.framework = "Flask"
            self.profile.architecture_pattern = "Microframework"
        elif "streamlit" in all_deps:
            self.profile.framework = "Streamlit"
            self.profile.architecture_pattern = "Dashboard"
        elif "langgraph" in all_deps or "langchain" in all_deps:
            self.profile.framework = "LangGraph/LangChain"
            self.profile.architecture_pattern = "Agent Pipeline"

        # JS frameworks
        elif "next" in all_deps:
            self.profile.framework = "Next.js"
            self.profile.architecture_pattern = "SSR Framework"
        elif "react" in all_deps:
            self.profile.framework = "React"
            self.profile.architecture_pattern = "SPA"
        elif "express" in all_deps:
            self.profile.framework = "Express.js"
            self.profile.architecture_pattern = "REST API"
        elif "vue" in all_deps:
            self.profile.framework = "Vue.js"
            self.profile.architecture_pattern = "SPA"

        # .NET
        elif any("aspnetcore" in d for d in all_deps) or "microsoft.aspnetcore.app" in all_deps:
            self.profile.framework = "ASP.NET Core"
            self.profile.architecture_pattern = "MVC"

        # Go
        elif "gin-gonic/gin" in all_deps or "gin" in all_deps:
            self.profile.framework = "Gin"
            self.profile.architecture_pattern = "REST API"

        # Fallback architecture detection
        if not self.profile.architecture_pattern:
            paths = [m.path for m in self.graph.modules.values()]
            if any("controller" in p.lower() for p in paths) and any("model" in p.lower() for p in paths):
                self.profile.architecture_pattern = "MVC"
            elif any("service" in p.lower() for p in paths) and any("repository" in p.lower() for p in paths):
                self.profile.architecture_pattern = "Layered"
            else:
                self.profile.architecture_pattern = "Flat/Script"

    def _extract_dependencies_from_metadata(self) -> set[str]:
        deps: set[str] = set()

        # 1. Parse pyproject.toml
        pyproject_path = self.root / "pyproject.toml"
        if pyproject_path.is_file():
            try:
                import tomllib
                content = pyproject_path.read_bytes()
                data = tomllib.loads(content.decode("utf-8", errors="replace"))

                project = data.get("project", {})
                for dep in project.get("dependencies", []):
                    m = re.match(r"^([a-zA-Z0-9_\-]+)", dep.strip())
                    if m:
                        deps.add(m.group(1).lower())
                for option, opt_deps in project.get("optional-dependencies", {}).items():
                    for dep in opt_deps:
                        m = re.match(r"^([a-zA-Z0-9_\-]+)", dep.strip())
                        if m:
                            deps.add(m.group(1).lower())

                poetry = data.get("tool", {}).get("poetry", {})
                for dep in poetry.get("dependencies", {}):
                    deps.add(dep.lower())
                for group, group_data in poetry.get("group", {}).items():
                    for dep in group_data.get("dependencies", {}):
                        deps.add(dep.lower())

                if "pytest" in data.get("tool", {}):
                    deps.add("pytest")
                if "ruff" in data.get("tool", {}):
                    deps.add("ruff")
                if "mypy" in data.get("tool", {}):
                    deps.add("mypy")
            except Exception:
                try:
                    content = pyproject_path.read_text(encoding="utf-8", errors="replace")
                    for dep in re.findall(r'"([a-zA-Z0-9_\-]+)(?:[>=<~! ]|$)', content):
                        deps.add(dep.lower())
                except Exception:
                    pass

        # 2. Parse setup.py
        setup_py = self.root / "setup.py"
        if setup_py.is_file():
            try:
                content = setup_py.read_text(encoding="utf-8", errors="replace")
                matches = re.findall(r"['\"]([a-zA-Z0-9_\-]+)(?:[>=<~! ]|['\"])", content)
                for m in matches:
                    deps.add(m.lower())
            except Exception:
                pass

        # 3. Parse requirements.txt
        for req_name in ["requirements.txt", "dev-requirements.txt", "requirements-dev.txt"]:
            req_file = self.root / req_name
            if req_file.is_file():
                try:
                    content = req_file.read_text(encoding="utf-8", errors="replace")
                    for line in content.splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            m = re.match(r"^([a-zA-Z0-9_\-]+)", line)
                            if m:
                                deps.add(m.group(1).lower())
                except Exception:
                    pass

        # 4. Parse package.json
        pkg_json = self.root / "package.json"
        if pkg_json.is_file():
            try:
                data = json.loads(pkg_json.read_text(encoding="utf-8", errors="replace"))
                for dep_type in ["dependencies", "devDependencies", "peerDependencies"]:
                    for dep in data.get(dep_type, {}):
                        deps.add(dep.lower())
            except Exception:
                pass

        # 5. Extract from CI configs
        workflows_dir = self.root / ".github" / "workflows"
        if workflows_dir.is_dir():
            try:
                for f in workflows_dir.iterdir():
                    if f.is_file() and (f.suffix in [".yml", ".yaml"]):
                        content = f.read_text(encoding="utf-8", errors="replace")
                        if "pytest" in content:
                            deps.add("pytest")
                        if "mypy" in content:
                            deps.add("mypy")
                        if "ruff" in content:
                            deps.add("ruff")
                        if "tox" in content:
                            deps.add("tox")
                        if "django" in content.lower():
                            deps.add("django")
            except Exception:
                pass

        return deps

    def _detect_tooling(self) -> None:
        configs = {m.path for m in self.graph.modules.values() if m.is_config}
        config_names = {Path(p).name for p in configs}

        # Formatter
        if "pyproject.toml" in config_names:
            pyproject_content = self._read_config("pyproject.toml")
            if "[tool.black]" in pyproject_content:
                self.profile.formatter = "black"
            elif "[tool.ruff.format]" in pyproject_content:
                self.profile.formatter = "ruff format"
        if ".prettierrc" in config_names or ".prettierrc.json" in config_names:
            self.profile.formatter = "prettier"

        # Linter
        if "pyproject.toml" in config_names:
            pyproject_content = self._read_config("pyproject.toml")
            if "[tool.ruff]" in pyproject_content:
                self.profile.linter = "ruff"
        if any(name.startswith(".eslintrc") for name in config_names):
            self.profile.linter = "eslint"

        # Type checker
        if "mypy.ini" in config_names or "pyproject.toml" in config_names:
            pyproject_content = self._read_config("pyproject.toml")
            if "[tool.mypy]" in pyproject_content:
                self.profile.type_checker = "mypy"
        if "tsconfig.json" in config_names:
            self.profile.type_checker = "tsc"

        # Package manager
        if "uv.lock" in config_names:
            self.profile.package_manager = "uv"
        elif "requirements.txt" in config_names or "pyproject.toml" in config_names:
            self.profile.package_manager = "pip"
        if "yarn.lock" in config_names:
            self.profile.package_manager = "yarn"
        elif "pnpm-lock.yaml" in config_names:
            self.profile.package_manager = "pnpm"
        elif "package-lock.json" in config_names:
            self.profile.package_manager = "npm"

    def _detect_testing(self) -> None:
        deps = {d.package.lower() for d in self.graph.dependencies}
        metadata_deps = self._extract_dependencies_from_metadata()
        all_deps = deps | metadata_deps
        test_modules = self.graph.get_test_modules()

        has_pytest_ini = (self.root / "pytest.ini").is_file() or (self.root / "tox.ini").is_file()

        # Detect framework
        if "pytest" in all_deps or has_pytest_ini or any("conftest.py" in m.path for m in test_modules):
            self.profile.test_framework = "pytest"
            self.profile.test_command = "pytest"
        elif "unittest" in all_deps or any("unittest" in m.imports for m in test_modules):
            self.profile.test_framework = "unittest"
            self.profile.test_command = "python -m unittest discover"
        elif "jest" in all_deps:
            self.profile.test_framework = "jest"
            self.profile.test_command = "npx jest"
        elif "vitest" in all_deps:
            self.profile.test_framework = "vitest"
            self.profile.test_command = "npx vitest"

        # Test directory
        test_dirs = set()
        for mod in test_modules:
            parts = Path(mod.path).parts
            if len(parts) > 1:
                test_dirs.add(parts[0])

        if not test_dirs:
            for dirname in ["tests", "test", "spec", "testing"]:
                if (self.root / dirname).is_dir():
                    test_dirs.add(dirname)

        if test_dirs:
            self.profile.test_directory = sorted(test_dirs)[0]

        # Detect conventions
        if test_modules:
            names = [Path(m.path).name for m in test_modules]
            if any(n.startswith("test_") for n in names):
                self.profile.test_convention = "test_*.py"
            elif any(n.endswith("_test.py") for n in names):
                self.profile.test_convention = "*_test.py"
            elif any(".test." in n for n in names):
                self.profile.test_convention = "*.test.*"
            elif any(".spec." in n for n in names):
                self.profile.test_convention = "*.spec.*"
        else:
            self.profile.test_convention = "test_*.py"

        # Check for fixtures and mocks
        for mod in test_modules:
            if "fixture" in str(mod.imports).lower() or "conftest" in mod.path:
                self.profile.uses_fixtures = True
            if "mock" in str(mod.imports).lower() or "unittest.mock" in str(mod.imports):
                self.profile.uses_mocks = True

        # Check for fixtures and mocks
        for mod in test_modules:
            if "fixture" in str(mod.imports).lower() or "conftest" in mod.path:
                self.profile.uses_fixtures = True
            if "mock" in str(mod.imports).lower() or "unittest.mock" in str(mod.imports):
                self.profile.uses_mocks = True

    def _detect_ci(self) -> None:
        # GitHub Actions
        workflows = self.root / ".github" / "workflows"
        if workflows.is_dir():
            self.profile.ci_system = "GitHub Actions"
            return
        # GitLab CI
        if (self.root / ".gitlab-ci.yml").is_file():
            self.profile.ci_system = "GitLab CI"
            return
        # Jenkins
        if (self.root / "Jenkinsfile").is_file():
            self.profile.ci_system = "Jenkins"
            return
        # CircleCI
        if (self.root / ".circleci" / "config.yml").is_file():
            self.profile.ci_system = "CircleCI"
            return

    def _detect_conventions(self) -> None:
        # Sample function names to detect naming convention
        func_names = [
            fn.name for fn in self.graph.functions.values()
            if not fn.name.startswith("_") and not fn.name.startswith("__")
        ][:50]

        if func_names:
            snake_count = sum(1 for n in func_names if "_" in n and n == n.lower())
            camel_count = sum(1 for n in func_names if any(c.isupper() for c in n[1:]) and "_" not in n)
            if snake_count > camel_count:
                self.profile.naming_style = "snake_case"
            elif camel_count > snake_count:
                self.profile.naming_style = "camelCase"
            else:
                self.profile.naming_style = "mixed"

        # Detect docstring style from Python files
        for fn in list(self.graph.functions.values())[:20]:
            if fn.docstring:
                if ":param " in fn.docstring or ":type " in fn.docstring:
                    self.profile.docstring_style = "sphinx"
                    break
                if "Args:" in fn.docstring or "Returns:" in fn.docstring:
                    self.profile.docstring_style = "google"
                    break
                if "Parameters\n" in fn.docstring or "Parameters\r\n" in fn.docstring:
                    self.profile.docstring_style = "numpy"
                    break

    def _detect_commands(self) -> None:
        # package.json scripts
        pkg_json = self.root / "package.json"
        if pkg_json.is_file():
            try:
                data = json.loads(pkg_json.read_text(encoding="utf-8", errors="replace"))
                scripts = data.get("scripts", {})
                if "build" in scripts:
                    self.profile.build_command = f"npm run build"
                if "test" in scripts:
                    self.profile.test_command = f"npm test"
                if "dev" in scripts:
                    self.profile.dev_command = f"npm run dev"
            except (json.JSONDecodeError, OSError):
                pass

        # Makefile targets
        makefile = self.root / "Makefile"
        if makefile.is_file():
            try:
                content = makefile.read_text(encoding="utf-8", errors="replace")
                if "test:" in content and not self.profile.test_command:
                    self.profile.test_command = "make test"
                if "build:" in content and not self.profile.build_command:
                    self.profile.build_command = "make build"
            except OSError:
                pass

    def _compute_complexity(self) -> None:
        self.profile.total_files = len(self.graph.modules)
        self.profile.total_lines = sum(m.line_count for m in self.graph.modules.values())
        self.profile.total_symbols = len(self.graph.classes) + len(self.graph.functions)

        # Max module lines
        if self.graph.modules:
            self.profile.max_module_lines = max(m.line_count for m in self.graph.modules.values())

        # Complex modules (high line count or high symbol count)
        module_complexity = []
        for path, mod in self.graph.modules.items():
            if mod.is_config or mod.is_test:
                continue
            sym_count = len(mod.classes) + len(mod.functions)
            score = mod.line_count + sym_count * 10
            module_complexity.append((path, score))
        module_complexity.sort(key=lambda x: -x[1])
        self.profile.complex_modules = [p for p, _ in module_complexity[:5]]

        # Dependency depth (longest import chain)
        self.profile.dependency_depth = self._compute_max_depth()

    def _compute_max_depth(self) -> int:
        """BFS to find the longest import chain."""
        visited: dict[str, int] = {}
        max_depth = 0
        for path in self.graph.modules:
            if path in visited:
                continue
            queue = [(path, 0)]
            while queue:
                current, depth = queue.pop(0)
                if current in visited:
                    continue
                visited[current] = depth
                max_depth = max(max_depth, depth)
                for dep in self.graph.get_dependencies(current):
                    if dep not in visited:
                        queue.append((dep, depth + 1))
        return max_depth

    def _read_config(self, filename: str) -> str:
        """Read a config file from the repo root."""
        path = self.root / filename
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        return ""
