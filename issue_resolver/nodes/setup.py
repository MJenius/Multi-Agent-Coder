"""Setup Node -- Detect repository archetype and prime runtime environment config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from issue_resolver.runtime_context import set_environment_config
from issue_resolver.state import AgentState
from issue_resolver.tools.repo_tools import IGNORE_DIRS, generate_symbol_map
from issue_resolver.utils.logger import append_to_history


def _load_root_gitignore_patterns(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return []

    patterns: list[str] = []
    try:
        for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                patterns.append(stripped)
    except OSError:
        return []
    return patterns


def _detect_dotnet_test_framework(root: Path) -> str:
    packages = {
        "xunit": ("xunit", "xunit.runner", "xunit.runner.visualstudio"),
        "nunit": ("nunit", "nunit3testadapter"),
        "mstest": ("mstest", "microsoft.net.test.sdk"),
    }

    csproj_files = list(root.rglob("*.csproj"))
    if not csproj_files:
        return "unknown"

    content_blob = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in csproj_files
        if p.is_file()
    ).lower()

    for framework, needles in packages.items():
        if any(needle in content_blob for needle in needles):
            return framework
    return "unknown"


def _detect_python_test_framework(root: Path) -> str:
    """Detect pytest vs unittest in Python projects."""
    # Check for pytest markers
    if (root / "conftest.py").is_file():
        return "pytest"
    
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        content = pyproject.read_text(encoding="utf-8", errors="replace").lower()
        if "pytest" in content or "tools.pytest" in content:
            return "pytest"
    
    pytest_ini = root / "pytest.ini"
    if pytest_ini.is_file():
        return "pytest"
    
    # Default to pytest (more common in modern Python)
    return "pytest"


def _detect_nodejs_test_framework(root: Path) -> str:
    """Detect jest vs vitest vs other in Node.js projects."""
    package_json = root / "package.json"
    if not package_json.is_file():
        return "unknown"
    
    try:
        import json
        content = json.loads(package_json.read_text(encoding="utf-8", errors="replace"))
        deps = {**(content.get("devDependencies") or {}), **(content.get("dependencies") or {})}
        
        if "jest" in deps:
            return "jest"
        if "vitest" in deps:
            return "vitest"
        if "mocha" in deps:
            return "mocha"
        if "jasmine" in deps:
            return "jasmine"
    except Exception:
        pass
    
    return "jest"  # Default to jest


def _detect_environment(root: Path) -> tuple[str, dict[str, Any]]:
    has_sln = any(root.rglob("*.sln"))
    has_csproj = any(root.rglob("*.csproj"))
    has_dotnet = has_sln or has_csproj
    has_node = (root / "package.json").is_file()
    has_python = (root / "requirements.txt").is_file() or (root / "pyproject.toml").is_file()

    if has_dotnet:
        return "dotnet", {
            "test_framework": _detect_dotnet_test_framework(root),
            "detector_evidence": "sln/csproj detected",
            "build_command": "dotnet build",
            "test_command": "dotnet test",
        }
    if has_node:
        return "nodejs", {
            "test_framework": _detect_nodejs_test_framework(root),
            "detector_evidence": "package.json detected",
            "build_command": "npm run build",
            "test_command": "npm test",
        }
    if has_python:
        return "python", {
            "test_framework": _detect_python_test_framework(root),
            "detector_evidence": "requirements/pyproject detected",
            "build_command": "python -m py_compile",
            "test_command": "pytest",
        }
    return "unknown", {
        "test_framework": "unknown",
        "detector_evidence": "no archetype markers",
        "build_command": "",
        "test_command": "",
    }


def setup_node(state: AgentState) -> dict:
    repo_path = state.get("repo_path", "./sandbox_workspace")
    issue_text = state.get("issue", "")
    issue_title = issue_text.splitlines()[0].strip() if issue_text else ""

    root = Path(repo_path).resolve()
    env_type, extra = _detect_environment(root)
    gitignore_patterns = _load_root_gitignore_patterns(root)

    merged_ignore = sorted(set(IGNORE_DIRS).union(gitignore_patterns))
    env_config = {
        "repo_root": str(root),
        "environment_type": env_type,
        "test_framework": extra.get("test_framework", "unknown"),
        "build_command": extra.get("build_command", ""),
        "test_command": extra.get("test_command", ""),
        "detector_evidence": extra.get("detector_evidence", ""),
        "ignore_dirs": sorted(IGNORE_DIRS),
        "gitignore_patterns": gitignore_patterns,
        "merged_ignore_spec": merged_ignore,
        "issue_title": issue_title,
    }

    set_environment_config(env_config)
    
    # Generate symbol map for Planner context (capped at 100 symbols)
    symbol_map = generate_symbol_map(str(root))
    
    history_msg = f"Detected {env_type} (framework={env_config['test_framework']})"
    if len(symbol_map) > 20:  # Non-empty symbol map
        history_msg += f"; symbol map generated ({len(symbol_map.splitlines())} symbols)"

    return {
        "environment_config": env_config,
        "symbol_map": symbol_map,
        "history": append_to_history(
            "Setup",
            "Environment Detection",
            history_msg,
        ),
    }
