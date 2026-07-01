"""Setup Node -- Detect repository archetype and prime runtime environment config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from issue_resolver.runtime_context import set_environment_config, reset_runtime_context
from issue_resolver.state import AgentState
from issue_resolver.tools.repo_tools import IGNORE_DIRS, _generate_symbol_map_impl
from issue_resolver.utils.logger import append_to_history
from issue_resolver.utils.metadata_detector import detect_environment_metadata


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


def setup_node(state: AgentState) -> dict:
    # 1. Reset per-repo runtime context state selectively
    reset_runtime_context()
    
    repo_path = state.get("repo_path", "./sandbox_workspace")
    issue_text = state.get("issue", "")
    issue_title = issue_text.splitlines()[0].strip() if issue_text else ""

    # Verify verification environment is available (sandbox or local fallback)
    from issue_resolver.tools.sandbox_tools import get_sandbox_container
    import subprocess
    import shutil
    
    sandbox_available = False
    try:
        sandbox = get_sandbox_container()
        if sandbox is not None:
            sandbox_available = True
    except Exception:
        pass
        
    local_available = False
    if not sandbox_available:
        print("[Setup] Sandbox container is NOT running. Checking for local verification fallback...")
        if shutil.which("python") or shutil.which("python3") or shutil.which("pytest") or shutil.which("mypy") or shutil.which("ruff"):
            local_available = True
        else:
            try:
                subprocess.run(["python", "--version"], capture_output=True, timeout=2)
                local_available = True
            except Exception:
                pass
                
        if not local_available:
            raise RuntimeError(
                "Verification Infrastructure Error: Neither Docker sandbox container is running, "
                "nor are local verification tools (python, pytest, mypy, ruff) available in the system PATH. "
                "At least one verification method must be available before starting resolution."
            )
        print("[Setup] Local verification tools are available. Using local verification fallback.")

    root = Path(repo_path).resolve()
    
    # 2. Run central metadata detection
    metadata = detect_environment_metadata(root)
    env_type = metadata["primary_language"]
    
    gitignore_patterns = _load_root_gitignore_patterns(root)
    merged_ignore = sorted(set(IGNORE_DIRS).union(gitignore_patterns))
    
    env_config = {
        "repo_root": str(root),
        "environment_type": env_type,
        "test_framework": metadata["test_framework"],
        "build_command": metadata["build_command"],
        "test_command": metadata["test_command"],
        "detector_evidence": f"detected from manifest files in {root}",
        "ignore_dirs": sorted(IGNORE_DIRS),
        "gitignore_patterns": gitignore_patterns,
        "merged_ignore_spec": merged_ignore,
        "issue_title": issue_title,
        # Expanded metadata properties (formatter, linter, type checker, coverage)
        "formatter": metadata["formatter"],
        "linter": metadata["linter"],
        "type_checker": metadata["type_checker"],
        "ci_system": metadata["ci_system"],
        "coverage_tool": metadata["coverage_tool"],
    }

    set_environment_config(env_config)
    
    from issue_resolver.utils.code_mapper import CodeMapper
    symbol_map = CodeMapper.generate_repo_map(str(root))
    
    history_msg = f"Detected {env_type} (framework={metadata['framework']}, test_runner={env_config['test_framework']})"
    if len(symbol_map) > 20:  # Non-empty symbol map
        history_msg += f"; symbol map generated ({len(symbol_map.splitlines())} symbols)"

    # Tracing diagnostics
    print(f"[Setup] Setup completed selectively resetting old state.")
    print(f"[Setup] Env type: {env_type}, framework: {metadata['framework']}, formatter: {metadata['formatter']}, linter: {metadata['linter']}, type_checker: {metadata['type_checker']}")

    return {
        "environment_config": env_config,
        "symbol_map": symbol_map,
        "history": append_to_history(
            "Setup",
            "Environment Detection",
            history_msg,
        ),
    }

