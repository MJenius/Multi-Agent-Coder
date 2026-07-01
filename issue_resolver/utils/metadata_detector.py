import re
import json
import tomllib
from pathlib import Path
from typing import Any

def detect_dependencies_from_files(root: Path) -> set[str]:
    deps: set[str] = set()

    # 1. Parse pyproject.toml
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.is_file():
        try:
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
    setup_py = root / "setup.py"
    if setup_py.is_file():
        try:
            content = setup_py.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"['\"]([a-zA-Z0-9_\-]+)(?:[>=<~! ]|['\"])", content)
            for m in matches:
                deps.add(m.lower())
        except Exception:
            pass

    # 3. Parse requirements files
    for req_name in ["requirements.txt", "dev-requirements.txt", "requirements-dev.txt"]:
        req_file = root / req_name
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
    pkg_json = root / "package.json"
    if pkg_json.is_file():
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8", errors="replace"))
            for dep_type in ["dependencies", "devDependencies", "peerDependencies"]:
                for dep in data.get(dep_type, {}):
                    deps.add(dep.lower())
        except Exception:
            pass

    # 5. Parse lock files
    for lock_name in ["uv.lock", "poetry.lock"]:
        lock_file = root / lock_name
        if lock_file.is_file():
            try:
                content = lock_file.read_text(encoding="utf-8", errors="replace")
                for pkg in re.findall(r'(?m)^name\s*=\s*["\']([a-zA-Z0-9_\-]+)["\']', content):
                    deps.add(pkg.lower())
            except Exception:
                pass

    pkg_lock = root / "package-lock.json"
    if pkg_lock.is_file():
        try:
            data = json.loads(pkg_lock.read_text(encoding="utf-8", errors="replace"))
            packages = data.get("packages", {})
            for pkg_path in packages:
                name = pkg_path.replace("node_modules/", "")
                if name:
                    deps.add(name.lower())
            for dep in data.get("dependencies", {}):
                deps.add(dep.lower())
        except Exception:
            pass

    return deps

def detect_environment_metadata(root: Path) -> dict[str, Any]:
    """
    Detect all relevant metadata of the repository directly from files.
    Returns a dictionary of detected tooling and options.
    """
    dependencies = detect_dependencies_from_files(root)
    
    # 1. Determine Language
    has_dotnet = any(root.glob("*.sln")) or any(root.glob("*.csproj")) or any(root.rglob("*.csproj"))
    has_node = (root / "package.json").is_file()
    has_python = (root / "requirements.txt").is_file() or (root / "pyproject.toml").is_file() or (root / "uv.lock").is_file()
    has_go = (root / "go.mod").is_file()
    has_rust = (root / "Cargo.toml").is_file()
    
    language = "unknown"
    if has_dotnet:
        language = "dotnet"
    elif has_node:
        language = "nodejs"
    elif has_python:
        language = "python"
    elif has_go:
        language = "go"
    elif has_rust:
        language = "rust"
    else:
        # Fallback to extension counts
        ext_counts = {"python": 0, "dotnet": 0, "nodejs": 0, "go": 0, "rust": 0}
        ext_map = {".py": "python", ".cs": "dotnet", ".js": "nodejs", ".ts": "nodejs", ".go": "go", ".rs": "rust"}
        for p in root.glob("*"):
            if p.is_file() and p.suffix in ext_map:
                ext_counts[ext_map[p.suffix]] += 1
        for dirname in ["src", "lib", "app", "tests"]:
            if (root / dirname).is_dir():
                try:
                    for p in (root / dirname).rglob("*"):
                        if p.is_file() and p.suffix in ext_map:
                            ext_counts[ext_map[p.suffix]] += 1
                except Exception:
                    pass
        max_lang = max(ext_counts, key=ext_counts.get)
        if ext_counts[max_lang] > 0:
            language = max_lang

    # 2. Determine Framework
    framework = "none"
    architecture = "Flat/Script"
    if language == "python":
        has_django_files = (root / "manage.py").is_file()
        if "django" in dependencies or has_django_files:
            framework = "Django"
            architecture = "MVC (MTV)"
        elif "fastapi" in dependencies:
            framework = "FastAPI"
            architecture = "REST API"
        elif "flask" in dependencies:
            framework = "Flask"
            architecture = "Microframework"
        elif "streamlit" in dependencies:
            framework = "Streamlit"
            architecture = "Dashboard"
        elif "langgraph" in dependencies or "langchain" in dependencies:
            framework = "LangGraph/LangChain"
            architecture = "Agent Pipeline"
    elif language == "nodejs":
        if "next" in dependencies:
            framework = "Next.js"
            architecture = "SSR Framework"
        elif "react" in dependencies:
            framework = "React"
            architecture = "SPA"
        elif "express" in dependencies:
            framework = "Express.js"
            architecture = "REST API"
        elif "vue" in dependencies:
            framework = "Vue.js"
            architecture = "SPA"
    elif language == "dotnet":
        framework = "ASP.NET Core" if "microsoft.aspnetcore.app" in dependencies else "dotNET"
        architecture = "MVC"
    elif language == "go":
        if "gin" in dependencies or "gin-gonic/gin" in dependencies:
            framework = "Gin"
            architecture = "REST API"

    # 3. Determine Test Framework & Commands
    test_framework = "unknown"
    test_command = ""
    test_directory = "tests"
    if language == "python":
        has_pytest_ini = (root / "pytest.ini").is_file() or (root / "tox.ini").is_file() or (root / "conftest.py").is_file()
        if "pytest" in dependencies or has_pytest_ini:
            test_framework = "pytest"
            test_command = "pytest"
        elif "unittest" in dependencies:
            test_framework = "unittest"
            test_command = "python -m unittest discover"
        else:
            test_framework = "pytest"
            test_command = "pytest"
    elif language == "nodejs":
        if "vitest" in dependencies:
            test_framework = "vitest"
            test_command = "npx vitest"
        elif "jest" in dependencies:
            test_framework = "jest"
            test_command = "npx jest"
        elif "mocha" in dependencies:
            test_framework = "mocha"
            test_command = "npx mocha"
        else:
            test_framework = "jest"
            test_command = "npm test"
    elif language == "dotnet":
        packages = {
            "xunit": ("xunit", "xunit.runner"),
            "nunit": ("nunit", "nunit3testadapter"),
            "mstest": ("mstest", "microsoft.net.test.sdk"),
        }
        for fw, needles in packages.items():
            if any(needle in dependencies for needle in needles):
                test_framework = fw
                break
        if test_framework == "unknown":
            test_framework = "xunit"
        test_command = "dotnet test"
    elif language == "go":
        test_framework = "go test"
        test_command = "go test ./..."
    elif language == "rust":
        test_framework = "cargo test"
        test_command = "cargo test"

    # Test directories check
    for dirname in ["tests", "test", "spec", "testing"]:
        if (root / dirname).is_dir():
            test_directory = dirname
            break

    # 4. Determine Build/Dev Commands
    build_command = ""
    dev_command = ""
    if language == "python":
        build_command = "python -m py_compile"
    elif language == "nodejs":
        pkg_json = root / "package.json"
        if pkg_json.is_file():
            try:
                data = json.loads(pkg_json.read_text(encoding="utf-8", errors="replace"))
                scripts = data.get("scripts", {})
                build_command = "npm run build" if "build" in scripts else ""
                dev_command = "npm run dev" if "dev" in scripts else ""
            except Exception:
                pass
        if not build_command:
            build_command = "npm run build"
    elif language == "dotnet":
        build_command = "dotnet build"
    elif language == "go":
        build_command = "go build ./..."
    elif language == "rust":
        build_command = "cargo build"

    # 5. Determine Package Manager
    package_manager = "unknown"
    if language == "python":
        if (root / "uv.lock").is_file():
            package_manager = "uv"
        elif (root / "poetry.lock").is_file():
            package_manager = "poetry"
        elif (root / "Pipfile.lock").is_file():
            package_manager = "pipenv"
        else:
            package_manager = "pip"
    elif language == "nodejs":
        if (root / "yarn.lock").is_file():
            package_manager = "yarn"
        elif (root / "pnpm-lock.yaml").is_file():
            package_manager = "pnpm"
        else:
            package_manager = "npm"
    elif language == "dotnet":
        package_manager = "nuget"
    elif language == "go":
        package_manager = "go modules"
    elif language == "rust":
        package_manager = "cargo"

    # 6. Determine Formatter
    formatter = "unknown"
    pyproject_toml = root / "pyproject.toml"
    if pyproject_toml.is_file():
        try:
            content = pyproject_toml.read_text(encoding="utf-8", errors="replace")
            if "[tool.black]" in content:
                formatter = "black"
            elif "[tool.ruff.format]" in content:
                formatter = "ruff format"
            elif "[tool.yapf]" in content:
                formatter = "yapf"
        except Exception:
            pass
    if formatter == "unknown":
        if language == "python" and "black" in dependencies:
            formatter = "black"
        elif language == "python" and "ruff" in dependencies:
            formatter = "ruff format"
        elif (root / ".prettierrc").is_file() or (root / ".prettierrc.json").is_file() or "prettier" in dependencies:
            formatter = "prettier"
        elif language == "go":
            formatter = "gofmt"
        elif language == "rust":
            formatter = "rustfmt"

    # 7. Determine Linter
    linter = "unknown"
    if pyproject_toml.is_file():
        try:
            content = pyproject_toml.read_text(encoding="utf-8", errors="replace")
            if "[tool.ruff]" in content:
                linter = "ruff"
            elif "[tool.pylint]" in content:
                linter = "pylint"
            elif "[tool.flake8]" in content:
                linter = "flake8"
        except Exception:
            pass
    if linter == "unknown":
        if "ruff" in dependencies:
            linter = "ruff"
        elif "eslint" in dependencies or (root / ".eslintrc").is_file() or (root / ".eslintrc.json").is_file():
            linter = "eslint"
        elif language == "rust":
            linter = "clippy"

    # 8. Determine Type Checker
    type_checker = "unknown"
    if pyproject_toml.is_file():
        try:
            content = pyproject_toml.read_text(encoding="utf-8", errors="replace")
            if "[tool.mypy]" in content:
                type_checker = "mypy"
            elif "[tool.pyright]" in content:
                type_checker = "pyright"
        except Exception:
            pass
    if type_checker == "unknown":
        if "mypy" in dependencies:
            type_checker = "mypy"
        elif "pyright" in dependencies:
            type_checker = "pyright"
        elif "typescript" in dependencies or (root / "tsconfig.json").is_file():
            type_checker = "tsc"

    # 9. Determine CI Workflow
    ci_system = "none"
    if (root / ".github" / "workflows").is_dir():
        ci_system = "GitHub Actions"
    elif (root / ".gitlab-ci.yml").is_file():
        ci_system = "GitLab CI"
    elif (root / ".circleci" / "config.yml").is_file():
        ci_system = "CircleCI"

    # 10. Determine Coverage Tools
    coverage_tool = "none"
    if language == "python":
        if "pytest-cov" in dependencies:
            coverage_tool = "pytest-cov"
        elif "coverage" in dependencies:
            coverage_tool = "coverage"
    elif language == "nodejs":
        if "nyc" in dependencies:
            coverage_tool = "nyc"
        elif "c8" in dependencies:
            coverage_tool = "c8"
        elif "istanbul" in dependencies:
            coverage_tool = "istanbul"

    return {
        "primary_language": language,
        "framework": framework,
        "architecture_pattern": architecture,
        "test_framework": test_framework,
        "test_command": test_command,
        "test_directory": test_directory,
        "build_command": build_command,
        "dev_command": dev_command,
        "package_manager": package_manager,
        "formatter": formatter,
        "linter": linter,
        "type_checker": type_checker,
        "ci_system": ci_system,
        "coverage_tool": coverage_tool,
    }
