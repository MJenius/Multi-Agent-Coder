"""
Repository Tools -- Local codebase search utilities for the Researcher agent.

Each function is decorated with @tool so it can be bound to a tool-capable chat model
via .bind_tools(). All tools include RAM-safety guards:
  - list_files:   caps output at 200 files
  - search_code:  caps output at 30 matches
  - read_file:    truncates at 500 lines
  - ALL TOOLS:    timeout protection (30s default) to prevent hangs on large repos
"""

from __future__ import annotations

import os
import re
import threading
from functools import wraps
from pathlib import Path

from langchain_core.tools import tool
from issue_resolver.config import SANDBOX_WORKSPACE_DIR
from issue_resolver.runtime_context import get_environment_config

try:
    import pathspec
except ImportError:  # pragma: no cover - optional at import time, required in runtime env
    pathspec = None


# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFINEMENT -- Reject any path outside the allowed workspace
# ─────────────────────────────────────────────────────────────────────────────
_ALLOWED_ROOT = Path(SANDBOX_WORKSPACE_DIR).resolve()


def _check_confinement(resolved_path: Path, label: str = "path") -> str | None:
    """Return an error string if resolved_path is outside _ALLOWED_ROOT, else None."""
    try:
        resolved_path.relative_to(_ALLOWED_ROOT)
        return None
    except ValueError:
        return (
            f"Error: {label} '{resolved_path}' is outside the allowed workspace "
            f"'{_ALLOWED_ROOT}'. Path traversal is not permitted."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TIMEOUT GUARD -- Prevent hangs on large repositories
# ─────────────────────────────────────────────────────────────────────────────
# Cross-platform timeout implementation using threading (works on Windows + Unix)
#
# Why this matters:
#   - Unix signal.SIGALRM doesn't exist on Windows
#   - Threading-based timeout works everywhere
#   - 30-second default prevents 4+ minute hangs on 2500+ file repos (like QRCoder)
#   - Graceful failure with informative error message
#
_timeout_occurred = False

def with_timeout(seconds: int = 30):
    """Decorator to add timeout protection to repo tool functions.
    
    Args:
        seconds: Maximum execution time before timeout. Default 30s.
    
    Usage:
        @tool
        @with_timeout(30)
        def my_tool(...) -> str:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result_container = [None]
            exception_container = [None]
            event = threading.Event()
            
            def target():
                try:
                    result_container[0] = func(*args, **kwargs)
                except Exception as e:
                    exception_container[0] = e
                finally:
                    event.set()
            
            thread = threading.Thread(target=target, daemon=False)
            thread.start()
            
            if not event.wait(timeout=seconds):
                # Timeout occurred
                func_name = func.__name__
                return (
                    f"ERROR: {func_name}() exceeded {seconds}s timeout. "
                    f"Repository may be too large or disk I/O is slow. "
                    f"Try narrowing search to a specific folder (e.g., './src' or './QRCoder')."
                )
            
            if exception_container[0]:
                raise exception_container[0]
            
            return result_container[0]
        
        return wrapper
    return decorator



# CRITICAL: Every tool (list_files, search_code, read_file, generate_repo_map,
# get_symbol_definition) must use this exact set to prevent token drowning.
#
# Why each category matters:
#   - .git, .gitignore: Version control metadata (irrelevant to code search)
#   - bin, obj: C# build outputs with THOUSANDS of auto-generated .json/.cache/.props files
#   - .vs, packages: Visual Studio artifacts and NuGet cache (critical for .NET repos)
#   - __pycache__, .pytest_cache, .mypy_cache: Python bytecode and type-checking cache
#   - node_modules: Thousands of npm dependencies (99% noise)
#   - venv, .venv, env: Python virtual environments (duplicated dependencies)
#   - build, dist, target: Common build output directories
#   - .idea, .vscode, .github: IDE and GitHub workflow configs (not source code)
#   - htmlcov, .tox, .coverage: Test/coverage artifacts
#
# WITHOUT THIS FILTERING:
#   A single C# project's bin/ folder can contain 10,000+ files, causing:
#   1. Massive tree output that floods the LLM context
#   2. Token exhaustion on grep operations (30+ matches from .json metadata)
#   3. False "0 results found" errors (agent gives up after context limit)
#   4. 5+ minute hangs (tool processes thousands of non-source files)
#
IGNORE_DIRS = {
    # Version control
    ".git", ".gitignore", ".github",
    # Python
    ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", ".coverage",
    # Node.js
    "node_modules", ".npm",
    # C# / .NET (CRITICAL -- prevents bin/obj avalanche)
    "bin", "obj", ".vs", "packages",
    # General build outputs
    "build", "dist", "target", "htmlcov",
    # IDE configs (not source)
    ".idea", ".vscode",
}


def _load_root_gitignore_patterns(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return []
    try:
        lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _get_effective_ignore_parts(root: Path) -> tuple[set[str], object | None]:
    env = get_environment_config()
    root_gitignore = _load_root_gitignore_patterns(root)
    configured_gitignore = env.get("gitignore_patterns", []) if isinstance(env, dict) else []
    merged_patterns = list(dict.fromkeys(list(configured_gitignore) + root_gitignore))
    ignore_dirs = set(env.get("ignore_dirs", IGNORE_DIRS)) if isinstance(env, dict) else set(IGNORE_DIRS)

    if pathspec is None or not merged_patterns:
        return ignore_dirs, None
    try:
        matcher = pathspec.PathSpec.from_lines("gitwildmatch", merged_patterns)
    except Exception:
        matcher = None
    return ignore_dirs, matcher


def _is_ignored(rel_path: Path, ignore_dirs: set[str], matcher: object | None) -> bool:
    parts = rel_path.parts
    if any(part in ignore_dirs or part.startswith(".") for part in parts if part):
        return True
    if matcher is None:
        return False
    rel_posix = rel_path.as_posix()
    try:
        return bool(matcher.match_file(rel_posix))
    except Exception:
        return False


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "into", "when", "where", "have", "has", "was", "are",
        "issue", "error", "fails", "failure", "fix", "bug", "test", "build",
    }
    ordered: list[str] = []
    for token in tokens:
        if token in stop_words:
            continue
        if token not in ordered:
            ordered.append(token)
    return ordered[:12]


def _score_path(path_text: str, keywords: list[str]) -> int:
    lowered = path_text.lower()
    score = 0
    for kw in keywords:
        if kw in lowered:
            score += 3
        else:
            parts = lowered.replace("-", "/").replace("_", "/").split("/")
            if kw in parts:
                score += 2
    return score

# ---------------------------------------------------------------------------
# Tool 1 -- list_files
# ---------------------------------------------------------------------------
@tool
@with_timeout(20)
def list_files(directory: str) -> str:
    """Recursively list all code files in a DIRECTORY to understand project structure.
    
    CRITICAL: This tool ONLY accepts directory paths (like '.' or './src'). 
    DO NOT pass a specific file path (like 'src/main.py') to this tool.
    If you want to view a file, use the 'read_file' tool instead.

    Args:
        directory: Absolute or relative path to the root directory to scan.

    Returns:
        A newline-separated list of code file paths.
    """
    root = Path(directory).resolve()
    
    # PATH CONFINEMENT: reject directories outside the sandbox
    err = _check_confinement(root, "directory")
    if err:
        return err
    
    # HARD GUARD: If it's a file, return an explicit error string to guide the LLM
    if root.is_file():
        return f"ERROR: '{directory}' is a FILE. You must use the 'read_file' tool to see its contents."
        
    if not root.is_dir():
        return f"Error: '{directory}' is not a valid directory."

    ignore_dirs, matcher = _get_effective_ignore_parts(root)
    supported_exts = {".py", ".cs", ".xaml", ".cpp", ".h", ".js", ".ts", ".jsx", ".tsx", ".go", ".java"}
    code_files: list[str] = []
    
    for dirpath, _dirnames, filenames in os.walk(root):
        # ✅ UNIFIED FILTERING: All tools use the same IGNORE_DIRS constant
        rel_dir = Path(dirpath).relative_to(root)
        if _is_ignored(rel_dir, ignore_dirs, matcher):
            continue
            
        for fname in sorted(filenames):
            if any(fname.endswith(ext) for ext in supported_exts):
                rel = Path(dirpath, fname).relative_to(root)
                if _is_ignored(rel, ignore_dirs, matcher):
                    continue
                code_files.append(str(rel))
                if len(code_files) >= 200:
                    code_files.append("[TRUNCATED -- 200 file limit reached]")
                    return "\n".join(code_files)

    if not code_files:
        return "(no supported code files found)"
    return "\n".join(code_files)


# ---------------------------------------------------------------------------
# Tool 2 -- search_code
# ---------------------------------------------------------------------------
@tool
@with_timeout(20)
def search_code(query: str, directory: str) -> str:
    """Search for a string in all supported code files under a directory (grep-like).

    Args:
        query: The exact substring or function name to search for.
        directory: Root directory to search.

    Returns:
        Matching lines formatted as  file:line_number: <content>,
        capped at 30 matches.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        return f"Error: '{directory}' is not a valid directory."

    # PATH CONFINEMENT
    err = _check_confinement(root, "directory")
    if err:
        return err

    ignore_dirs, matcher = _get_effective_ignore_parts(root)
    supported_exts = {".py", ".cs", ".xaml", ".cpp", ".h", ".js", ".ts", ".jsx", ".tsx", ".go", ".java"}
    matches: list[str] = []
    
    for dirpath, _dirnames, filenames in os.walk(root):
        # ✅ UNIFIED FILTERING: Uses IGNORE_DIRS to prevent grep drowning
        rel_dir = Path(dirpath).relative_to(root)
        if _is_ignored(rel_dir, ignore_dirs, matcher):
            continue
            
        for fname in sorted(filenames):
            if not any(fname.endswith(ext) for ext in supported_exts):
                continue
                
            fpath = Path(dirpath, fname)
            rel = fpath.relative_to(root)
            if _is_ignored(rel, ignore_dirs, matcher):
                continue
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
                
            for idx, line in enumerate(lines, start=1):
                if query.lower() in line.lower():
                    matches.append(f"{rel}:{idx}: {line.rstrip()}")
                    if len(matches) >= 30:
                        matches.append("[TRUNCATED -- 30 match limit reached]")
                        return "\n".join(matches)

    if not matches:
        return f"No matches found for '{query}'."
    return "\n".join(matches)


# ---------------------------------------------------------------------------
# Tool 3 -- read_file
# ---------------------------------------------------------------------------
@tool
@with_timeout(15)
def read_file(file_path: str) -> str:
    """Read the full content of a file, truncated to 500 lines.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        The file content (up to 500 lines). A [TRUNCATED] marker is appended
        if the file exceeds 500 lines.
    """
    fpath = Path(file_path).resolve()

    # PATH CONFINEMENT
    err = _check_confinement(fpath, "file_path")
    if err:
        return err

    if not fpath.is_file():
        return f"Error: '{file_path}' is not a valid file."

    try:
        lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Error reading file: {exc}"

    truncated = len(lines) > 500
    output_lines = lines[:500]

    # Prepend line numbers for easy reference
    numbered = [f"{i}: {line}" for i, line in enumerate(output_lines, start=1)]
    if truncated:
        numbered.append(f"[TRUNCATED at 500 / {len(lines)} lines]")

    return "\n".join(numbered)


# ---------------------------------------------------------------------------
# Tool 4 -- generate_repo_map
# ---------------------------------------------------------------------------
@tool
@with_timeout(30)
def generate_repo_map(directory: str, max_depth: int = 2) -> str:
    """Generates a high-level tree-view of the directory structure (ignoring common build/virtualenv folders)
    and includes the contents of README.md if present. This gives a Map of the codebase.
    
    The tree is bounded to avoid overwhelming the LLM with too much information.
    By default, only shows 2 levels of depth (e.g., top-level folders and their immediate children).
    
    HARDCODED IGNORE LIST: Specifically excludes C# build artifacts (bin/, obj/) and other build noise
    to prevent mapping thousands of useless compiled files.

    Args:
        directory: Root directory to map.
        max_depth: Maximum depth to traverse (default 2). Use higher values for deeper exploration.

    Returns:
        A string containing the tree structure and README content (if any).
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        return f"Error: '{directory}' is not a valid directory."

    # PATH CONFINEMENT
    err = _check_confinement(root, "directory")
    if err:
        return err

    # ✅ SAFETY: Convert max_depth to int if LLM passes it as string
    max_depth = int(max_depth) if isinstance(max_depth, str) else max_depth

    ignore_dirs, matcher = _get_effective_ignore_parts(root)
    env = get_environment_config()
    issue_hint = env.get("issue_title", "") if isinstance(env, dict) else ""
    keywords = _extract_keywords(issue_hint)
    
    tree_lines = []
    
    top_ranked: list[str] = []

    def walk_tree(dir_path: Path, prefix: str = "", current_depth: int = 0):
        # Stop if we exceed max_depth
        if current_depth > max_depth:
            return
            
        if len(tree_lines) > 200:
            return
            
        try:
            items = sorted(os.listdir(dir_path))
        except OSError:
            return
            
        # Weighted mapping expands high-signal directories first.
        dirs = []
        files = []
        for item in items:
            p = dir_path / item
            rel = p.relative_to(root)
            if _is_ignored(rel, ignore_dirs, matcher):
                continue
            if p.is_dir():
                dirs.append(item)
            else:
                files.append(item)

        dirs = sorted(
            dirs,
            key=lambda d: (_score_path(str((dir_path / d).relative_to(root)), keywords), d.lower()),
            reverse=True,
        )

        if current_depth == 0 and dirs:
            top_ranked.extend(dirs[: min(5, len(dirs))])

        expanded_budget = 4 if current_depth == 0 else len(dirs)
                
        for i, d in enumerate(dirs):
            is_last = (i == len(dirs) - 1) and (len(files) == 0)
            connector = "└── " if is_last else "├── "
            rel_dir = (dir_path / d).relative_to(root).as_posix()
            score = _score_path(rel_dir, keywords)
            tree_lines.append(f"{prefix}{connector}{d}/ [score={score}]")
            if len(tree_lines) >= 200:
                return
            if i < expanded_budget:
                extension = "    " if is_last else "│   "
                walk_tree(dir_path / d, prefix + extension, current_depth + 1)
            else:
                tree_lines.append(f"{prefix}    ... collapsed; drill down if needed")
            
        # Only show files at max_depth level, not deeper
        if current_depth < max_depth:
            for i, f in enumerate(files):
                is_last = (i == len(files) - 1)
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{f}")
                if len(tree_lines) >= 200:
                    return
            
    tree_lines.append(str(root))
    walk_tree(root, "", current_depth=0)
    
    if len(tree_lines) >= 200:
        tree_lines.append("[TRUNCATED -- Repository map too large]")
        
    map_str = "\n".join(tree_lines)
    
    readme_content = ""
    for rm_name in ["README.md", "README.txt", "README"]:
        rm_path = root / rm_name
        if rm_path.is_file():
            try:
                lines = rm_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if len(lines) > 100:
                    lines = lines[:100] + ["... [README TRUNCATED] ..."]
                readme_content = f"\n\n--- {rm_name} ---\n" + "\n".join(lines)
                break
            except OSError:
                continue

    result = f"Repository Map:\n{map_str}{readme_content}"

    # Token guard: keep outputs bounded and preserve high-signal guidance.
    if len(result) > 10000:
        priority = ", ".join(top_ranked[:3]) if top_ranked else "src, tests, and feature folders"
        result = (
            result[:9800]
            + "\n\n[TRUNCATED at 10,000 characters]\n"
            + f"Drill down into subdirectories with highest relevance first: {priority}."
        )
    
    # ⚠️ OUTPUT VALIDATION: Warn if map is suspiciously large (possible filter bypass)
    if len(tree_lines) >= 200:
        print(f"[WARNING] generate_repo_map produced {len(tree_lines)} lines. Verify IGNORE_DIRS is filtering bin/obj/packages.")
    
    return result

# ---------------------------------------------------------------------------
# Tool 5 -- get_symbol_definition
# ---------------------------------------------------------------------------
@tool
@with_timeout(20)
def get_symbol_definition(symbol: str, directory: str) -> str:
    """Finds the definition of a specific class or function symbol in the codebase.
    Uses regex as a fallback to locate 'def symbol', 'class symbol', 'function symbol', etc.

    Args:
        symbol: The class or function name to locate.
        directory: Root directory to search.

    Returns:
        The line and file where the symbol is defined, or an error/not found message.
    """
    import re
    root = Path(directory).resolve()
    if not root.is_dir():
        return f"Error: '{directory}' is not a valid directory."

    # PATH CONFINEMENT
    err = _check_confinement(root, "directory")
    if err:
        return err

    ignore_dirs, matcher = _get_effective_ignore_parts(root)

    # Generic patterns to find definitions for various languages
    pattern1 = re.compile(r'\b(?:class|def|function|interface|struct|enum)\s+' + re.escape(symbol) + r'\b')
    pattern2 = re.compile(r'\b(?:const|let|var)\s+' + re.escape(symbol) + r'\s*=\s*(?:function|\()')
    pattern3 = re.compile(r'\b' + re.escape(symbol) + r'\s*\(') # Fallback loosely finds method definitions
    
    supported_exts = {".py", ".cs", ".xaml", ".cpp", ".h", ".js", ".ts", ".jsx", ".tsx", ".go", ".java"}
    matches = []
    
    for dirpath, _dirnames, filenames in os.walk(root):
        # ✅ UNIFIED FILTERING: Uses IGNORE_DIRS constant like all other tools
        rel_dir = Path(dirpath).relative_to(root)
        if _is_ignored(rel_dir, ignore_dirs, matcher):
            continue
            
        for fname in sorted(filenames):
            if not any(fname.endswith(ext) for ext in supported_exts):
                continue
                
            fpath = Path(dirpath, fname)
            rel = fpath.relative_to(root)
            if _is_ignored(rel, ignore_dirs, matcher):
                continue
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            
            for idx, line in enumerate(lines, start=1):
                if pattern1.search(line) or pattern2.search(line) or pattern3.search(line):
                    matches.append(f"{rel}:{idx}: {line.strip()}")
                    if len(matches) >= 30:
                        matches.append("[TRUNCATED -- 30 limit reached]")
                        return "\n".join(matches)

    if not matches:
        return f"No definition found for symbol '{symbol}'."
    return "\n".join(matches)


# ---------------------------------------------------------------------------
# Tool 6 -- generate_symbol_map
# ---------------------------------------------------------------------------
@tool
@with_timeout(20)
def generate_symbol_map(directory: str) -> str:
    """Generate a symbol-level map (classes, functions, methods) with line numbers.
    
    Uses regex to extract function/class definitions from all code files.
    Returns tab-separated entries: symbol_name | line_number | type | file
    
    Useful for Planner to understand repo structure without reading full files.
    Capped at 100 symbols to avoid overwhelming context.
    
    Args:
        directory: Root directory to scan.
    
    Returns:
        Tab-separated symbol list, or error message if generation fails.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        return f"Error: '{directory}' is not a valid directory."
    
    # PATH CONFINEMENT
    err = _check_confinement(root, "directory")
    if err:
        return err
    
    ignore_dirs, matcher = _get_effective_ignore_parts(root)
    supported_exts = {".py", ".cs", ".xaml", ".cpp", ".h", ".js", ".ts", ".jsx", ".tsx", ".go", ".java"}
    
    symbols: list[tuple[str, int, str, str]] = []  # (name, line, type, file)
    
    # Regex patterns for different languages
    class_pattern = re.compile(r'^\s*(?:class|interface|struct)\s+(\w+)')
    def_pattern = re.compile(r'^\s*(?:def|function|func|private|public)*\s*(?:\w+\s+)*(\w+)\s*\(')
    
    for dirpath, _dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        if _is_ignored(rel_dir, ignore_dirs, matcher):
            continue
        
        for fname in sorted(filenames):
            if not any(fname.endswith(ext) for ext in supported_exts):
                continue
            
            fpath = Path(dirpath, fname)
            rel = fpath.relative_to(root)
            if _is_ignored(rel, ignore_dirs, matcher):
                continue
            
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            
            for idx, line in enumerate(lines, start=1):
                # Detect class/interface/struct
                class_match = class_pattern.search(line)
                if class_match:
                    symbols.append((class_match.group(1), idx, "class", str(rel).replace("\\", "/")))
                    if len(symbols) >= 100:
                        break
                
                # Detect function/method definitions (exclude class definitions already caught)
                if not class_match:
                    def_match = def_pattern.search(line)
                    if def_match:
                        func_name = def_match.group(1)
                        # Skip if it's obviously not a function  (e.g., 'if', 'for', 'while')
                        if func_name not in ('if', 'for', 'while', 'switch', 'catch', 'elif', 'else', 'elif'):
                            symbols.append((func_name, idx, "function", str(rel).replace("\\", "/")))
                            if len(symbols) >= 100:
                                break
                
                if len(symbols) >= 100:
                    break
            
            if len(symbols) >= 100:
                break
    
    if not symbols:
        return "(No symbols found in codebase)"
    
    # Format output: symbol_name | line_number | type | file
    formatted = []
    for name, line, sym_type, file in sorted(symbols, key=lambda x: x[3] + ":" + str(x[1]))[:100]:
        formatted.append(f"{name:40} | {line:5} | {sym_type:10} | {file}")
    
    return "\n".join(formatted)


# ---------------------------------------------------------------------------
# Convenience list for .bind_tools()
# ---------------------------------------------------------------------------
REPO_TOOLS = [list_files, search_code, read_file, generate_repo_map, get_symbol_definition, generate_symbol_map]
