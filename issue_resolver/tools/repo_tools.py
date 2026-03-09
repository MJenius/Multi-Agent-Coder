"""
Repository Tools -- Local codebase search utilities for the Researcher agent.

Each function is decorated with @tool so it can be bound to ChatOllama
via .bind_tools().  All tools include RAM-safety guards:
  - list_files:   caps output at 200 files
  - search_code:  caps output at 30 matches
  - read_file:    truncates at 500 lines
  - ALL TOOLS:    timeout protection (30s default) to prevent hangs on large repos
"""

from __future__ import annotations

import os
import time
import threading
from functools import wraps
from pathlib import Path

from langchain_core.tools import tool


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
    
    # HARD GUARD: If it's a file, return an explicit error string to guide the LLM
    if root.is_file():
        return f"ERROR: '{directory}' is a FILE. You must use the 'read_file' tool to see its contents."
        
    if not root.is_dir():
        return f"Error: '{directory}' is not a valid directory."

    supported_exts = {".py", ".cs", ".xaml", ".cpp", ".h", ".js", ".ts", ".jsx", ".tsx", ".go", ".java"}
    code_files: list[str] = []
    
    for dirpath, _dirnames, filenames in os.walk(root):
        # ✅ UNIFIED FILTERING: All tools use the same IGNORE_DIRS constant
        parts = Path(dirpath).relative_to(root).parts
        if any(p in IGNORE_DIRS or p.startswith(".") for p in parts):
            continue
            
        for fname in sorted(filenames):
            if any(fname.endswith(ext) for ext in supported_exts):
                rel = Path(dirpath, fname).relative_to(root)
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

    supported_exts = {".py", ".cs", ".xaml", ".cpp", ".h", ".js", ".ts", ".jsx", ".tsx", ".go", ".java"}
    matches: list[str] = []
    
    for dirpath, _dirnames, filenames in os.walk(root):
        # ✅ UNIFIED FILTERING: Uses IGNORE_DIRS to prevent grep drowning
        parts = Path(dirpath).relative_to(root).parts
        if any(p in IGNORE_DIRS or p.startswith(".") for p in parts):
            continue
            
        for fname in sorted(filenames):
            if not any(fname.endswith(ext) for ext in supported_exts):
                continue
                
            fpath = Path(dirpath, fname)
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
                
            for idx, line in enumerate(lines, start=1):
                if query.lower() in line.lower():
                    rel = fpath.relative_to(root)
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
    if not fpath.is_file():
        return f"Error: '{file_path}' is not a valid file."

    try:
        lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Error reading file: {exc}"

    truncated = len(lines) > 500
    output_lines = lines[:500]

    # Prepend line numbers for easy reference
    numbered = [f"{i}: {l}" for i, l in enumerate(output_lines, start=1)]
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

    # ✅ SAFETY: Convert max_depth to int if LLM passes it as string
    max_depth = int(max_depth) if isinstance(max_depth, str) else max_depth

    # ✅ UNIFIED FILTERING: Uses the module-level IGNORE_DIRS constant
    # This ensures consistent behavior across ALL repo tools (list_files, search_code, etc.)
    
    tree_lines = []
    
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
            
        # Filter items: SKIP ignored directories and hidden folders
        dirs = []
        files = []
        for item in items:
            # ✅ UNIFIED FILTER: Prevents recursing into bin/, obj/, .vs, packages, etc.
            if item in IGNORE_DIRS or item.startswith("."):
                continue
            p = dir_path / item
            if p.is_dir():
                dirs.append(item)
            else:
                files.append(item)
                
        for i, d in enumerate(dirs):
            is_last = (i == len(dirs) - 1) and (len(files) == 0)
            connector = "└── " if is_last else "├── "
            tree_lines.append(f"{prefix}{connector}{d}/")
            if len(tree_lines) >= 200: return
            extension = "    " if is_last else "│   "
            walk_tree(dir_path / d, prefix + extension, current_depth + 1)
            
        # Only show files at max_depth level, not deeper
        if current_depth < max_depth:
            for i, f in enumerate(files):
                is_last = (i == len(files) - 1)
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{f}")
                if len(tree_lines) >= 200: return
            
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

    # Generic patterns to find definitions for various languages
    pattern1 = re.compile(r'\b(?:class|def|function|interface|struct|enum)\s+' + re.escape(symbol) + r'\b')
    pattern2 = re.compile(r'\b(?:const|let|var)\s+' + re.escape(symbol) + r'\s*=\s*(?:function|\()')
    pattern3 = re.compile(r'\b' + re.escape(symbol) + r'\s*\(') # Fallback loosely finds method definitions
    
    supported_exts = {".py", ".cs", ".xaml", ".cpp", ".h", ".js", ".ts", ".jsx", ".tsx", ".go", ".java"}
    matches = []
    
    for dirpath, _dirnames, filenames in os.walk(root):
        # ✅ UNIFIED FILTERING: Uses IGNORE_DIRS constant like all other tools
        parts = Path(dirpath).relative_to(root).parts
        if any(p in IGNORE_DIRS or p.startswith(".") for p in parts):
            continue
            
        for fname in sorted(filenames):
            if not any(fname.endswith(ext) for ext in supported_exts):
                continue
                
            fpath = Path(dirpath, fname)
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            
            for idx, line in enumerate(lines, start=1):
                if pattern1.search(line) or pattern2.search(line) or pattern3.search(line):
                    rel = fpath.relative_to(root)
                    matches.append(f"{rel}:{idx}: {line.strip()}")
                    if len(matches) >= 30:
                        matches.append("[TRUNCATED -- 30 limit reached]")
                        return "\n".join(matches)

    if not matches:
        return f"No definition found for symbol '{symbol}'."
    return "\n".join(matches)

# ---------------------------------------------------------------------------
# Convenience list for .bind_tools()
# ---------------------------------------------------------------------------
REPO_TOOLS = [list_files, search_code, read_file, generate_repo_map, get_symbol_definition]
