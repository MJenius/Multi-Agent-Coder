"""
Repository Tools -- Local codebase search utilities for the Researcher agent.

Each function is decorated with @tool so it can be bound to ChatOllama
via .bind_tools().  All tools include RAM-safety guards:
  - list_files:   caps output at 200 files
  - search_code:  caps output at 30 matches
  - read_file:    truncates at 500 lines
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Tool 1 -- list_files
# ---------------------------------------------------------------------------
@tool
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
        # Skip hidden dirs and common build dirs
        parts = Path(dirpath).relative_to(root).parts
        if any(p.startswith(".") or p in ("__pycache__", "node_modules", "bin", "obj", "build", "dist", "venv") for p in parts):
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
        parts = Path(dirpath).relative_to(root).parts
        if any(p.startswith(".") or p in ("__pycache__", "node_modules", "bin", "obj", "build", "dist", "venv") for p in parts):
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
def generate_repo_map(directory: str) -> str:
    """Generates a high-level tree-view of the directory structure (ignoring common build/virtualenv folders)
    and includes the contents of README.md if present. This gives a Map of the codebase.

    Args:
        directory: Root directory to map.

    Returns:
        A string containing the tree structure and README content (if any).
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        return f"Error: '{directory}' is not a valid directory."

    ignore_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", "bin", "obj", "build", "dist", ".idea", ".vscode"}
    
    tree_lines = []
    
    def walk_tree(dir_path: Path, prefix: str = ""):
        if len(tree_lines) > 200:
            return
            
        try:
            items = sorted(os.listdir(dir_path))
        except OSError:
            return
            
        # Filter items
        dirs = []
        files = []
        for item in items:
            if item in ignore_dirs or item.startswith("."):
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
            walk_tree(dir_path / d, prefix + extension)
            
        for i, f in enumerate(files):
            is_last = (i == len(files) - 1)
            connector = "└── " if is_last else "├── "
            tree_lines.append(f"{prefix}{connector}{f}")
            if len(tree_lines) >= 200: return
            
    tree_lines.append(str(root))
    walk_tree(root)
    
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

    return f"Repository Map:\n{map_str}{readme_content}"

# ---------------------------------------------------------------------------
# Tool 5 -- get_symbol_definition
# ---------------------------------------------------------------------------
@tool
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
        parts = Path(dirpath).relative_to(root).parts
        if any(p.startswith(".") or p in ("__pycache__", "node_modules", "bin", "obj", "build", "dist", "venv") for p in parts):
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
