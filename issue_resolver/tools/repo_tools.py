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
    """Recursively list all .py files in a directory to understand project structure.

    Args:
        directory: Absolute or relative path to the root directory to scan.

    Returns:
        A newline-separated list of .py file paths relative to *directory*,
        capped at 200 entries.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        return f"Error: '{directory}' is not a valid directory."

    py_files: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        # Skip hidden dirs and __pycache__
        parts = Path(dirpath).relative_to(root).parts
        if any(p.startswith(".") or p == "__pycache__" for p in parts):
            continue
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                rel = Path(dirpath, fname).relative_to(root)
                py_files.append(str(rel))
                if len(py_files) >= 200:
                    py_files.append("[TRUNCATED -- 200 file limit reached]")
                    return "\n".join(py_files)

    if not py_files:
        return "(no .py files found)"
    return "\n".join(py_files)


# ---------------------------------------------------------------------------
# Tool 2 -- search_code
# ---------------------------------------------------------------------------
@tool
def search_code(query: str, directory: str) -> str:
    """Search for a string in all .py files under a directory (grep-like).

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

    matches: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        parts = Path(dirpath).relative_to(root).parts
        if any(p.startswith(".") or p == "__pycache__" for p in parts):
            continue
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
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
# Convenience list for .bind_tools()
# ---------------------------------------------------------------------------
REPO_TOOLS = [list_files, search_code, read_file]
