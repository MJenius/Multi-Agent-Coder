"""Convenience wrappers exposing LSP operations as reusable tool-like functions.

These are consumed by the Localizer, Debugger, and Planner.  They
automatically obtain the LSP bridge from ``runtime_context`` and
fall back to ``None`` / empty lists when LSP is unavailable.
"""

from __future__ import annotations

from typing import Any

import issue_resolver.runtime_context as runtime_context


def lsp_find_definition(
    symbol: str, file_path: str, line: int, character: int = 0,
) -> dict[str, Any] | None:
    """Find the definition of *symbol* via the LSP bridge.

    Returns ``{file, line, end_line}`` or ``None`` if LSP is
    unavailable or the definition cannot be resolved.
    """
    bridge = runtime_context.get_lsp_bridge()
    if bridge is None or not bridge.is_available:
        return None
    try:
        return bridge.find_definition(symbol, file_path, line, character)
    except Exception:
        return None


def lsp_find_references(
    symbol: str, file_path: str, line: int, character: int = 0,
) -> list[dict[str, Any]]:
    """Find all references to *symbol* via the LSP bridge.

    Returns ``[{file, line, context}]`` or ``[]`` if LSP is
    unavailable.
    """
    bridge = runtime_context.get_lsp_bridge()
    if bridge is None or not bridge.is_available:
        return []
    try:
        return bridge.find_references(symbol, file_path, line, character)
    except Exception:
        return []


def lsp_get_type_info(
    file_path: str, line: int, character: int,
) -> str | None:
    """Get type/docstring info for a position via the LSP bridge.

    Returns a string with type information or ``None``.
    """
    bridge = runtime_context.get_lsp_bridge()
    if bridge is None or not bridge.is_available:
        return None
    try:
        return bridge.find_hover(file_path, line, character)
    except Exception:
        return None


def is_lsp_available() -> bool:
    """Check if the LSP bridge is available and initialised."""
    bridge = runtime_context.get_lsp_bridge()
    return bridge is not None and bridge.is_available
