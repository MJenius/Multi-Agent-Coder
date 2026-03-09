"""
Logger utility for Phase 5 Observability.
Provides tools to safely add actions to the history agent state
and mirrors full texts using Python's standard `logging` module.
"""

from __future__ import annotations

import logging
import datetime
import json
import re

# Set up standard logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
sys_logger = logging.getLogger("issue_resolver")

# ---------------------------------------------------------------------------
# Secret redaction — prevents PATs from leaking into logs / state history
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r'ghp_[A-Za-z0-9]{36,}'),           # GitHub PAT (classic)
    re.compile(r'github_pat_[A-Za-z0-9_]{22,}'),    # GitHub PAT (fine-grained)
    re.compile(r'gho_[A-Za-z0-9]{36,}'),             # GitHub OAuth token
    re.compile(r'ghs_[A-Za-z0-9]{36,}'),             # GitHub App token
]


def _redact_secrets(text: str) -> str:
    """Replace known secret patterns with a masked placeholder."""
    for pat in _SECRET_PATTERNS:
        text = pat.sub("****", text)
    return text


def append_to_history(node_name: str, action: str, content: str, max_length: int = 500) -> list[dict]:
    """
    Format a history entry.

    For state history, the content is truncated to save RAM.
    The full untruncated content is printed to the console log.
    
    Returns a LIST of exactly one dictionary. Because LangGraph uses `operator.add`
    on the `Annotated[list[dict], operator.add]` state field, returning
    a list allows it to append properly instead of overwriting.
    """
    # 0. Redact secrets before any output
    content = _redact_secrets(content)

    # 1. Console Log (Full Text)
    # We prefix multi-line content for better readability in the terminal
    if "\n" in content:
        sys_logger.info(f"[{node_name}] {action}:\n{content}")
    else:
        sys_logger.info(f"[{node_name}] {action}: {content}")

    # 2. State History (Truncated)
    truncated_content = content
    if len(content) > max_length:
        truncated_content = content[:max_length] + f"\n... [Truncated {len(content) - max_length} chars]"

    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "node": node_name,
        "action": action,
        "content": truncated_content,
    }
    
    return [entry]


def get_token_estimate(text: str) -> int:
    """
    A very rough character-based token estimation.
    Usually 1 token ~= 4 chars in English text.
    """
    return len(text) // 4
