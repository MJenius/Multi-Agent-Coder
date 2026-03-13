"""In-process runtime context shared across nodes and tools."""

from __future__ import annotations

from typing import Any

_ENVIRONMENT_CONFIG: dict[str, Any] = {}


def set_environment_config(config: dict[str, Any]) -> None:
    global _ENVIRONMENT_CONFIG
    _ENVIRONMENT_CONFIG = dict(config)


def get_environment_config() -> dict[str, Any]:
    return dict(_ENVIRONMENT_CONFIG)
