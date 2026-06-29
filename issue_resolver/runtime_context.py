"""In-process runtime context shared across nodes and tools."""

from __future__ import annotations

from typing import Any

_ENVIRONMENT_CONFIG: dict[str, Any] = {}
_KNOWLEDGE_GRAPH: Any = None
_MODEL_ROUTER: Any = None
_PLUGIN_REGISTRY: Any = None


def set_environment_config(config: dict[str, Any]) -> None:
    global _ENVIRONMENT_CONFIG
    _ENVIRONMENT_CONFIG = dict(config)


def get_environment_config() -> dict[str, Any]:
    return dict(_ENVIRONMENT_CONFIG)


def set_knowledge_graph(graph: Any) -> None:
    global _KNOWLEDGE_GRAPH
    _KNOWLEDGE_GRAPH = graph


def get_knowledge_graph() -> Any:
    return _KNOWLEDGE_GRAPH


def set_model_router(router: Any) -> None:
    global _MODEL_ROUTER
    _MODEL_ROUTER = router


def get_model_router() -> Any:
    return _MODEL_ROUTER


def set_plugin_registry(registry: Any) -> None:
    global _PLUGIN_REGISTRY
    _PLUGIN_REGISTRY = registry


def get_plugin_registry() -> Any:
    return _PLUGIN_REGISTRY

