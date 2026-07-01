"""In-process runtime context shared across nodes and tools.

Provides centralized access to all intelligence singletons so that
downstream nodes (Planner, Coder, Debugger, Reviewer, Verifier,
Context Curator) all consume the same Repository Intelligence rather
than independently building their own indexes.
"""

from __future__ import annotations

from typing import Any

_ENVIRONMENT_CONFIG: dict[str, Any] = {}
_KNOWLEDGE_GRAPH: Any = None
_MODEL_ROUTER: Any = None
_PLUGIN_REGISTRY: Any = None
_EMBEDDING_INDEX: Any = None
_HYBRID_RETRIEVER: Any = None
_REPO_PROFILE: Any = None
_LSP_BRIDGE: Any = None
_EXECUTION_INTELLIGENCE: dict[str, Any] = {}


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


def set_embedding_index(index: Any) -> None:
    global _EMBEDDING_INDEX
    _EMBEDDING_INDEX = index


def get_embedding_index() -> Any:
    return _EMBEDDING_INDEX


def set_hybrid_retriever(retriever: Any) -> None:
    global _HYBRID_RETRIEVER
    _HYBRID_RETRIEVER = retriever


def get_hybrid_retriever() -> Any:
    return _HYBRID_RETRIEVER


def set_repo_profile(profile: Any) -> None:
    global _REPO_PROFILE
    _REPO_PROFILE = profile


def get_repo_profile() -> Any:
    return _REPO_PROFILE


def set_lsp_bridge(bridge: Any) -> None:
    global _LSP_BRIDGE
    _LSP_BRIDGE = bridge


def get_lsp_bridge() -> Any:
    return _LSP_BRIDGE


def set_execution_intelligence(data: dict[str, Any]) -> None:
    global _EXECUTION_INTELLIGENCE
    _EXECUTION_INTELLIGENCE = dict(data)


def get_execution_intelligence() -> dict[str, Any]:
    return dict(_EXECUTION_INTELLIGENCE)


def get_repo_intelligence() -> dict[str, Any]:
    """Single-call accessor for all intelligence data.

    Every downstream component should use this instead of
    independently searching files.
    """
    return {
        "graph": _KNOWLEDGE_GRAPH,
        "embeddings": _EMBEDDING_INDEX,
        "retriever": _HYBRID_RETRIEVER,
        "profile": _REPO_PROFILE,
        "lsp": _LSP_BRIDGE,
    }


def reset_runtime_context() -> None:
    """Selectively reset per-repository state, preserving reusable assets like model routers and plugin registries."""
    global _ENVIRONMENT_CONFIG, _KNOWLEDGE_GRAPH, _EMBEDDING_INDEX, _HYBRID_RETRIEVER, _REPO_PROFILE, _LSP_BRIDGE, _EXECUTION_INTELLIGENCE
    _ENVIRONMENT_CONFIG = {}
    _KNOWLEDGE_GRAPH = None
    _EMBEDDING_INDEX = None
    _HYBRID_RETRIEVER = None
    _REPO_PROFILE = None
    _LSP_BRIDGE = None
    _EXECUTION_INTELLIGENCE = {}

