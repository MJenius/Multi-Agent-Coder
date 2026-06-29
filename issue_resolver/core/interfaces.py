"""Abstract interfaces for the Multi-Agent Issue Resolver.

Every major subsystem is governed by an abstract interface defined here.
Concrete implementations live in their respective subpackages. This
decoupling allows swapping backends (storage, models, tools) without
changing agent logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Data transfer objects shared across subsystems
# ---------------------------------------------------------------------------


class Confidence(Enum):
    """Confidence level for retrieved context or agent decisions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_score(cls, score: float) -> "Confidence":
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.45:
            return cls.MEDIUM
        return cls.LOW


@dataclass
class ToolResult:
    """Standardised result from any tool invocation."""

    success: bool
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Standardised response from any model invocation."""

    content: str
    model_name: str
    role: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    reasoning_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredResult:
    """A retrieved item with its composite confidence score."""

    path: str
    content: str
    score: float
    confidence: Confidence = field(init=False)
    signal_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = Confidence.from_score(self.score)


@dataclass
class VerificationResult:
    """Outcome of a single verification step."""

    step_name: str
    passed: bool
    output: str
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract: Tool
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Abstract tool that agents use to interact with the environment.

    All tools (filesystem, git, docker, LSP, embedding search, AST editor,
    GitHub, benchmark runner) implement this interface so they can be
    discovered and invoked uniformly by any agent.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (e.g. ``"filesystem"``)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for prompt injection."""

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Run the tool with the given parameters."""


# ---------------------------------------------------------------------------
# Abstract: MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore(ABC):
    """Persistent storage backend for repository memory.

    Initially implemented as JSON.  The interface is designed so that
    SQLite, DuckDB, Chroma, LanceDB, or PostgreSQL can be swapped in
    by implementing this interface — without changing any agent logic.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Retrieve a value by key.  Returns ``None`` if missing."""

    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        """Store *value* under *key*, overwriting any existing entry."""

    @abstractmethod
    def query(self, prefix: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return all records whose key starts with *prefix*,
        optionally filtered by metadata fields."""

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys that start with *prefix*."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete the entry under *key*.  Return ``True`` if it existed."""


# ---------------------------------------------------------------------------
# Abstract: ModelProvider
# ---------------------------------------------------------------------------


class ModelProvider(ABC):
    """Wraps a single LLM provider (NVIDIA, Anthropic, local Ollama …).

    ``ModelRouter`` selects the right provider for a given role and
    delegates invocation to it.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """E.g. ``"nvidia"``, ``"openai"``, ``"ollama"``."""

    @abstractmethod
    def invoke(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Send *messages* to the model and return a structured response."""


# ---------------------------------------------------------------------------
# Abstract: RetrievalStrategy
# ---------------------------------------------------------------------------


class RetrievalStrategy(ABC):
    """Pluggable retrieval backend.

    Implementations may use keyword search, semantic embeddings, graph
    traversal, or a hybrid of all three.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """E.g. ``"hybrid"``, ``"keyword"``, ``"semantic"``."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        repo_path: str,
        top_k: int = 10,
        context: dict[str, Any] | None = None,
    ) -> list[ScoredResult]:
        """Return the *top_k* most relevant results for *query*."""


# ---------------------------------------------------------------------------
# Abstract: VerificationStep
# ---------------------------------------------------------------------------


class VerificationStep(ABC):
    """A single stage in the verification pipeline (lint, type-check, …).

    New stages can be registered via the plugin system without modifying
    core orchestration.
    """

    @property
    @abstractmethod
    def step_name(self) -> str:
        """E.g. ``"ruff_lint"``, ``"mypy_typecheck"``."""

    @property
    @abstractmethod
    def language(self) -> str:
        """Language this step applies to (``"python"``, ``"*"`` for all)."""

    @abstractmethod
    def run(self, repo_path: str, changed_files: list[str]) -> VerificationResult:
        """Execute the verification step and return structured results."""

    def is_available(self, repo_path: str) -> bool:
        """Return ``True`` if the tool backing this step is installed."""
        return True


# ---------------------------------------------------------------------------
# Abstract: AgentPlugin
# ---------------------------------------------------------------------------


class AgentPlugin(ABC):
    """Registration point for extending the system with new capabilities.

    A plugin may contribute agents, tools, retrieval strategies, model
    providers, and verification steps.  Plugins are discovered at startup
    from ``issue_resolver/plugins/`` or registered via ``plugins.json``.
    """

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Unique plugin identifier."""

    def register_tools(self) -> list[Tool]:
        """Return any tools this plugin provides."""
        return []

    def register_verification_steps(self) -> list[VerificationStep]:
        """Return any verification steps this plugin provides."""
        return []

    def register_retrieval_strategies(self) -> list[RetrievalStrategy]:
        """Return any retrieval strategies this plugin provides."""
        return []

    def register_model_providers(self) -> list[ModelProvider]:
        """Return any model providers this plugin provides."""
        return []
