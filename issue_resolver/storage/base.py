"""Storage data classes shared across all storage backends.

These are the canonical shapes for data flowing into and out of the
``MemoryStore``.  Agents never interact with storage internals — they
use these data classes exclusively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RepoSummary:
    """Cached repository-level analysis results."""

    repo_path: str
    commit_hash: str = ""
    language: str = ""
    framework: str = ""
    architecture_pattern: str = ""
    test_framework: str = ""
    build_command: str = ""
    test_command: str = ""
    formatter: str = ""
    linter: str = ""
    ci_system: str = ""
    entrypoints: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    critical_files: list[str] = field(default_factory=list)
    coding_conventions: dict[str, str] = field(default_factory=dict)
    complexity_estimate: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "commit_hash": self.commit_hash,
            "language": self.language,
            "framework": self.framework,
            "architecture_pattern": self.architecture_pattern,
            "test_framework": self.test_framework,
            "build_command": self.build_command,
            "test_command": self.test_command,
            "formatter": self.formatter,
            "linter": self.linter,
            "ci_system": self.ci_system,
            "entrypoints": self.entrypoints,
            "modules": self.modules,
            "critical_files": self.critical_files,
            "coding_conventions": self.coding_conventions,
            "complexity_estimate": self.complexity_estimate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepoSummary":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FixRecord:
    """A single past issue fix attempt (success or failure)."""

    issue_id: str
    issue_text: str
    issue_category: str = ""
    repo_path: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    patch: str = ""
    test_result: str = ""
    files_changed: list[str] = field(default_factory=list)
    success: bool = False
    failures: list[str] = field(default_factory=list)
    lessons_learned: str = ""
    timestamp: str = ""
    models_used: list[str] = field(default_factory=list)
    tokens_consumed: int = 0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "issue_text": self.issue_text,
            "issue_category": self.issue_category,
            "repo_path": self.repo_path,
            "plan": self.plan,
            "patch": self.patch,
            "test_result": self.test_result,
            "files_changed": self.files_changed,
            "success": self.success,
            "failures": self.failures,
            "lessons_learned": self.lessons_learned,
            "timestamp": self.timestamp,
            "models_used": self.models_used,
            "tokens_consumed": self.tokens_consumed,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FixRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SymbolRecord:
    """Cached symbol information (function, class, variable)."""

    name: str
    kind: str  # "function", "class", "method", "variable"
    file_path: str
    line_number: int = 0
    signature: str = ""
    docstring: str = ""
    parent_class: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "signature": self.signature,
            "docstring": self.docstring,
            "parent_class": self.parent_class,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SymbolRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
