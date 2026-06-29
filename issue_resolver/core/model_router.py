"""Dynamic model routing engine.

Model selection is **not fixed** — it considers issue type, repository
language, repository size, token budget, retry count, and previous model
failures.  Routing rules are defined in ``model_routing.json`` and can
be changed without modifying source code.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from issue_resolver.core.interfaces import ModelResponse


@dataclass
class ModelConfig:
    """Configuration for a single model assignment."""

    model: str
    api_key_env: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    temperature: float = 0.0
    max_tokens: int = 4096
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass
class RoutingOverride:
    """A conditional override for model selection."""

    condition: dict[str, str]  # e.g. {"repo_language": "python", "retry_count": ">1"}
    model: str
    api_key_env: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoleConfig:
    """Full routing configuration for a single agent role."""

    default: ModelConfig
    overrides: list[RoutingOverride] = field(default_factory=list)
    fallback_models: list[str] = field(default_factory=list)


class ModelRouter:
    """Selects the right model for a given agent role based on context.

    Usage::

        router = ModelRouter.from_file("model_routing.json")
        config = router.resolve("planner", context={"repo_language": "python"})
        # config.model, config.api_key, config.temperature, ...
    """

    def __init__(self) -> None:
        self._roles: dict[str, RoleConfig] = {}
        self._failed_models: set[str] = set()
        self._quota_exceeded: set[str] = set()

    # ----- loading -----

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelRouter":
        """Load routing configuration from a JSON file."""
        router = cls()
        path = Path(path)
        if not path.is_file():
            print(f"[ModelRouter] Config file not found: {path}. Using empty config.")
            return router

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[ModelRouter] Failed to load config: {exc}")
            return router

        for role_name, role_data in data.get("roles", {}).items():
            default_data = role_data.get("default", {})
            default_config = ModelConfig(
                model=default_data.get("model", ""),
                api_key_env=default_data.get("api_key_env", ""),
                base_url=default_data.get("base_url", "https://integrate.api.nvidia.com/v1"),
                temperature=default_data.get("temperature", 0.0),
                max_tokens=default_data.get("max_tokens", 4096),
                extra_body=default_data.get("extra_body", {}),
            )

            overrides = []
            for override_data in role_data.get("overrides", []):
                overrides.append(RoutingOverride(
                    condition=override_data.get("condition", {}),
                    model=override_data.get("model", ""),
                    api_key_env=override_data.get("api_key_env", ""),
                    temperature=override_data.get("temperature"),
                    max_tokens=override_data.get("max_tokens"),
                    extra_body=override_data.get("extra_body", {}),
                ))

            fallbacks = role_data.get("fallback_models", [])

            router._roles[role_name] = RoleConfig(
                default=default_config,
                overrides=overrides,
                fallback_models=fallbacks,
            )

        print(f"[ModelRouter] Loaded {len(router._roles)} role configurations")
        return router

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRouter":
        """Load from an already-parsed dict (useful for testing)."""
        # Write to temp and delegate — keeps logic centralised
        import tempfile
        router = cls()
        for role_name, role_data in data.get("roles", {}).items():
            default_data = role_data.get("default", {})
            default_config = ModelConfig(
                model=default_data.get("model", ""),
                api_key_env=default_data.get("api_key_env", ""),
                base_url=default_data.get("base_url", "https://integrate.api.nvidia.com/v1"),
                temperature=default_data.get("temperature", 0.0),
                max_tokens=default_data.get("max_tokens", 4096),
                extra_body=default_data.get("extra_body", {}),
            )
            overrides = []
            for override_data in role_data.get("overrides", []):
                overrides.append(RoutingOverride(
                    condition=override_data.get("condition", {}),
                    model=override_data.get("model", ""),
                    api_key_env=override_data.get("api_key_env", ""),
                    temperature=override_data.get("temperature"),
                    max_tokens=override_data.get("max_tokens"),
                    extra_body=override_data.get("extra_body", {}),
                ))
            fallbacks = role_data.get("fallback_models", [])
            router._roles[role_name] = RoleConfig(
                default=default_config, overrides=overrides, fallback_models=fallbacks,
            )
        return router

    # ----- resolution -----

    def _matches_condition(self, condition: dict[str, str], context: dict[str, Any]) -> bool:
        """Check if all condition predicates match the given context."""
        for key, expected in condition.items():
            actual = context.get(key)
            if actual is None:
                return False

            # Support simple comparison operators
            if isinstance(expected, str) and expected.startswith(">"):
                try:
                    threshold = int(expected[1:])
                    if int(actual) <= threshold:
                        return False
                except (ValueError, TypeError):
                    return False
            elif isinstance(expected, str) and expected.startswith("<"):
                try:
                    threshold = int(expected[1:])
                    if int(actual) >= threshold:
                        return False
                except (ValueError, TypeError):
                    return False
            else:
                if str(actual).lower() != str(expected).lower():
                    return False
        return True

    def resolve(
        self,
        role: str,
        context: dict[str, Any] | None = None,
    ) -> ModelConfig:
        """Resolve the best model config for *role* given *context*.

        Context keys that influence routing:
        - ``issue_type``: bug, feature, security, ...
        - ``repo_language``: python, javascript, ...
        - ``repo_size``: small, medium, large
        - ``token_budget``: remaining token budget
        - ``retry_count``: how many retries have occurred
        - ``previous_failures``: list of models that failed
        """
        ctx = context or {}
        role_cfg = self._roles.get(role)
        if not role_cfg:
            # Fallback: return a default config
            return ModelConfig(
                model="meta/llama-3.3-70b-instruct",
                api_key_env="NVIDIA_API_KEY_TIER2",
            )

        # Check overrides first (order matters — first match wins)
        for override in role_cfg.overrides:
            if self._matches_condition(override.condition, ctx):
                model = override.model
                if model in self._failed_models or model in self._quota_exceeded:
                    continue
                return ModelConfig(
                    model=model,
                    api_key_env=override.api_key_env or role_cfg.default.api_key_env,
                    base_url=role_cfg.default.base_url,
                    temperature=override.temperature if override.temperature is not None else role_cfg.default.temperature,
                    max_tokens=override.max_tokens if override.max_tokens is not None else role_cfg.default.max_tokens,
                    extra_body=override.extra_body or role_cfg.default.extra_body,
                )

        # Default model (skip if failed)
        if role_cfg.default.model not in self._failed_models and role_cfg.default.model not in self._quota_exceeded:
            return role_cfg.default

        # Try fallbacks
        for fallback_model in role_cfg.fallback_models:
            if fallback_model not in self._failed_models and fallback_model not in self._quota_exceeded:
                return ModelConfig(
                    model=fallback_model,
                    api_key_env=role_cfg.default.api_key_env,
                    base_url=role_cfg.default.base_url,
                    temperature=role_cfg.default.temperature,
                    max_tokens=role_cfg.default.max_tokens,
                    extra_body=role_cfg.default.extra_body,
                )

        # All models failed — return default anyway and let the caller handle the error
        return role_cfg.default

    def get_candidates(self, role: str) -> list[str]:
        """Return all candidate model names for *role* (default + fallbacks)."""
        role_cfg = self._roles.get(role)
        if not role_cfg:
            return []
        candidates = [role_cfg.default.model]
        candidates.extend(role_cfg.fallback_models)
        return [m for m in candidates if m not in self._failed_models and m not in self._quota_exceeded]

    # ----- failure tracking -----

    def mark_failed(self, model: str) -> None:
        """Mark a model as temporarily failed (will be skipped)."""
        self._failed_models.add(model)

    def mark_quota_exceeded(self, model: str) -> None:
        """Mark a model as having exceeded its quota."""
        self._quota_exceeded.add(model)

    def clear_failures(self) -> None:
        """Reset all failure/quota tracking."""
        self._failed_models.clear()
        self._quota_exceeded.clear()

    # ----- introspection -----

    def list_roles(self) -> list[str]:
        return list(self._roles.keys())

    def get_role_config(self, role: str) -> RoleConfig | None:
        return self._roles.get(role)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_GLOBAL_ROUTER: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    """Return the global model router (lazily initialised)."""
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None:
        # Try to load from project root
        candidates = [
            Path("model_routing.json"),
            Path(__file__).resolve().parent.parent.parent / "model_routing.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                _GLOBAL_ROUTER = ModelRouter.from_file(candidate)
                return _GLOBAL_ROUTER
        _GLOBAL_ROUTER = ModelRouter()
    return _GLOBAL_ROUTER


def set_model_router(router: ModelRouter) -> None:
    """Replace the global model router."""
    global _GLOBAL_ROUTER
    _GLOBAL_ROUTER = router
