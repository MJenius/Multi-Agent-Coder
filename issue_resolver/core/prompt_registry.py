"""Versioned prompt management for all agents.

Every agent prompt has a version identifier so that prompt changes
can be benchmarked independently from code changes.  Prompts can
be overridden via configuration without modifying source code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptEntry:
    """A single versioned prompt template."""

    name: str
    version: str
    template: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_id(self) -> str:
        return f"{self.name}_v{self.version}"


class PromptRegistry:
    """Central registry of all agent prompts with version tracking.

    Usage::

        registry = PromptRegistry()
        registry.register("planner", "2.0", "You are the Planner. ...")
        prompt = registry.get("planner")           # latest
        prompt = registry.get("planner", "1.0")     # specific version
    """

    def __init__(self) -> None:
        # {name -> {version -> PromptEntry}}
        self._prompts: dict[str, dict[str, PromptEntry]] = {}
        self._latest: dict[str, str] = {}  # name -> latest version

    # ----- registration -----

    def register(
        self,
        name: str,
        version: str,
        template: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register or replace a prompt version."""
        entry = PromptEntry(
            name=name,
            version=version,
            template=template,
            metadata=metadata or {},
        )
        self._prompts.setdefault(name, {})[version] = entry
        # Track latest by version string ordering
        if name not in self._latest or version > self._latest[name]:
            self._latest[name] = version

    # ----- retrieval -----

    def get(self, name: str, version: str | None = None) -> str:
        """Return the prompt template text.

        If *version* is ``None``, the latest registered version is returned.
        Raises ``KeyError`` if the prompt name or version is not found.
        """
        versions = self._prompts.get(name)
        if not versions:
            raise KeyError(f"No prompt registered with name '{name}'")

        target_version = version or self._latest.get(name)
        if target_version is None or target_version not in versions:
            raise KeyError(
                f"Prompt '{name}' version '{target_version}' not found. "
                f"Available: {list(versions.keys())}"
            )
        return versions[target_version].template

    def get_entry(self, name: str, version: str | None = None) -> PromptEntry:
        """Return the full ``PromptEntry`` (includes metadata)."""
        versions = self._prompts.get(name)
        if not versions:
            raise KeyError(f"No prompt registered with name '{name}'")
        target_version = version or self._latest.get(name)
        if target_version is None or target_version not in versions:
            raise KeyError(f"Prompt '{name}' version '{target_version}' not found")
        return versions[target_version]

    def get_version(self, name: str) -> str:
        """Return the latest version string for *name*."""
        if name not in self._latest:
            raise KeyError(f"No prompt registered with name '{name}'")
        return self._latest[name]

    def list_prompts(self) -> dict[str, list[str]]:
        """Return ``{name: [versions]}`` for all registered prompts."""
        return {
            name: sorted(versions.keys())
            for name, versions in self._prompts.items()
        }

    # ----- persistence (optional config override) -----

    def load_overrides(self, path: str | Path) -> int:
        """Load prompt overrides from a JSON file.

        File format::

            {
                "planner": {"version": "2.1", "template": "..."},
                "coder":   {"version": "3.0", "template": "..."}
            }

        Returns the number of prompts overridden.
        """
        path = Path(path)
        if not path.is_file():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0
        count = 0
        for name, entry in data.items():
            if isinstance(entry, dict) and "version" in entry and "template" in entry:
                self.register(name, entry["version"], entry["template"])
                count += 1
        return count


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_GLOBAL_PROMPT_REGISTRY = PromptRegistry()


def get_prompt_registry() -> PromptRegistry:
    """Return the global prompt registry singleton."""
    return _GLOBAL_PROMPT_REGISTRY
