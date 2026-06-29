"""Plugin discovery and registration system.

Plugins are discovered from ``issue_resolver/plugins/`` at startup.
Each plugin is a Python package containing a ``plugin.py`` module with
a ``register(registry)`` function.  Alternatively, a ``plugins.json``
manifest can declare explicit plugin paths and ordering.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from issue_resolver.core.interfaces import (
    AgentPlugin,
    ModelProvider,
    RetrievalStrategy,
    Tool,
    VerificationStep,
)


class PluginRegistry:
    """Central registry for all discovered plugins and their contributions.

    Usage::

        registry = PluginRegistry()
        registry.discover("issue_resolver/plugins")
        tools = registry.get_tools()
    """

    def __init__(self) -> None:
        self._plugins: dict[str, AgentPlugin] = {}
        self._tools: dict[str, Tool] = {}
        self._verification_steps: list[VerificationStep] = []
        self._retrieval_strategies: dict[str, RetrievalStrategy] = {}
        self._model_providers: dict[str, ModelProvider] = {}

    # ----- registration -----

    def register_plugin(self, plugin: AgentPlugin) -> None:
        """Manually register a plugin and its contributions."""
        name = plugin.plugin_name
        if name in self._plugins:
            print(f"[PluginRegistry] Replacing existing plugin '{name}'")
        self._plugins[name] = plugin

        for tool in plugin.register_tools():
            self._tools[tool.name] = tool

        self._verification_steps.extend(plugin.register_verification_steps())

        for strategy in plugin.register_retrieval_strategies():
            self._retrieval_strategies[strategy.strategy_name] = strategy

        for provider in plugin.register_model_providers():
            self._model_providers[provider.provider_name] = provider

    def register_tool(self, tool: Tool) -> None:
        """Register a standalone tool (outside of a plugin)."""
        self._tools[tool.name] = tool

    def register_verification_step(self, step: VerificationStep) -> None:
        """Register a standalone verification step."""
        self._verification_steps.append(step)

    def register_retrieval_strategy(self, strategy: RetrievalStrategy) -> None:
        """Register a standalone retrieval strategy."""
        self._retrieval_strategies[strategy.strategy_name] = strategy

    def register_model_provider(self, provider: ModelProvider) -> None:
        """Register a standalone model provider."""
        self._model_providers[provider.provider_name] = provider

    # ----- discovery -----

    def discover(self, plugins_dir: str | Path) -> int:
        """Auto-discover plugins from a directory.

        Each subdirectory is expected to be a Python package containing
        a ``plugin.py`` module with a ``register(registry)`` function.

        Returns the number of plugins loaded.
        """
        plugins_path = Path(plugins_dir)
        if not plugins_path.is_dir():
            return 0

        loaded = 0
        for child in sorted(plugins_path.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            plugin_module = child / "plugin.py"
            if not plugin_module.is_file():
                continue
            try:
                module_name = f"issue_resolver.plugins.{child.name}.plugin"
                mod = importlib.import_module(module_name)
                if hasattr(mod, "register"):
                    mod.register(self)
                    loaded += 1
                    print(f"[PluginRegistry] Loaded plugin '{child.name}'")
            except Exception as exc:
                print(f"[PluginRegistry] Failed to load plugin '{child.name}': {exc}")
        return loaded

    def load_manifest(self, manifest_path: str | Path) -> int:
        """Load plugins declared in a ``plugins.json`` manifest.

        Manifest format::

            {
                "plugins": [
                    {"name": "my_plugin", "module": "my_package.plugin"}
                ]
            }

        Returns the number of plugins loaded.
        """
        path = Path(manifest_path)
        if not path.is_file():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0

        loaded = 0
        for entry in data.get("plugins", []):
            module_path = entry.get("module", "")
            if not module_path:
                continue
            try:
                mod = importlib.import_module(module_path)
                if hasattr(mod, "register"):
                    mod.register(self)
                    loaded += 1
            except Exception as exc:
                print(f"[PluginRegistry] Manifest load failed for '{module_path}': {exc}")
        return loaded

    # ----- accessors -----

    def get_tool(self, name: str) -> Tool | None:
        """Return a tool by name, or ``None``."""
        return self._tools.get(name)

    def get_tools(self) -> dict[str, Tool]:
        """Return all registered tools."""
        return dict(self._tools)

    def get_verification_steps(self, language: str = "*") -> list[VerificationStep]:
        """Return verification steps applicable to *language*."""
        return [
            s for s in self._verification_steps
            if s.language in (language, "*")
        ]

    def get_retrieval_strategy(self, name: str) -> RetrievalStrategy | None:
        return self._retrieval_strategies.get(name)

    def get_retrieval_strategies(self) -> dict[str, RetrievalStrategy]:
        return dict(self._retrieval_strategies)

    def get_model_provider(self, name: str) -> ModelProvider | None:
        return self._model_providers.get(name)

    def get_plugins(self) -> dict[str, AgentPlugin]:
        return dict(self._plugins)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_GLOBAL_PLUGIN_REGISTRY = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Return the global plugin registry singleton."""
    return _GLOBAL_PLUGIN_REGISTRY
