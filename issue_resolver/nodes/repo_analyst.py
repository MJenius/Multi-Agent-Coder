"""Repo Analyst Node — analyzes repository frameworks, tools, and testing conventions.

Runs deep deterministic analysis of conventions, architectural style,
linter/formatter, and complexity metrics.  Results are stored in
``runtime_context`` so all downstream agents can access the profile.

No LLM call is needed — the static ``RepoAnalyzer`` already produces
a comprehensive ``RepoProfile`` covering language, framework,
architecture, naming, tooling, testing, CI, and complexity.
"""

from __future__ import annotations

import time

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.intelligence.repo_analyzer import RepoAnalyzer
import issue_resolver.runtime_context as runtime_context


def repo_analyst_node(state: AgentState) -> dict:
    """Analyze repository conventions and profile (deterministic, no LLM)."""
    print("[RepoAnalyst] Analyzing repository framework and conventions...")
    repo_path = state.get("repo_path", ".")

    graph = runtime_context.get_knowledge_graph()
    if graph is None:
        from issue_resolver.intelligence.graph_builder import GraphBuilder
        graph = GraphBuilder(repo_path).build()
        runtime_context.set_knowledge_graph(graph)

    t0 = time.monotonic()

    # Perform fully deterministic static analysis
    analyzer = RepoAnalyzer(repo_path, graph)
    profile = analyzer.analyse()

    # Store profile centrally for all downstream nodes
    runtime_context.set_repo_profile(profile)

    duration_ms = (time.monotonic() - t0) * 1000

    profile_dict = profile.to_dict()

    print(
        f"[RepoAnalyst] Static analysis complete in {duration_ms:.1f}ms: "
        f"{profile.primary_language or 'unknown'} / "
        f"{profile.framework or 'no framework'} / "
        f"{profile.test_framework or 'no test runner'}"
    )

    # Record trace event if running
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "repo_analysed",
            "RepoAnalyst",
            f"Analysed repository style and conventions (deterministic)",
            duration_ms=duration_ms,
            details={
                "language": profile.primary_language,
                "framework": profile.framework,
                "formatter": profile.formatter,
                "test_framework": profile.test_framework,
            },
        )

    return {
        "repo_profile": profile_dict,
        "history": append_to_history(
            "RepoAnalyst",
            "Analyse",
            f"Detected {profile.primary_language or 'unknown'} language, "
            f"{profile.framework or 'no'} framework, "
            f"{profile.test_framework or 'no'} test runner. (deterministic, no LLM)",
        ),
    }
