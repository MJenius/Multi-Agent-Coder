"""Repo Analyst Node — analyzes repository frameworks, tools, and testing conventions.

Runs deep analysis of conventions, architectural style, linter/formatter,
and complexity metrics to provide structured constraints to all agents.
"""

from __future__ import annotations

import time
from langchain_core.messages import HumanMessage, SystemMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.intelligence.repo_analyzer import RepoAnalyzer
from issue_resolver.llm_utils import invoke_with_role_fallback
import issue_resolver.runtime_context as runtime_context
from issue_resolver.core.prompt_registry import get_prompt_registry

# Register prompt template on import
_DEFAULT_PROMPT = """\
You are an expert repository analyst. You are given a static analysis profile of a code repository and its knowledge graph.
Analyze the repository design patterns, architectural style, coding style, framework integration, testing conventions, and potential risk areas.
Summarize your findings in a structured, concise summary for developer planning.
"""

get_prompt_registry().register("repo_analyst", "1.0", _DEFAULT_PROMPT)


def repo_analyst_node(state: AgentState) -> dict:
    """Analyze repository conventions and profile."""
    print("[RepoAnalyst] Analyzing repository framework and conventions...")
    repo_path = state.get("repo_path", ".")

    graph = runtime_context.get_knowledge_graph()
    if graph is None:
        from issue_resolver.intelligence.graph_builder import GraphBuilder
        graph = GraphBuilder(repo_path).build()
        runtime_context.set_knowledge_graph(graph)

    # 1. Perform static analysis
    analyzer = RepoAnalyzer(repo_path, graph)
    profile = analyzer.analyse()

    t0 = time.monotonic()

    # 2. Refine analysis with LLM context if requested
    prompt = get_prompt_registry().get("repo_analyst")

    static_summary = profile.to_prompt_context()

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Static Analysis Profile:\n{static_summary}\n\nKnowledge Graph Summary:\n{graph.to_text_summary(limit=100) if hasattr(graph, 'to_text_summary') else ''}"),
    ]

    response, model_name = invoke_with_role_fallback(
        role="repo_analyst",
        candidates=["qwen/qwen3.5-122b-a10b", "meta/llama-3.3-70b-instruct"],
        messages=messages,
        temperature=0.4,
        max_tokens=2048,
        context={"repo_size": profile.total_files},
    )

    llm_synthesis = response.content.strip()
    duration_ms = (time.monotonic() - t0) * 1000

    profile_dict = profile.to_dict()
    profile_dict["llm_synthesis"] = llm_synthesis

    print(f"[RepoAnalyst] Deep analysis complete (using {model_name} in {duration_ms:.1f}ms)")

    # Record trace event if running
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "repo_analysed",
            "RepoAnalyst",
            f"Analysed repository style and conventions",
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
            f"Detected {profile.primary_language or 'unknown'} language, {profile.framework or 'no'} framework, {profile.test_framework or 'no'} test runner.",
        ),
    }
