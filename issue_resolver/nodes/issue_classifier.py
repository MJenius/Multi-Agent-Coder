"""Classifier node for categorising incoming GitHub issues.

Classifies issues into categories: Bug, Performance, Security,
Documentation, Testing, Feature, API Change, Dependency Update,
Refactor, Typing, Configuration.
"""

from __future__ import annotations

import re
from langchain_core.messages import HumanMessage, SystemMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.llm_utils import invoke_with_role_fallback
from issue_resolver.core.prompt_registry import get_prompt_registry

# Register prompt template on import
_DEFAULT_PROMPT = """\
You are an expert AI triage agent. Your job is to classify the user's issue description into exactly ONE of the following categories:

Categories:
- Bug (functional error, unexpected behavior, crash)
- Performance (slowness, memory leak, high CPU)
- Security (vulnerability, credentials exposure)
- Documentation (missing docs, typos in readme, comments)
- Testing (adding tests, broken test suite)
- Feature (request for new functionality)
- API Change (breaking interface changes, endpoint updates)
- Dependency Update (updating packages, changing lockfiles)
- Refactor (improving code structure, cleanups)
- Typing (adding type hints, fixing type checks)
- Configuration (updating setting files like .env or config files)

Output ONLY the category name from the list above. Do not output anything else.
"""

get_prompt_registry().register("issue_classifier", "1.0", _DEFAULT_PROMPT)


def issue_classifier_node(state: AgentState) -> dict:
    """Classify the incoming issue category."""
    print("[Classifier] Categorising issue...")
    issue = state.get("issue", "")

    prompt = get_prompt_registry().get("issue_classifier")

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Issue Description:\n{issue}"),
    ]

    response, model_name = invoke_with_role_fallback(
        role="issue_classifier",
        candidates=["meta/llama-3.3-70b-instruct"],
        messages=messages,
        temperature=0.2,
        max_tokens=128,
        context={"issue_length": len(issue)},
    )

    output = response.content.strip()

    # Match output with valid categories
    valid_categories = {
        "Bug", "Performance", "Security", "Documentation", "Testing",
        "Feature", "API Change", "Dependency Update", "Refactor", "Typing",
        "Configuration"
    }

    category = "Bug"  # Default
    for cat in valid_categories:
        if cat.lower() in output.lower():
            category = cat
            break

    print(f"[Classifier] Classified issue as: {category} (using {model_name})")

    # Record trace event if running
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "issue_classified",
            "Classifier",
            f"classified as {category}",
            details={"model": model_name, "raw_output": output},
        )

    return {
        "issue_category": category,
        "history": append_to_history(
            "Classifier",
            "Classify",
            f"Classified issue as {category}",
        ),
    }
