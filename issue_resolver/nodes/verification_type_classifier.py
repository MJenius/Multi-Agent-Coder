"""Verification Type Classifier node.

Classifies incoming GitHub issues into one of the following verification types:
- runtime tests
- static type checking
- linting
- documentation validation
- performance benchmarking
- configuration verification
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
You are an expert AI triage agent. Your job is to classify the user's issue description into exactly ONE of the following verification types, which determines how a proposed fix should be validated:

Verification Types:
- runtime tests (functional code changes, bug fixes, features that require executing the code or unit tests)
- static type checking (type hints, type annotations, mypy or pyright type fixes)
- linting (code style, formatting, syntax validation, ruff or flake8 cleanups)
- documentation validation (changes to README, docstrings, markdown documentation, or comments)
- performance benchmarking (performance improvements, profiling, speedups, memory leak checks)
- configuration verification (changes to pyproject.toml, configuration files, CI/CD, or .env files)

Output ONLY the verification type name from the list above. Do not output anything else.
"""

get_prompt_registry().register("verification_type_classifier", "1.0", _DEFAULT_PROMPT)


def verification_type_classifier_node(state: AgentState) -> dict:
    """Classify the verification type needed for the issue."""
    print("[VerificationClassifier] Categorising verification type...")
    issue = state.get("issue", "")

    prompt = get_prompt_registry().get("verification_type_classifier")

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Issue Description:\n{issue}"),
    ]

    response, model_name = invoke_with_role_fallback(
        role="verification_type_classifier",
        candidates=["meta/llama-3.3-70b-instruct"],
        messages=messages,
        temperature=0.2,
        max_tokens=128,
        context={"issue_length": len(issue)},
    )

    output = response.content.strip()

    # Match output with valid verification types
    valid_types = {
        "runtime tests", "static type checking", "linting",
        "documentation validation", "performance benchmarking",
        "configuration verification"
    }

    verification_type = "runtime tests"  # Default
    for val_type in valid_types:
        if val_type.lower() in output.lower():
            verification_type = val_type
            break

    print(f"[VerificationClassifier] Classified verification type as: {verification_type} (using {model_name})")

    # Record trace event if running
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "verification_type_classified",
            "VerificationClassifier",
            f"classified as {verification_type}",
            details={"model": model_name, "raw_output": output},
        )

    return {
        "verification_type": verification_type,
        "history": append_to_history(
            "VerificationClassifier",
            "Classify",
            f"Classified verification type as {verification_type}",
        ),
    }
