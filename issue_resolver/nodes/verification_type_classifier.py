"""Verification Type Classifier node.

Classifies incoming GitHub issues into one of the following verification types
completely deterministically based on the issue category.
"""

from __future__ import annotations

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history


def verification_type_classifier_node(state: AgentState) -> dict:
    """Classify the verification type needed for the issue (deterministic)."""
    category = state.get("issue_category", "Bug")
    print(f"[VerificationClassifier] Categorising verification type for category: {category}...")

    # Mapping category to verification type deterministically
    mapping = {
        "Bug": "runtime tests",
        "Feature": "runtime tests",
        "API Change": "runtime tests",
        "Security": "runtime tests",
        "Testing": "runtime tests",
        "Refactor": "runtime tests",         # Run tests to prevent regressions
        "Dependency Update": "runtime tests", # Run tests to check dependency breaking changes
        "Typing": "static type checking",
        "Configuration": "configuration verification",
        "Documentation": "documentation validation",
        "Performance": "performance benchmarking",
    }

    verification_type = mapping.get(category, "runtime tests")

    print(f"[VerificationClassifier] Deterministically classified verification type as: {verification_type}")

    # Record trace event if running
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "verification_type_classified",
            "VerificationClassifier",
            f"classified as {verification_type} (deterministic)",
            details={"category": category, "verification_type": verification_type},
        )

    return {
        "verification_type": verification_type,
        "history": append_to_history(
            "VerificationClassifier",
            "Classify",
            f"Classified verification type as {verification_type} (deterministic from category {category})",
        ),
    }
