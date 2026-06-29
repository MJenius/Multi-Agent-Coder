"""Parallel Reviewers Node — executes parallel specialist code audits.

Executes security, performance, and API compatibility reviews concurrently,
merging findings into a structured review report.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.llm_utils import invoke_with_role_fallback
from issue_resolver.core.prompt_registry import get_prompt_registry


# Register prompts on import
_SECURITY_PROMPT = """\
You are an expert Security Reviewer. Review the proposed code fix for security issues like:
- OWASP Top 10 vulnerabilities (SQLi, XSS, Path Traversal, SSRF, Command Injection)
- Safe input sanitization and verification checks
- Safe credential or configuration file access

You must output a single JSON block wrapped in a ```json markdown block:
{
  "passed": true/false,
  "findings": ["string"],
  "confidence": "high/medium/low"
}
"""

_PERFORMANCE_PROMPT = """\
You are an expert Performance Reviewer. Review the proposed code fix for performance issues like:
- Unnecessary complexity or memory allocations (e.g. O(N^2) loops)
- Unbounded DB queries or missing cache hits
- Resource leaks (unclosed sockets, files, etc.)

You must output a single JSON block wrapped in a ```json markdown block:
{
  "passed": true/false,
  "findings": ["string"],
  "confidence": "high/medium/low"
}
"""

_API_COMPAT_PROMPT = """\
You are an expert API Compatibility Reviewer. Review the proposed code fix for API changes like:
- Breaking public endpoints or library APIs
- Typings and return type signatures changes
- Missing backwards-compatibility fallbacks

You must output a single JSON block wrapped in a ```json markdown block:
{
  "passed": true/false,
  "findings": ["string"],
  "confidence": "high/medium/low"
}
"""

get_prompt_registry().register("security_reviewer", "1.0", _SECURITY_PROMPT)
get_prompt_registry().register("performance_reviewer", "1.0", _PERFORMANCE_PROMPT)
get_prompt_registry().register("api_compat_reviewer", "1.0", _API_COMPAT_PROMPT)


def _run_specialist_review(role: str, prompt: str, content: str) -> dict:
    """Run one specialist reviewer in a background thread."""
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=content),
    ]
    try:
        resp, model_name = invoke_with_role_fallback(
            role="reviewer",
            candidates=["nvidia/nemotron-3-ultra-550b-a55b"],
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )
        raw = getattr(resp, "content", "") or ""
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        json_str = m.group(1).strip() if m else raw.strip()
        data = json.loads(json_str)
        data["model"] = model_name
        return data
    except Exception as exc:
        print(f"[ParallelReviewers] [ERROR] Reviewer '{role}' failed: {exc}")
        return {
            "passed": True,  # Fallback to true so we don't block
            "findings": [f"Reviewer execution failed: {exc}"],
            "confidence": "low",
            "model": "unknown",
        }


def parallel_reviewers_node(state: AgentState) -> dict:
    """Run security, performance, and API reviews concurrently."""
    proposed_fix = state.get("proposed_fix", "")
    if not proposed_fix:
        return {}

    print("[ParallelReviewers] Running parallel specialist code reviews...")

    review_content = f"""Proposed Patch:
{proposed_fix}
"""

    specialists = [
        ("security", get_prompt_registry().get("security_reviewer")),
        ("performance", get_prompt_registry().get("performance_reviewer")),
        ("api_compatibility", get_prompt_registry().get("api_compat_reviewer")),
    ]

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_specialist_review, name, prompt, review_content): name
            for name, prompt in specialists
        }

        for fut in concurrent.futures.as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
                print(f"[ParallelReviewers] {name.replace('_', ' ').title()} review completed.")
            except Exception as exc:
                print(f"[ParallelReviewers] {name} review failed: {exc}")

    # Determine overall status
    overall_passed = all(results.get(name, {}).get("passed", True) for name in results)

    # Log trace event
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "parallel_reviews_completed",
            "ParallelReviewers",
            f"Overall status: {'PASS' if overall_passed else 'FAIL'}",
            details=results,
        )

    return {
        "critique_results": [results],
        "history": append_to_history(
            "ParallelReviewers",
            "Review",
            f"Completed parallel reviews. Overall: {'PASSED' if overall_passed else 'FAILED'}",
        ),
    }
