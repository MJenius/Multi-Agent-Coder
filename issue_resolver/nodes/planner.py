"""Planner Agent Node — generates structured implementation plans.

Produces structured JSON plans including dependency graphs, affected modules,
confidence scores, test strategy, and rollback plans.
"""

from __future__ import annotations

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.config import PLANNER_MODEL_CANDIDATES, PLANNER_MAX_ITERATIONS
from issue_resolver.llm_utils import invoke_with_role_fallback, calculate_max_tokens
from issue_resolver.core.prompt_registry import get_prompt_registry
import issue_resolver.runtime_context as runtime_context

# Register prompt template on import
_SYSTEM_PROMPT = r"""You are the Planner. Given a GitHub issue and repository context, generate a structured JSON стратегический план.

You must output a single JSON block wrapped in a ```json markdown block. Do not output any explanation outside of the code block.

JSON Structure:
{
  "files_to_edit": ["string"],
  "symbols_affected": ["string"],
  "implementation_steps": [
    {
      "step": 1,
      "file": "string",
      "action": "string",
      "depends_on": []
    }
  ],
  "dependency_ordering": [1],
  "execution_order": [1],
  "affected_modules": ["string"],
  "estimated_blast_radius": "low/medium/high",
  "confidence_score": 0.0,
  "assumptions": ["string"],
  "unresolved_questions": ["string"],
  "test_strategy": "string",
  "risk_estimate": "low/medium/high",
  "rollback_plan": "string"
}

Critical Rules:
1. "dependency_ordering" and "execution_order" should specify step numbers in sequence.
2. Under "test_strategy", specify how tests will verify the change (e.g. pytest commands, unittest assertions).
3. "confidence_score" should be between 0.0 and 1.0 based on available context.
4. "implementation_steps" must list incremental modifications with explicit dependencies.
5. If the issue is about parsing, serialization, or type formatting, prioritize modifications to general utilities/encoders over specific wrappers or endpoints.
6. Keep the scope of edits minimal and precise. Avoid massive repository-wide code edits.
7. Drive your decisions (e.g. files to edit, affected modules, execution order, dependency mapping) primarily using the Repository Intelligence section. The Repository Graph structure and relationship links are the authoritative source of repository structure. Align your step dependencies with the import/dependency relationships shown in the graph.
8. Restrict proposed edits ("files_to_edit") strictly to the files identified in the Localization Results. Do not propose editing files that have not been localized.
9. For each step in the implementation plan, explicitly specify the reason why that file was selected for modification.
"""

get_prompt_registry().register("planner", "2.0", _SYSTEM_PROMPT)


def planner_node(state: AgentState) -> dict:
    """Generate structured fix strategy."""
    print("[Planner] Generating structured fix strategy...")

    issue_text = state.get("issue", "(no issue)")
    file_context = state.get("file_context", [])
    symbol_map = state.get("symbol_map", "")
    iterations = state.get("iterations", 0)
    plan_iteration = state.get("plan_iteration", 0)

    if plan_iteration >= PLANNER_MAX_ITERATIONS:
        print(f"[Planner] Plan refinement limit ({PLANNER_MAX_ITERATIONS}) reached")
        return {
            "plan": state.get("plan", "(plan not generated)"),
            "plan_iteration": plan_iteration + 1,
            "history": append_to_history(
                "Planner",
                "Refinement Limit",
                f"Plan refinement reached {PLANNER_MAX_ITERATIONS} iterations.",
            ),
        }

    # Retrieve profile and intelligence summaries
    repo_profile = state.get("repo_profile", {})
    graph = runtime_context.get_knowledge_graph()

    repo_intel_parts = []

    if repo_profile:
        repo_intel_parts.append(
            f"### Naming/Architecture Conventions:\n"
            f"- Language: {repo_profile.get('primary_language')}\n"
            f"- Framework: {repo_profile.get('framework')}\n"
            f"- Architecture Pattern: {repo_profile.get('architecture_pattern')}\n"
            f"- Package Manager: {repo_profile.get('package_manager')}\n"
            f"- Linter: {repo_profile.get('linter')}\n"
            f"- Formatter: {repo_profile.get('formatter')}\n"
            f"- Testing: {repo_profile.get('test_framework')} (Directory: {repo_profile.get('test_directory')})"
        )

    if graph:
        # Get a nice textual summary of the repository graph (limit to 150 lines)
        graph_summary = graph.to_text_summary(limit=150)
        repo_intel_parts.append(f"### Repository Graph Structure:\n{graph_summary}")

    context_confidence = state.get("context_confidence", {})
    if context_confidence:
        confidence_lines = [f"- `{path}`: {conf} confidence" for path, conf in context_confidence.items()]
        repo_intel_parts.append("### Context File Confidence Scores:\n" + "\n".join(confidence_lines))

    # Add structured localization results to intelligence context
    localization_result = state.get("localization_result", {})
    if localization_result:
        loc_lines = [
            "### Localization Results:",
            f"- Overall Confidence: {localization_result.get('confidence', 0.0):.2f}",
        ]
        
        primary_files = localization_result.get("primary_files", [])
        if primary_files:
            loc_lines.append("  - Primary Files:")
            for f in primary_files[:5]:
                loc_lines.append(f"    - `{f['path']}` (Score: {f['score']:.2f}, Confidence: {f['confidence']})")
                
        symbols = localization_result.get("symbols", [])
        if symbols:
            loc_lines.append("  - Symbols Found:")
            for s in symbols[:10]:
                loc_lines.append(f"    - `{s['name']}` ({s['kind']}) in `{s['file_path']}` lines {s['line_number']}-{s['end_line']} (Source: {s['source']})")
                
        references = localization_result.get("references", [])
        if references:
            loc_lines.append("  - References / Call Graph:")
            for r in references[:5]:
                loc_lines.append(f"    - {r}")
                
        test_files = localization_result.get("test_files", [])
        if test_files:
            loc_lines.append(f"  - Related Test Files: {', '.join(test_files[:5])}")
            
        deps = localization_result.get("dependency_neighbors", [])
        if deps:
            loc_lines.append(f"  - Import Dependency Neighbors: {', '.join(deps[:5])}")
            
        repo_intel_parts.append("\n".join(loc_lines))

    # Format planning prompt
    context_parts = []
    if repo_intel_parts:
        context_parts.append("## Repository Intelligence\n" + "\n\n".join(repo_intel_parts))

    if file_context:
        context_parts.append("## Code Context\n" + "\n\n".join(file_context[:6]))

    context_str = "\n\n".join(context_parts) if context_parts else "(no context available)"

    prompt_content = f"""Issue Description:
{issue_text}

## Repository Context
{context_str}

Generate a structured plan in JSON format.
"""

    messages = [
        SystemMessage(content=get_prompt_registry().get("planner")),
        HumanMessage(content=prompt_content),
    ]

    try:
        resp, chosen_model = invoke_with_role_fallback(
            role="planner",
            candidates=PLANNER_MODEL_CANDIDATES,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
            context={"issue_category": state.get("issue_category", "Bug")},
        )
        print(f"[Planner] Using model: {chosen_model}")
    except Exception as exc:
        print(f"[Planner] [ERROR] LLM failed: {exc}")
        error_msg = f"Planner failed to generate strategy: {exc}"
        return {
            "errors": error_msg,
            "iterations": iterations + 1,
            "history": append_to_history("Planner", "Error", error_msg),
        }

    raw = getattr(resp, "content", "") or ""
    if not raw:
        print("[Planner] [ERROR] Empty LLM response")
        return {
            "errors": "Planner returned empty response",
            "iterations": iterations + 1,
            "history": append_to_history("Planner", "Error", "Empty response"),
        }

    # Extract JSON block
    json_block = ""
    m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        json_block = m.group(1).strip()
    else:
        # Try finding raw JSON structure
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
        if m:
            json_block = m.group(1).strip()

    if not json_block:
        # Fallback to saving raw response if JSON formatting fails
        print("[Planner] [WARNING] Could not parse markdown block. Using raw string.")
        json_block = raw.strip()

    # Verify JSON structure
    plan_dict = {}
    try:
        plan_dict = json.loads(json_block)
        print(f"[Planner] Plan successfully parsed as JSON with {len(plan_dict.get('implementation_steps', []))} steps")
    except json.JSONDecodeError:
        print("[Planner] [WARNING] Plan was not valid JSON. Storing as raw text.")
        plan_dict = {
            "raw_plan": json_block,
            "files_to_edit": [],
            "implementation_steps": [],
        }

    # Diagnostics logging (Requirement 6)
    print("[Planner] Diagnostics: Plan targets and choices:")
    for f in plan_dict.get("files_to_edit", []):
        print(f"  - Selected file to edit: `{f}`")
    print(f"  - Selected verification/test strategy: {plan_dict.get('test_strategy', 'none')}")
    print(f"  - Assigned plan confidence: {plan_dict.get('confidence_score', 0.0)}")

    # Log plan execution trace
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "plan_generated",
            "Planner",
            "Generated implementation strategy",
            details={
                "model": chosen_model,
                "files_to_edit": plan_dict.get("files_to_edit", []),
                "confidence": plan_dict.get("confidence_score", 0.0),
                "blast_radius": plan_dict.get("estimated_blast_radius", "unknown"),
            },
        )

    return {
        "plan": json.dumps(plan_dict, indent=2),
        "structured_plan": plan_dict,
        "plan_iteration": plan_iteration + 1,
        "iterations": iterations + 1,
        "history": append_to_history(
            "Planner",
            "Strategy Generated",
            f"Proposed changes to: {', '.join(plan_dict.get('files_to_edit', [])) or 'none'}",
        ),
    }
