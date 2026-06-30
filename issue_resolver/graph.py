from __future__ import annotations
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

from issue_resolver.utils.logger import append_to_history

from issue_resolver.state import AgentState
from issue_resolver.state import AgentState
from issue_resolver.nodes import (
    setup_node,
    researcher_node,
    localizer_node,
    planner_node,
    testgen_node,
    test_validator_node,
    coder_node,
    reviewer_node,
    failure_handler_node,
    issue_classifier_node,
    verification_type_classifier_node,
    repo_intelligence_node,
    repo_analyst_node,
    context_curator_node,
    candidate_generator_node,
    candidate_evaluator_node,
    incremental_patcher_node,
    parallel_reviewers_node,
    self_critique_node,
    debugger_node,
)

# ---------------------------------------------------------------------------
# Subgraph States
# ---------------------------------------------------------------------------

class WorkspaceDiscoveryState(TypedDict):
    issue: str
    repo_path: str
    issue_category: str
    verification_type: str
    repo_intelligence: dict
    repo_profile: dict
    file_context: list[str]
    context_confidence: dict
    symbol_map: str
    plan: str
    plan_iteration: int
    iterations: int
    history: Annotated[list[dict], operator.add]
    environment_config: dict
    current_view_file: str
    current_view_line: int
    
    # New discovery state
    localization_result: dict
    localization_confidence: float
    issue_suitability: dict
    classification_method: str

class PatchEngineeringState(TypedDict):
    issue: str
    repo_path: str
    plan: str
    structured_plan: dict
    file_context: list[str]
    coder_retry_budget: int
    candidate_patches: list[dict]
    candidate_scores: list[dict]
    proposed_fix: str
    errors: str
    iterations: int
    history: Annotated[list[dict], operator.add]
    ast_validation_passed: bool
    ast_error_detail: str
    test_error_context: str
    error_line_numbers: str
    validation_status: str
    test_code: str
    test_file_path: str
    environment_config: dict

class VerificationState(TypedDict):
    issue: str
    repo_path: str
    plan: str
    structured_plan: dict
    file_context: list[str]
    proposed_fix: str
    test_code: str
    test_file_path: str
    test_framework_used: str
    test_runs_initially: bool
    verification_type: str
    errors: str
    iterations: int
    critique_results: list[dict]
    history: Annotated[list[dict], operator.add]
    environment_config: dict

# ---------------------------------------------------------------------------
# Subgraph Routing Decisions
# ---------------------------------------------------------------------------

def _route_patch_reviewer(state: PatchEngineeringState) -> str:
    val_status = state.get("validation_status", "")
    budget = state.get("coder_retry_budget", 0)
    if val_status in ("failed", "applied_with_errors") and budget > 0:
        return "debugger"
    return "end"

def _route_researcher(state: WorkspaceDiscoveryState) -> str:
    loc_res = state.get("localization_result", {})
    if loc_res.get("needs_researcher_fallback", True):
        print("[Graph] Localizer confidence low, routing to Researcher fallback.")
        return "researcher"
    print("[Graph] Localizer confidence sufficient, skipping Researcher fallback.")
    return "context_curator"

# ---------------------------------------------------------------------------
# Subgraphs Construction
# ---------------------------------------------------------------------------

# 1. Workspace Discovery Subgraph
wd_builder = StateGraph(WorkspaceDiscoveryState)
wd_builder.add_node("setup", setup_node)
wd_builder.add_node("repo_intelligence_node", repo_intelligence_node)
wd_builder.add_node("repo_analyst", repo_analyst_node)
wd_builder.add_node("localizer", localizer_node)
wd_builder.add_node("issue_classifier", issue_classifier_node)
wd_builder.add_node("verification_type_classifier", verification_type_classifier_node)
wd_builder.add_node("researcher", researcher_node)
wd_builder.add_node("context_curator", context_curator_node)
wd_builder.add_node("planner", planner_node)

wd_builder.set_entry_point("setup")
wd_builder.add_edge("setup", "repo_intelligence_node")
wd_builder.add_edge("repo_intelligence_node", "repo_analyst")
wd_builder.add_edge("repo_analyst", "localizer")
wd_builder.add_edge("localizer", "issue_classifier")
wd_builder.add_edge("issue_classifier", "verification_type_classifier")
wd_builder.add_conditional_edges(
    "verification_type_classifier",
    _route_researcher,
    {
        "researcher": "researcher",
        "context_curator": "context_curator"
    }
)
wd_builder.add_edge("researcher", "context_curator")
wd_builder.add_edge("context_curator", "planner")
wd_builder.add_edge("planner", END)
wd_graph = wd_builder.compile()

# 2. Patch Engineering Subgraph
pe_builder = StateGraph(PatchEngineeringState)
pe_builder.add_node("candidate_generator", candidate_generator_node)
pe_builder.add_node("candidate_evaluator", candidate_evaluator_node)
pe_builder.add_node("incremental_patcher", incremental_patcher_node)
pe_builder.add_node("reviewer", reviewer_node)
pe_builder.add_node("debugger", debugger_node)

pe_builder.set_entry_point("candidate_generator")
pe_builder.add_edge("candidate_generator", "candidate_evaluator")
pe_builder.add_edge("candidate_evaluator", "incremental_patcher")
pe_builder.add_edge("incremental_patcher", "reviewer")
pe_builder.add_conditional_edges(
    "reviewer",
    _route_patch_reviewer,
    {
        "debugger": "debugger",
        "end": END
    }
)
pe_builder.add_edge("debugger", "candidate_generator")
pe_graph = pe_builder.compile()

# 3. Verification Subgraph
v_builder = StateGraph(VerificationState)
v_builder.add_node("test_generator", testgen_node)
v_builder.add_node("test_validator", test_validator_node)
v_builder.add_node("parallel_reviewers", parallel_reviewers_node)
v_builder.add_node("self_critique", self_critique_node)

v_builder.set_entry_point("test_generator")
v_builder.add_edge("test_generator", "test_validator")
v_builder.add_edge("test_validator", "parallel_reviewers")
v_builder.add_edge("parallel_reviewers", "self_critique")
v_builder.add_edge("self_critique", END)
v_graph = v_builder.compile()

# ---------------------------------------------------------------------------
# Node Wrappers for Main Graph
# ---------------------------------------------------------------------------

def workspace_discovery_node(state: AgentState) -> dict:
    sub_state = {
        "issue": state.get("issue"),
        "repo_path": state.get("repo_path"),
        "issue_category": state.get("issue_category", "Bug"),
        "verification_type": state.get("verification_type", "runtime tests"),
        "repo_intelligence": state.get("repo_intelligence", {}),
        "repo_profile": state.get("repo_profile", {}),
        "file_context": list(state.get("file_context", [])),
        "context_confidence": state.get("context_confidence", {}),
        "symbol_map": state.get("symbol_map", ""),
        "plan": state.get("plan", ""),
        "plan_iteration": state.get("plan_iteration", 0),
        "iterations": state.get("iterations", 0),
        "history": [],
        "environment_config": state.get("environment_config", {}),
        "current_view_file": state.get("current_view_file"),
        "current_view_line": state.get("current_view_line", 1),
        "localization_result": state.get("localization_result", {}),
        "localization_confidence": state.get("localization_confidence", 0.0),
        "issue_suitability": state.get("issue_suitability", {}),
        "classification_method": state.get("classification_method", ""),
    }
    res = wd_graph.invoke(sub_state)
    completion_entry = append_to_history("WorkspaceDiscovery", "Complete", "Discovery and intelligence gathering complete. Plan generated.")[0]
    return {
        "plan": res.get("plan"),
        "issue_category": res.get("issue_category"),
        "verification_type": res.get("verification_type"),
        "repo_intelligence": res.get("repo_intelligence"),
        "repo_profile": res.get("repo_profile"),
        "file_context": res.get("file_context"),
        "context_confidence": res.get("context_confidence"),
        "symbol_map": res.get("symbol_map"),
        "environment_config": res.get("environment_config"),
        "iterations": res.get("iterations"),
        "current_view_file": res.get("current_view_file"),
        "current_view_line": res.get("current_view_line"),
        "localization_result": res.get("localization_result"),
        "localization_confidence": res.get("localization_confidence"),
        "issue_suitability": res.get("issue_suitability"),
        "classification_method": res.get("classification_method"),
        "history": res.get("history", []) + [completion_entry],
    }


def verification_node(state: AgentState) -> dict:
    sub_state = {
        "issue": state.get("issue"),
        "repo_path": state.get("repo_path"),
        "plan": state.get("plan"),
        "structured_plan": state.get("structured_plan", {}),
        "file_context": list(state.get("file_context", [])),
        "proposed_fix": state.get("proposed_fix", ""),
        "test_code": state.get("test_code", ""),
        "test_file_path": state.get("test_file_path", ""),
        "test_framework_used": state.get("test_framework_used", "pytest"),
        "test_runs_initially": state.get("test_runs_initially"),
        "verification_type": state.get("verification_type", "runtime tests"),
        "errors": state.get("errors", ""),
        "iterations": state.get("iterations", 0),
        "critique_results": state.get("critique_results", []),
        "history": [],
        "environment_config": state.get("environment_config", {}),
    }
    res = v_graph.invoke(sub_state)
    completion_entry = append_to_history("Verification", "Complete", f"Verification complete. Runs initially: {res.get('test_runs_initially')}.")[0]
    return {
        "test_code": res.get("test_code"),
        "test_file_path": res.get("test_file_path"),
        "test_framework_used": res.get("test_framework_used"),
        "test_runs_initially": res.get("test_runs_initially"),
        "critique_results": res.get("critique_results"),
        "errors": res.get("errors"),
        "iterations": res.get("iterations"),
        "history": res.get("history", []) + [completion_entry],
    }

def patch_engineering_node(state: AgentState) -> dict:
    # Decrement coder retry budget on invocation
    curr_budget = state.get("coder_retry_budget", 3)
    new_budget = max(0, curr_budget - 1)
    
    sub_state = {
        "issue": state.get("issue"),
        "repo_path": state.get("repo_path"),
        "plan": state.get("plan"),
        "structured_plan": state.get("structured_plan", {}),
        "file_context": list(state.get("file_context", [])),
        "coder_retry_budget": new_budget,
        "candidate_patches": state.get("candidate_patches", []),
        "candidate_scores": state.get("candidate_scores", []),
        "proposed_fix": state.get("proposed_fix", ""),
        "errors": state.get("errors", ""),
        "iterations": state.get("iterations", 0),
        "history": [],
        "ast_validation_passed": state.get("ast_validation_passed", True),
        "ast_error_detail": state.get("ast_error_detail", ""),
        "test_error_context": state.get("test_error_context", ""),
        "error_line_numbers": state.get("error_line_numbers", ""),
        "validation_status": state.get("validation_status", ""),
        "test_code": state.get("test_code", ""),
        "test_file_path": state.get("test_file_path", ""),
        "environment_config": state.get("environment_config", {}),
    }
    res = pe_graph.invoke(sub_state)
    completion_entry = append_to_history("PatchEngineering", "Complete", f"Patch engineering complete. Status: {res.get('validation_status')}.")[0]
    return {
        "proposed_fix": res.get("proposed_fix"),
        "validation_status": res.get("validation_status"),
        "errors": res.get("errors"),
        "ast_validation_passed": res.get("ast_validation_passed"),
        "ast_error_detail": res.get("ast_error_detail"),
        "coder_retry_budget": res.get("coder_retry_budget"),
        "iterations": res.get("iterations"),
        "is_resolved": res.get("validation_status") == "passed",
        "history": res.get("history", []) + [completion_entry],
    }

def _route_parent(state: AgentState) -> str:
    status = state.get("validation_status", "")
    if status == "passed" or status == "inconclusive":
        return "end"
    return "failure_handler"

# ---------------------------------------------------------------------------
# Main Graph Build
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("workspace_discovery", workspace_discovery_node)
    graph.add_node("verification", verification_node)
    graph.add_node("patch_engineering", patch_engineering_node)
    graph.add_node("failure_handler", failure_handler_node)
    
    graph.set_entry_point("workspace_discovery")
    graph.add_edge("workspace_discovery", "verification")
    graph.add_edge("verification", "patch_engineering")
    graph.add_conditional_edges(
        "patch_engineering",
        _route_parent,
        {
            "end": END,
            "failure_handler": "failure_handler"
        }
    )
    graph.add_edge("failure_handler", END)
    compiled = graph.compile()
    print("[OK] LangGraph state machine compiled successfully!")
    return compiled

app = build_graph()
