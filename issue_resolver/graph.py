from __future__ import annotations
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

from issue_resolver.state import AgentState
from issue_resolver.nodes import (
    setup_node,
    researcher_node,
    planner_node,
    testgen_node,
    test_validator_node,
    coder_node,
    reviewer_node,
    failure_handler_node,
)

class WorkspaceDiscoveryState(TypedDict):
    issue: str
    repo_path: str
    file_context: list[str]
    symbol_map: str
    plan: str
    plan_iteration: int
    iterations: int
    history: Annotated[list[dict], operator.add]
    environment_config: dict
    current_view_file: str
    current_view_line: int

class PatchEngineeringState(TypedDict):
    issue: str
    repo_path: str
    plan: str
    file_context: list[str]
    coder_retry_budget: int
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
    file_context: list[str]
    test_code: str
    test_file_path: str
    test_framework_used: str
    test_runs_initially: bool
    errors: str
    iterations: int
    history: Annotated[list[dict], operator.add]
    environment_config: dict

def _route_patch_reviewer(state: PatchEngineeringState) -> str:
    ast_pass = state.get("ast_validation_passed", True)
    val_status = state.get("validation_status", "")
    budget = state.get("coder_retry_budget", 0)
    if (not ast_pass or val_status == "failed") and budget > 0:
        return "coder"
    return "end"

wd_builder = StateGraph(WorkspaceDiscoveryState)
wd_builder.add_node("setup", setup_node)
wd_builder.add_node("researcher", researcher_node)
wd_builder.add_node("planner", planner_node)
wd_builder.set_entry_point("setup")
wd_builder.add_edge("setup", "researcher")
wd_builder.add_edge("researcher", "planner")
wd_builder.add_edge("planner", END)
wd_graph = wd_builder.compile()

pe_builder = StateGraph(PatchEngineeringState)
pe_builder.add_node("coder", coder_node)
pe_builder.add_node("reviewer", reviewer_node)
pe_builder.set_entry_point("coder")
pe_builder.add_edge("coder", "reviewer")
pe_builder.add_conditional_edges(
    "reviewer",
    _route_patch_reviewer,
    {
        "coder": "coder",
        "end": END
    }
)
pe_graph = pe_builder.compile()

v_builder = StateGraph(VerificationState)
v_builder.add_node("test_generator", testgen_node)
v_builder.add_node("test_validator", test_validator_node)
v_builder.set_entry_point("test_generator")
v_builder.add_edge("test_generator", "test_validator")
v_builder.add_edge("test_validator", END)
v_graph = v_builder.compile()

def workspace_discovery_node(state: AgentState) -> dict:
    sub_state = {
        "issue": state.get("issue"),
        "repo_path": state.get("repo_path"),
        "file_context": list(state.get("file_context", [])),
        "symbol_map": state.get("symbol_map", ""),
        "plan": state.get("plan", ""),
        "plan_iteration": state.get("plan_iteration", 0),
        "iterations": state.get("iterations", 0),
        "history": [],
        "environment_config": state.get("environment_config", {}),
        "current_view_file": state.get("current_view_file"),
        "current_view_line": state.get("current_view_line", 1),
    }
    res = wd_graph.invoke(sub_state)
    return {
        "plan": res.get("plan"),
        "file_context": res.get("file_context"),
        "symbol_map": res.get("symbol_map"),
        "environment_config": res.get("environment_config"),
        "iterations": res.get("iterations"),
        "current_view_file": res.get("current_view_file"),
        "current_view_line": res.get("current_view_line"),
        "history": [{"node": "WorkspaceDiscovery", "action": "Complete", "content": "Discovery phase complete. Plan generated."}],
    }

def verification_node(state: AgentState) -> dict:
    sub_state = {
        "issue": state.get("issue"),
        "repo_path": state.get("repo_path"),
        "plan": state.get("plan"),
        "file_context": list(state.get("file_context", [])),
        "test_code": state.get("test_code", ""),
        "test_file_path": state.get("test_file_path", ""),
        "test_framework_used": state.get("test_framework_used", "pytest"),
        "test_runs_initially": state.get("test_runs_initially"),
        "errors": state.get("errors", ""),
        "iterations": state.get("iterations", 0),
        "history": [],
        "environment_config": state.get("environment_config", {}),
    }
    res = v_graph.invoke(sub_state)
    return {
        "test_code": res.get("test_code"),
        "test_file_path": res.get("test_file_path"),
        "test_framework_used": res.get("test_framework_used"),
        "test_runs_initially": res.get("test_runs_initially"),
        "errors": res.get("errors"),
        "iterations": res.get("iterations"),
        "history": [{"node": "Verification", "action": "Complete", "content": f"Verification complete. Runs initially: {res.get('test_runs_initially')}."}],
    }

def patch_engineering_node(state: AgentState) -> dict:
    sub_state = {
        "issue": state.get("issue"),
        "repo_path": state.get("repo_path"),
        "plan": state.get("plan"),
        "file_context": list(state.get("file_context", [])),
        "coder_retry_budget": state.get("coder_retry_budget", 3),
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
    return {
        "proposed_fix": res.get("proposed_fix"),
        "validation_status": res.get("validation_status"),
        "errors": res.get("errors"),
        "ast_validation_passed": res.get("ast_validation_passed"),
        "ast_error_detail": res.get("ast_error_detail"),
        "coder_retry_budget": res.get("coder_retry_budget"),
        "iterations": res.get("iterations"),
        "is_resolved": res.get("validation_status") == "passed",
        "history": [{"node": "PatchEngineering", "action": "Complete", "content": f"Patch engineering complete. Status: {res.get('validation_status')}."}],
    }

def _route_parent(state: AgentState) -> str:
    status = state.get("validation_status", "")
    if status == "passed" or status == "inconclusive":
        return "end"
    return "failure_handler"

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
    print("[OK] Graph compiled successfully!")
    return compiled

app = build_graph()
