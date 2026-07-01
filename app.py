import streamlit as st
import os
import shutil
import git
import json
from issue_resolver.graph import app as agent_graph
from issue_resolver.utils.github_utils import fetch_issue_details, submit_pull_request
from issue_resolver.utils.issue_utils import extract_critical_sections
from issue_resolver.config import SANDBOX_WORKSPACE_DIR
from issue_resolver.core.execution_trace import start_trace, get_trace

import stat

def _rmtree_readonly(func, path, excinfo):
    """
    Error handler for shutil.rmtree.
    On Windows, git files in .git/objects are often read-only, which causes
    PermissionError. This function changes the permissions and retries.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

st.set_page_config(page_title="Multi-Agent Issue Resolver v2", layout="wide")

# Design Customization
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        color: #ffffff;
        background-color: #262730;
    }
    .thought-trace {
        background-color: #1e1e26;
        padding: 15px;
        border-radius: 10px;
        height: 400px;
        overflow-y: scroll;
        border: 1px solid #3e3e4a;
        font-family: 'Courier New', Courier, monospace;
    }
    .metric-card {
        background-color: #1a1c23;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4e88ff;
        margin-bottom: 10px;
    }
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .badge-high { background-color: #1e4620; color: #a2e8a2; }
    .badge-medium { background-color: #614002; color: #ffd685; }
    .badge-low { background-color: #5c1919; color: #ffb3b3; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Multi-Agent Autonomous Software Engineer v2")
st.subheader("Repository-Aware Multi-Agent System using LangGraph")

# Sidebar for Inputs
with st.sidebar:
    st.header("GitHub Credentials")
    github_token = st.text_input("GitHub PAT", type="password")
    repo_url = st.text_input("Repository URL (e.g., owner/repo)")
    issue_number = st.number_input("Issue Number", min_value=1, step=1)
    
    st.header("System Settings")
    sandbox_dir = SANDBOX_WORKSPACE_DIR
    if st.button("Clear Sandbox"):
        if os.path.exists(sandbox_dir):
            shutil.rmtree(sandbox_dir, onerror=_rmtree_readonly)
            st.success("Sandbox cleared!")

# Main Execution Flow
if st.button("🚀 Start Resolution Process"):
    if not (github_token and repo_url and issue_number):
        st.error("Please provide all required inputs.")
    else:
        # 1. Fetch Issue Details
        with st.spinner("Fetching issue details..."):
            try:
                title, body = fetch_issue_details(repo_url, int(issue_number), github_token)
                st.success(f"Fetched Issue: {title}")
                body_text = body if body else ""
                issue_content = f"Title: {title}\n\nBody: {body_text}"
                
                # Apply smart context management
                issue_content = extract_critical_sections(issue_content, max_length=4000)
                issue_content += "\n\nCRITICAL INSTRUCTION: The repository code is located strictly inside the './sandbox_workspace' directory. Do not search the root directory '.'"
            except Exception as e:
                st.error(f"Error fetching issue: {e}")
                if "404" in str(e):
                    st.info("💡 **Troubleshooting Tips:**\n"
                            "- Ensure your **GitHub Personal Access Token** is correct and has the necessary scopes (`repo` permissions).\n"
                            "- Ensure the **Repository Path** is in the format `owner/repo` (e.g. `psf/requests`).\n"
                            "- Verify that the **Issue Number** actually exists in that repository.")
                st.stop()

        # 2. Clone Repository
        with st.spinner("Cloning repository..."):
            if os.path.exists(sandbox_dir):
                shutil.rmtree(sandbox_dir, onerror=_rmtree_readonly)
            
            os.makedirs(sandbox_dir, exist_ok=True)
            try:
                repo_clone_url = f"https://github.com/{repo_url}.git"
                git.Repo.clone_from(
                    repo_clone_url, 
                    sandbox_dir,
                    multi_options=["-c core.autocrlf=false"],
                    allow_unsafe_options=True
                )
                st.success(f"Clone successful to {sandbox_dir}")
            except Exception as e:
                st.error(f"Error cloning repository: {e}")
                st.stop()

        # 3. Initialize State
        initial_state = {
            "issue": issue_content,
            "repo_path": sandbox_dir,
            "file_context": [],
            "plan": "",
            "proposed_fix": "",
            "errors": "",
            "validation_status": "",
            "next_step": "",
            "iterations": 0,
            "is_resolved": False,
            "environment_config": {},
            "contribution_guidelines": "",
            "history": [],
            
            # v2 State Fields
            "issue_category": "",
            "repo_intelligence": {},
            "repo_profile": {},
            "context_confidence": {},
            "structured_plan": {},
            "candidate_patches": [],
            "candidate_scores": [],
            "critique_results": [],
            "verification_report": {},
            "execution_trace": [],
            "coder_retry_budget": 3,
            "ast_validation_passed": True,
            "ast_error_detail": "",
            "test_error_context": "",
            "error_line_numbers": "",
            "test_code": "",
            "test_file_path": "",
        }

        # Initialize global trace session
        trace = start_trace(run_id=f"gui-{issue_number}")

        # 4. Stream Execution
        if "stop_requested" not in st.session_state:
            st.session_state.stop_requested = False

        trace_header = st.empty()
        trace_header.write("### 🧠 Agent Thought Trace")

        def _request_stop():
            st.session_state.stop_requested = True

        stop_container = st.empty()
        stop_btn = stop_container.button("🛑 STOP Execution", key="stop_btn", on_click=_request_stop)
        thought_container = st.empty()
        st.session_state.thought_container = thought_container
        st.session_state.thought_log = ""
        
        final_state = initial_state
        
        for event in agent_graph.stream(initial_state):
            if st.session_state.get("stop_requested"):
                st.session_state.stop_requested = False
                st.warning("⚠️ Execution stopped by user.")
                break
            
            for node_name, state_update in event.items():
                final_state.update(state_update)

        # Finalize trace info
        trace_summary = trace.finalize()
        final_state["execution_trace"] = trace_summary.get("events", [])

        # Clean UI elements
        trace_header.empty()
        stop_container.empty()
        thought_container.empty()

        vs = final_state.get("validation_status", "")
        if final_state.get("proposed_fix") and not final_state.get("errors"):
            if vs == "passed":
                final_state["is_resolved"] = True
            elif vs == "inconclusive":
                final_state["is_resolved"] = False
                final_state["resolution_note"] = "No test suite was found"
            
        st.session_state.final_state = final_state
        st.session_state.thought_log = st.session_state.get("thought_log", "")
        st.session_state.trace_summary = trace_summary

# Display Results from Session State
if "final_state" in st.session_state:
    final_state = st.session_state.final_state
    thought_log = st.session_state.thought_log
    trace_summary = st.session_state.get("trace_summary", {})
    
    st.write("### 🧠 Execution History")
    st.markdown(f'<div class="thought-trace">{thought_log}</div>', unsafe_allow_html=True)
    
    # Premium v2 Dashboard Layout
    st.write("### 📊 Subsystems Metadata Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        category = final_state.get("issue_category", "Bug")
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏷️ Issue Category</h4>
            <h2>{category}</h2>
            <p>Classified using LLM Triage Node</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        profile = final_state.get("repo_profile", {})
        language = profile.get("primary_language", "Python")
        framework = profile.get("framework", "FastAPI")
        st.markdown(f"""
        <div class="metric-card">
            <h4>Architecture & Style</h4>
            <h3>{language} | {framework}</h3>
            <p>Conventions: <code>{profile.get('naming_style', 'snake_case')}</code></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        summary_info = trace_summary.get("summary", {})
        total_time = summary_info.get("total_duration_ms", 0) / 1000
        total_toks = summary_info.get("total_tokens", 0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>⏱️ Execution Trace</h4>
            <h3>{total_time:.1f}s | {total_toks} tokens</h3>
            <p>Estimated Cost: <code>${total_toks * 0.000002:.4f}</code></p>
        </div>
        """, unsafe_allow_html=True)

    # 1. Knowledge Graph Summary
    intel = final_state.get("repo_intelligence", {})
    if intel:
        with st.expander("🕸️ Repository Knowledge Graph Detail"):
            st.write(f"**Total Modules:** {intel.get('total_modules', 0)}")
            st.write(f"**Total Classes:** {intel.get('total_classes', 0)}")
            st.write(f"**Total Functions:** {intel.get('total_functions', 0)}")
            st.write(f"**Detected Entrypoints:** `{', '.join(intel.get('entrypoints', [])) or 'None'}`")
            st.write(f"**Detected Config Files:** `{', '.join(intel.get('config_files', [])) or 'None'}`")

    # 2. Context Confidence
    confidences = final_state.get("context_confidence", {})
    if confidences:
        with st.expander("🔍 Hybrid Retrieval Context Confidence Scores"):
            st.write("Retrieved modules and their composite confidence signals:")
            for path, level in confidences.items():
                badge_class = f"badge-{level}"
                st.markdown(f"- <code>{path}</code> <span class='badge {badge_class}'>{level.upper()}</span>", unsafe_allow_html=True)

    # 3. Structured Plan
    plan_dict = final_state.get("structured_plan", {})
    if plan_dict:
        with st.expander("📅 Structured implementation Plan"):
            st.write(f"**Estimated Blast Radius:** `{plan_dict.get('estimated_blast_radius')}`")
            st.write(f"**Plan Confidence Score:** `{plan_dict.get('confidence_score')}`")
            st.write(f"**Risk Estimate:** `{plan_dict.get('risk_estimate')}`")
            st.write("**Implementation Steps:**")
            for step in plan_dict.get("implementation_steps", []):
                st.write(f"- `{step.get('step')}` **{step.get('file')}**: {step.get('action')}")

    # 4. Specialist Review Reports
    critique = final_state.get("critique_results", [])
    if critique and isinstance(critique[0], dict):
        with st.expander("🛡️ Specialist Audits (Security, Performance, API Compatibility)"):
            audits = critique[0]
            for audit_name, details in audits.items():
                status = "PASSED" if details.get("passed") else "FAILED"
                color = "green" if details.get("passed") else "red"
                st.markdown(f"#### **{audit_name.replace('_', ' ').title()} Review**: :{color}[{status}]")
                for finding in details.get("findings", []):
                    st.markdown(f"- {finding}")

    # Display final resolution details
    st.write("### 🏁 Final Result")
    
    if final_state.get("is_resolved"):
        st.success("✅ Issue Resolved!")
        st.write("#### Proposed Fix:")
        st.code(final_state.get("proposed_fix"), language="diff")
        
        if st.button("🚀 Submit Pull Request"):
            with st.spinner("Submitting Pull Request..."):
                try:
                    pr_url = submit_pull_request(
                        repo_path=sandbox_dir,
                        repo_full_name=repo_url,
                        issue_number=int(issue_number),
                        token=github_token,
                        proposed_fix=final_state.get("proposed_fix")
                    )
                    st.balloons()
                    st.success(f"PR Submitted Successfully! [View PR here]({pr_url})")
                except Exception as e:
                    st.error(f"Error submitting PR: {e}")
    else:
        st.error("❌ Failed to resolve the issue within the iteration limit.")
        if final_state.get("proposed_fix"):
            st.write("Last proposed fix (failed tests):")
            st.code(final_state.get("proposed_fix"), language="diff")
