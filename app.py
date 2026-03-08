import streamlit as st
import os
import shutil
import git
import json
from issue_resolver.graph import app as agent_graph
from issue_resolver.utils.github_utils import fetch_issue_details, submit_pull_request

import stat

def _rmtree_readonly(func, path, excinfo):
    """
    Error handler for shutil.rmtree.
    On Windows, git files in .git/objects are often read-only, which causes
    PermissionError. This function changes the permissions and retries.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

st.set_page_config(page_title="Multi-Agent Issue Resolver", layout="wide")

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
</style>
""", unsafe_allow_html=True)

st.title("🤖 Multi-Agent Issue Resolver")
st.subheader("Automate GitHub Issue Fixes with AI Agents")

# Sidebar for Inputs
with st.sidebar:
    st.header("GitHub Credentials")
    github_token = st.text_input("GitHub PAT", type="password")
    repo_url = st.text_input("Repository URL (e.g., owner/repo)")
    issue_number = st.number_input("Issue Number", min_value=1, step=1)
    
    st.header("System Settings")
    sandbox_dir = os.path.abspath("sandbox_workspace")
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
                issue_content = f"Title: {title}\n\nBody: {body}"
                issue_content += "\n\nCRITICAL INSTRUCTION: The repository code is located strictly inside the './sandbox_workspace' directory. Do not search the root directory '.'"
            except Exception as e:
                st.error(f"Error fetching issue: {e}")
                st.stop()

        # 2. Clone Repository
        with st.spinner("Cloning repository..."):
            if os.path.exists(sandbox_dir):
                shutil.rmtree(sandbox_dir, onerror=_rmtree_readonly)
            
            os.makedirs(sandbox_dir, exist_ok=True)
            try:
                # Use GitPython to clone with LF line endings on Windows host
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

        # 3. Initialize State and Graph
        initial_state = {
            "issue": issue_content,
            "repo_path": "./sandbox_workspace",
            "file_context": [],
            "proposed_fix": "",
            "errors": "",
            "iterations": 0,
            "is_resolved": False,
            "history": []
        }

        # 4. Stream Execution
        trace_header = st.empty()
        trace_header.write("### 🧠 Agent Thought Trace")
        thought_container = st.empty()
        thought_log = ""
        
        final_state = initial_state
        
        for event in agent_graph.stream(initial_state):
            for node_name, state_update in event.items():
                new_logs = state_update.get("history", [])
                if new_logs:
                    for entry in new_logs:
                        thought_log += f"**[{node_name}]**: {entry}\n\n"
                        thought_container.markdown(f'<div class="thought-trace">{thought_log}</div>', unsafe_allow_html=True)
                
                final_state.update(state_update)
                
        # Remove the trace window when execution finishes
        trace_header.empty()
        thought_container.empty()

        # Store in session state for persistence
        # Compute is_resolved from evidence as fallback
        if final_state.get("proposed_fix") and not final_state.get("errors"):
            final_state["is_resolved"] = True
            
        # Debug: log final state keys for troubleshooting
        print(f"[DEBUG] Final state keys: {list(final_state.keys())}")
        print(f"[DEBUG] is_resolved={final_state.get('is_resolved')}, proposed_fix={bool(final_state.get('proposed_fix'))}, errors='{final_state.get('errors', '')}'")
        
        st.session_state.final_state = final_state
        st.session_state.thought_log = thought_log

# Display Results from Session State
if "final_state" in st.session_state:
    final_state = st.session_state.final_state
    thought_log = st.session_state.thought_log
    
    st.write("### 🧠 Execution History")
    st.markdown(f'<div class="thought-trace">{thought_log}</div>', unsafe_allow_html=True)
    
    st.write("### 🏁 Final Result")
    
    # Debug section - visible in UI
    with st.expander("🔍 Debug: Final State Inspection"):
        st.write(f"**is_resolved:** `{final_state.get('is_resolved')}`")
        st.write(f"**proposed_fix (exists):** `{bool(final_state.get('proposed_fix'))}`")
        st.write(f"**errors:** `{repr(final_state.get('errors', 'KEY_MISSING'))}`")
        st.write(f"**iterations:** `{final_state.get('iterations')}`")
        st.write(f"**All keys:** `{list(final_state.keys())}`")
    
    if final_state.get("is_resolved"):
        st.success("✅ Issue Resolved!")
        st.write("#### Proposed Fix:")
        st.code(final_state.get("proposed_fix"), language="diff")
        
        # 6. Pull Request Submission
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
