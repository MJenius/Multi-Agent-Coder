"""
Sandbox tools -- Uses the Docker SDK to interact with the sandbox container.
"""

import os
import docker

def get_sandbox_container():
    """Finds the sandbox container using labels."""
    client = docker.from_env()
    # Use filters to find the container reliably
    containers = client.containers.list(filters={"label": "com.issue_resolver.role=sandbox"})
    if not containers:
        return None
    return containers[0]

def apply_diff_in_sandbox(diff: str, repo_path: str) -> str:
    """
    Applies a code diff (patch) inside the Docker sandbox.
    Before applying the patch, it ensures the workspace is clean.
    """
    sandbox = get_sandbox_container()
    if not sandbox:
        return "Error: Sandbox container not found. Is it running?"
        
    # 1. Idempotency: Clean the workspace before doing anything
    # git restore . will discard changes in working directory
    # git clean -fd will remove untracked files and directories
    print("[Sandbox] Cleaning workspace...")
    cleanup_result = sandbox.exec_run("git clean -fd", workdir="/workspace")
    cleanup_result = sandbox.exec_run("git restore .", workdir="/workspace")
    
    # Check if git failed (e.g., if there's no git repo)
    if cleanup_result.exit_code != 0:
         print("[Sandbox] git restore failed (maybe not a git repo or no commits yet). Ignoring cleanup.")
         
    # 2. Write the patch file to the mapped workspace directory
    patch_path = os.path.join(repo_path, "sandbox_workspace", "fix.patch")
    
    # Ensure sandbox_workspace exists (though docker-compose usually creates it)
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    
    try:
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(diff)
    except Exception as e:
        return f"Error writing patch file locally: {e}"
        
    # 3. Apply the patch inside the container
    print("[Sandbox] Applying patch...")
    patch_result = sandbox.exec_run("patch -p1 < fix.patch", workdir="/workspace")
    
    output = patch_result.output.decode('utf-8', errors='ignore')
    
    if patch_result.exit_code != 0:
         return f"Error applying patch:\n{output}"
         
    return f"Patch applied successfully.\n{output}"

def run_main_in_sandbox() -> str | tuple[bool, str]:
    """
    Runs the main application entry point inside the sandbox.
    Returns (success_boolean, output_string)
    """
    sandbox = get_sandbox_container()
    if not sandbox:
         return False, "Error: Sandbox container not found."
         
    print("[Sandbox] Running src/main.py...")
    result = sandbox.exec_run("python src/main.py", workdir="/workspace")
    
    # Decode and strip special characters to prevent UnicodeEncodeError
    output = result.output.decode('utf-8', errors='ignore')
    
    success = (result.exit_code == 0)
    return success, output
