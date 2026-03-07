"""
Sandbox tools -- Uses the Docker SDK to interact with the sandbox container.
"""

import os
import docker

def get_sandbox_container():
    """Finds the sandbox container using labels."""
    client = docker.from_env()
    containers = client.containers.list(filters={"label": "com.issue_resolver.role=sandbox"})
    if not containers:
        return None
    return containers[0]

def apply_diff_in_sandbox(diff: str, repo_path: str) -> str:
    """
    Applies a code diff (patch) inside the Docker sandbox.
    Uses bash -c for shell redirection and git for clean state management.
    """
    sandbox = get_sandbox_container()
    if not sandbox:
        return "Error: Sandbox container not found. Is it running?"
        
    # 1. Idempotency Baseline: Ensure files are committed so git restore works
    # This must be done inside the container workspace
    sandbox.exec_run("git init", workdir="/workspace")
    sandbox.exec_run("git add .", workdir="/workspace")
    # Commit a baseline so git restore has a target state
    sandbox.exec_run('bash -c "git commit -m \'baseline\' || true"', workdir="/workspace")
    
    print("[Sandbox] Cleaning workspace...")
    # Discard any failed changes from previous iterations
    sandbox.exec_run("git restore .", workdir="/workspace")
    sandbox.exec_run("git clean -fd", workdir="/workspace")
    
    # Clean up hallucinated prefixes
    diff = diff.replace("sandbox_workspace/", "")
    
    # 2. Write the patch file locally (mapped to the container)
    patch_path = os.path.join(repo_path, "fix.patch")
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    
    try:
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(diff)
    except Exception as e:
        return f"Error writing patch file locally: {e}"
        
    # 3. Apply the patch inside the container using a shell (bash -c)
    # The shell is REQUIRED to correctly interpret the '<' redirection operator
    print("[Sandbox] Applying patch...")
    patch_result = sandbox.exec_run(
        ["bash", "-c", "patch -p1 < fix.patch"], 
        workdir="/workspace"
    )
    
    output = patch_result.output.decode('utf-8', errors='ignore')
    
    if patch_result.exit_code != 0:
         return f"Error applying patch:\n{output}"
         
    return f"Patch applied successfully.\n{output}"

def run_main_in_sandbox() -> tuple[bool, str]:
    """
    Runs the main application entry point inside the sandbox.
    Uses PYTHONPATH to ensure the 'src' package is discoverable.
    """
    sandbox = get_sandbox_container()
    if not sandbox:
         return False, "Error: Sandbox container not found."
         
    print("[Sandbox] Running src/main.py...")
    # Set PYTHONPATH to '.' so 'from src.utils import calculate_total' works
    result = sandbox.exec_run(
        "python src/main.py", 
        workdir="/workspace",
        environment={"PYTHONPATH": "."}
    )
    
    output = result.output.decode('utf-8', errors='ignore')
    success = (result.exit_code == 0)
    return success, output
