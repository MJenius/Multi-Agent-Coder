import os
import git
from github import Github
from typing import Tuple
import shutil
import stat
from issue_resolver.tools.sandbox_tools import clean_sandbox

def fetch_issue_details(repo_full_name: str, issue_number: int, token: str) -> Tuple[str, str]:
    """
    Fetches the title and body of a GitHub issue.
    """
    g = Github(token)
    repo = g.get_repo(repo_full_name)
    issue = repo.get_issue(number=issue_number)
    return issue.title, issue.body

def submit_pull_request(
    repo_path: str,
    repo_full_name: str,
    issue_number: int,
    token: str,
    proposed_fix: str,
    base_branch: str = "main"
) -> str:
    """
    Creates a new branch, applies the fix, pushes to remote, and submits a PR.
    """
    g = Github(token)
    remote_repo = g.get_repo(repo_full_name)
    
    # Initialize GitPython repo
    repo = git.Repo(repo_path)
    
    # Create a new branch
    branch_name = f"fix/issue-{issue_number}"
    new_branch = repo.create_head(branch_name)
    new_branch.checkout()
    
    # Apply the proposed fix (assuming it's a unified diff)
    # We'll use the 'patch' command via subprocess for reliability if it's a diff,
    # or handle file writes if it's a full file.
    # The requirement says 'apply the proposed_fix', which usually implies a diff in this system.
    
    import docker
    from issue_resolver.tools.sandbox_tools import _repair_diff_hunks
    import re
    
    # 1. Clean the diff the same way the sandbox does before patching
    diff = proposed_fix.replace("\r\n", "\n").strip()
    diff = diff.replace("sandbox_workspace/", "")
    if diff.startswith("diff\n"):
        diff = diff[5:].strip()
        
    cleaned_lines = []
    for line in diff.split("\n"):
        if line.startswith("---") or line.startswith("+++"):
            cleaned_lines.append(line)
        else:
            m = re.match(r"^([\+\-\s])(?: *)?\d+:(?: *)(.*)$", line)
            if m:
                cleaned_lines.append(f"{m.group(1)}{m.group(2)}")
            else:
                cleaned_lines.append(line)
    diff = "\n".join(cleaned_lines)
    diff = _repair_diff_hunks(diff)
    
    patch_path = os.path.join(repo_path, "fix.patch")
    with open(patch_path, "w", encoding="utf-8") as f:
        if not diff.endswith("\n"):
            diff += "\n"
        f.write(diff)
    
    try:
        # Use the sandbox container to reliably apply the patch since Windows native doesn't have `patch`
        client = docker.from_env()
        containers = client.containers.list(filters={"label": "com.issue_resolver.role=sandbox"})
        if not containers:
            raise Exception("Sandbox container not found; cannot apply patch.")
        sandbox = containers[0]
        
        # We must reset any uncommitted changes from the test run so we don't double-apply
        sandbox.exec_run("git reset --hard", workdir="/workspace")
        
        # Apply the patch using the fuzzy Linux patching tool
        res = sandbox.exec_run(["bash", "-c", "patch -l --fuzz=3 -p1 < fix.patch"], workdir="/workspace")
        os.remove(patch_path)
        
        if res.exit_code != 0:
            raise Exception(f"Sandbox patch failed: {res.output.decode('utf-8', errors='ignore')}")
            
    except Exception as e:
        if os.path.exists(patch_path):
            os.remove(patch_path)
        raise Exception(f"Failed to apply patch: {e}")

    # Commit and push
    repo.git.add(A=True)
    repo.index.commit(f"Fix for issue #{issue_number}")
    
    # Push to remote (need to include token in URL for auth if not configured)
    origin = repo.remote(name='origin')
    # Update origin URL to include token for pushing
    remote_url = remote_repo.clone_url.replace("https://", f"https://{token}@")
    origin.set_url(remote_url)
    origin.push(branch_name)
    
    # Create PR
    pr = remote_repo.create_pull(
        title=f"Fix for issue #{issue_number}",
        body=f"This PR automatically resolves issue #{issue_number}.",
        head=branch_name,
        base=base_branch
    )
    
    # 7. CLEAN THE SANDBOX AFTER SUBMISSION
    # Container cleanup
    try:
        clean_sandbox()
    except:
        pass # Best effort

    # Host-side local workspace cleanup
    def _rmtree_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    try:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path, onerror=_rmtree_readonly)
    except:
        pass # Best effort

    return pr.html_url
