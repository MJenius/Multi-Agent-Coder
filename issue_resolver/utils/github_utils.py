import os
import git
from github import Github
from typing import Tuple

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
    
    import subprocess
    patch_path = os.path.join(repo_path, "fix.patch")
    with open(patch_path, "w") as f:
        f.write(proposed_fix)
    
    try:
        subprocess.run(["patch", "-p1", "-i", "fix.patch"], cwd=repo_path, check=True)
        os.remove(patch_path)
    except subprocess.CalledProcessError as e:
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
    
    return pr.html_url
