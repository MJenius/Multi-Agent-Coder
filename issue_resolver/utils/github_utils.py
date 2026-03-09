import os
import re
import git
from github import Github, GithubException
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


def _get_or_create_fork(g: Github, repo_full_name: str) -> str:
    """
    Returns the full_name of the user's fork.
    Creates the fork if it doesn't exist yet.
    """
    user = g.get_user()
    username = user.login
    repo_name = repo_full_name.split("/")[-1]
    fork_full_name = f"{username}/{repo_name}"

    # Check if fork already exists
    try:
        fork = g.get_repo(fork_full_name)
        if fork.fork and fork.parent and fork.parent.full_name == repo_full_name:
            print(f"[GitHub] Fork already exists: {fork_full_name}")
            return fork_full_name
    except GithubException:
        pass

    # Create the fork
    print(f"[GitHub] Creating fork of {repo_full_name}...")
    upstream = g.get_repo(repo_full_name)
    fork = user.create_fork(upstream)
    print(f"[GitHub] Fork created: {fork.full_name}")
    return fork.full_name


def _detect_default_branch(g: Github, repo_full_name: str) -> str:
    """Detect the default branch of the upstream repository."""
    repo = g.get_repo(repo_full_name)
    return repo.default_branch


def submit_pull_request(
    repo_path: str,
    repo_full_name: str,
    issue_number: int,
    token: str,
    proposed_fix: str,
    base_branch: str = None
) -> str:
    """
    Creates a new branch, applies the fix, pushes to remote, and submits a PR.
    Automatically forks the repo if the user doesn't have write access.
    """
    g = Github(token)
    remote_repo = g.get_repo(repo_full_name)

    # Auto-detect default branch if not specified
    if not base_branch:
        base_branch = remote_repo.default_branch
        print(f"[GitHub] Detected default branch: {base_branch}")

    # Check if user has push access — if not, fork
    username = g.get_user().login
    has_push = remote_repo.permissions and remote_repo.permissions.push
    if has_push:
        push_repo_name = repo_full_name
        pr_head_prefix = ""
        print(f"[GitHub] User has push access to {repo_full_name}")
    else:
        push_repo_name = _get_or_create_fork(g, repo_full_name)
        pr_head_prefix = f"{username}:"
        print(f"[GitHub] No push access. Using fork: {push_repo_name}")

    # Initialize GitPython repo
    repo = git.Repo(repo_path)

    # Create a new branch
    branch_name = f"fix/issue-{issue_number}"
    new_branch = repo.create_head(branch_name)
    new_branch.checkout()

    # Apply the proposed fix via Docker sandbox
    import docker
    from issue_resolver.tools.sandbox_tools import _repair_diff_hunks

    # Clean the diff the same way the sandbox does before patching
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
        client = docker.from_env()
        containers = client.containers.list(filters={"label": "com.issue_resolver.role=sandbox"})
        if not containers:
            raise Exception("Sandbox container not found; cannot apply patch.")
        sandbox = containers[0]

        sandbox.exec_run("git reset --hard", workdir="/workspace")

        res = sandbox.exec_run(["bash", "-c", "patch -l --fuzz=3 -p1 < fix.patch"], workdir="/workspace")
        os.remove(patch_path)

        if res.exit_code != 0:
            raise Exception(f"Sandbox patch failed: {res.output.decode('utf-8', errors='ignore')}")

    except Exception as e:
        if os.path.exists(patch_path):
            os.remove(patch_path)
        raise Exception(f"Failed to apply patch: {e}")

    # Commit
    repo.git.add(A=True)
    repo.index.commit(f"Fix for issue #{issue_number}")

    # Push — use the fork URL if we don't have direct push access
    origin = repo.remote(name='origin')
    push_url = f"https://{token}@github.com/{push_repo_name}.git"
    origin.set_url(push_url)

    # Force-push in case the branch already exists from a previous attempt
    try:
        origin.push(branch_name)
    except git.GitCommandError:
        origin.push(branch_name, force=True)

    # Create PR on the UPSTREAM repo (cross-fork if needed)
    pr_head = f"{pr_head_prefix}{branch_name}"
    try:
        pr = remote_repo.create_pull(
            title=f"Fix for issue #{issue_number}",
            body=f"This PR automatically resolves issue #{issue_number}.",
            head=pr_head,
            base=base_branch
        )
        print(f"[GitHub] PR created: {pr.html_url}")
    except GithubException as e:
        # If PR already exists, find and return it
        if e.status == 422 and "already exists" in str(e.data).lower():
            pulls = remote_repo.get_pulls(state="open", head=pr_head)
            for p in pulls:
                print(f"[GitHub] PR already exists: {p.html_url}")
                return p.html_url
        raise

    # Cleanup
    try:
        clean_sandbox()
    except Exception:
        pass

    def _rmtree_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    try:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path, onerror=_rmtree_readonly)
    except Exception:
        pass

    return pr.html_url
