"""Tool abstraction layer.

Wraps existing repo_tools, sandbox_tools, and other utilities behind
the ``Tool`` interface so that agents call ``registry.get_tool(name).execute(...)``
instead of importing utilities directly.  This makes future tool
additions trivial — just implement ``Tool`` and register.
"""

from __future__ import annotations

from typing import Any

from issue_resolver.core.interfaces import Tool, ToolResult


# ---------------------------------------------------------------------------
# Filesystem Tool
# ---------------------------------------------------------------------------


class FilesystemTool(Tool):
    """Provides list_files, read_file, search_code, generate_repo_map,
    get_symbol_definition, and file_viewer via a unified interface."""

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return (
            "Interact with the local filesystem: list files, read files, "
            "search code, generate repo maps, and view specific file ranges."
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")
        try:
            if action == "list_files":
                from issue_resolver.tools.repo_tools import list_files
                result = list_files.invoke({"directory": params.get("directory", ".")})
                return ToolResult(success=True, output=result)

            if action == "read_file":
                from issue_resolver.tools.repo_tools import read_file
                result = read_file.invoke({"file_path": params["file_path"]})
                return ToolResult(success=not result.startswith("Error"), output=result)

            if action == "search_code":
                from issue_resolver.tools.repo_tools import search_code
                result = search_code.invoke({
                    "query": params["query"],
                    "directory": params.get("directory", "."),
                })
                return ToolResult(success=True, output=result)

            if action == "generate_repo_map":
                from issue_resolver.tools.repo_tools import generate_repo_map
                result = generate_repo_map.invoke({
                    "directory": params.get("directory", "."),
                    "max_depth": params.get("max_depth", 2),
                })
                return ToolResult(success=True, output=result)

            if action == "get_symbol_definition":
                from issue_resolver.tools.repo_tools import get_symbol_definition
                result = get_symbol_definition.invoke({
                    "symbol_name": params["symbol_name"],
                    "directory": params.get("directory", "."),
                })
                return ToolResult(success=True, output=result)

            return ToolResult(success=False, output=f"Unknown filesystem action: {action}")

        except Exception as exc:
            return ToolResult(success=False, output=f"FilesystemTool error: {exc}")


# ---------------------------------------------------------------------------
# Git Tool
# ---------------------------------------------------------------------------


class GitTool(Tool):
    """Wraps Git operations for repository management."""

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return "Git operations: create branches, commit, diff, status, log."

    def execute(self, params: dict[str, Any]) -> ToolResult:
        import subprocess

        action = params.get("action", "")
        repo_path = params.get("repo_path", ".")

        cmd_map = {
            "status": ["git", "status", "--porcelain"],
            "diff": ["git", "diff"],
            "log": ["git", "log", "--oneline", "-10"],
            "branch": ["git", "checkout", "-b", params.get("branch_name", "fix")],
            "add": ["git", "add", "."],
            "commit": ["git", "commit", "-m", params.get("message", "automated fix")],
        }

        cmd = cmd_map.get(action)
        if not cmd:
            return ToolResult(success=False, output=f"Unknown git action: {action}")

        try:
            result = subprocess.run(
                cmd, cwd=repo_path, capture_output=True, text=True,
                timeout=30, check=False,
            )
            output = (result.stdout + "\n" + result.stderr).strip()
            return ToolResult(success=result.returncode == 0, output=output)
        except Exception as exc:
            return ToolResult(success=False, output=f"GitTool error: {exc}")


# ---------------------------------------------------------------------------
# Docker Sandbox Tool
# ---------------------------------------------------------------------------


class DockerTool(Tool):
    """Wraps Docker sandbox interactions for patch application and testing."""

    @property
    def name(self) -> str:
        return "docker"

    @property
    def description(self) -> str:
        return "Docker sandbox: apply diffs, run tests, clean sandbox."

    def execute(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")
        try:
            if action == "apply_diff":
                from issue_resolver.tools.sandbox_tools import apply_diff_in_sandbox
                result = apply_diff_in_sandbox(params["diff"], params["repo_path"])
                return ToolResult(
                    success="Error" not in result, output=result,
                )

            if action == "run_tests":
                from issue_resolver.tools.sandbox_tools import run_tests_in_sandbox
                success, output = run_tests_in_sandbox(params["diff"])
                return ToolResult(success=success, output=output)

            if action == "clean":
                from issue_resolver.tools.sandbox_tools import clean_sandbox
                clean_sandbox()
                return ToolResult(success=True, output="Sandbox cleaned")

            return ToolResult(success=False, output=f"Unknown docker action: {action}")
        except Exception as exc:
            return ToolResult(success=False, output=f"DockerTool error: {exc}")


# ---------------------------------------------------------------------------
# GitHub Tool
# ---------------------------------------------------------------------------


class GitHubTool(Tool):
    """Wraps GitHub API interactions."""

    @property
    def name(self) -> str:
        return "github"

    @property
    def description(self) -> str:
        return "GitHub API: fetch issues, submit pull requests."

    def execute(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")
        try:
            if action == "fetch_issue":
                from issue_resolver.utils.github_utils import fetch_issue_details
                title, body = fetch_issue_details(
                    params["repo_url"],
                    int(params["issue_number"]),
                    params["token"],
                )
                return ToolResult(
                    success=True,
                    output=f"Title: {title}\n\nBody: {body or ''}",
                    metadata={"title": title, "body": body},
                )

            if action == "submit_pr":
                from issue_resolver.utils.github_utils import submit_pull_request
                pr_url = submit_pull_request(
                    repo_path=params["repo_path"],
                    repo_full_name=params["repo_url"],
                    issue_number=int(params["issue_number"]),
                    token=params["token"],
                    proposed_fix=params["proposed_fix"],
                )
                return ToolResult(success=True, output=pr_url)

            return ToolResult(success=False, output=f"Unknown github action: {action}")
        except Exception as exc:
            return ToolResult(success=False, output=f"GitHubTool error: {exc}")


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_builtin_tools(registry: Any) -> None:
    """Register all built-in tools into the given PluginRegistry."""
    registry.register_tool(FilesystemTool())
    registry.register_tool(GitTool())
    registry.register_tool(DockerTool())
    registry.register_tool(GitHubTool())
