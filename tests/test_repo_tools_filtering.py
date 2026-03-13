"""Tests for repo tool filtering and weighted map safeguards."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from issue_resolver.config import SANDBOX_WORKSPACE_DIR
from issue_resolver.runtime_context import set_environment_config
from issue_resolver.tools.repo_tools import generate_repo_map, list_files


def test_generate_repo_map_token_guard_and_weighting():
    test_root = Path(SANDBOX_WORKSPACE_DIR) / f"_pytest_map_{uuid.uuid4().hex}"
    test_root.mkdir(parents=True, exist_ok=True)

    try:
        (test_root / "src" / "ui").mkdir(parents=True)
        (test_root / "src" / "backend").mkdir(parents=True)
        (test_root / "logs").mkdir(parents=True)
        (test_root / "logs" / "noise.ts").write_text("export const x = 1;", encoding="utf-8")
        (test_root / "src" / "ui" / "button.tsx").write_text("export const Button = 1;", encoding="utf-8")
        (test_root / "src" / "backend" / "service.ts").write_text("export const Svc = 1;", encoding="utf-8")
        (test_root / ".gitignore").write_text("logs/\n", encoding="utf-8")
        long_readme = "README line " + ("x" * 140) + "\n"
        (test_root / "README.md").write_text(long_readme * 120, encoding="utf-8")

        set_environment_config(
            {
                "issue_title": "Tab Button Shifting in UI view",
                "ignore_dirs": ["node_modules", "__pycache__"],
                "gitignore_patterns": ["logs/"],
            }
        )

        file_listing = list_files.invoke({"directory": str(test_root)})
        assert "logs/noise.ts" not in file_listing

        repo_map = generate_repo_map.invoke({"directory": str(test_root), "max_depth": 2})
        assert "[TRUNCATED at 10,000 characters]" in repo_map
        assert "Drill down into subdirectories" in repo_map
        assert "src/ui/" in repo_map or "ui/" in repo_map
    finally:
        shutil.rmtree(test_root, ignore_errors=True)
