"""Tests for setup node environment detection and ignore merging."""

from __future__ import annotations

from issue_resolver.nodes.setup import setup_node


def test_setup_node_dotnet_priority_and_framework_detection(tmp_path):
    (tmp_path / "project.sln").write_text("\n", encoding="utf-8")
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "tests.csproj").write_text(
        "<Project><ItemGroup><PackageReference Include='xunit' Version='2.5.0'/></ItemGroup></Project>",
        encoding="utf-8",
    )
    (tmp_path / ".gitignore").write_text("dist/\n", encoding="utf-8")

    out = setup_node(
        {
            "issue": "Title: Tab Button Shifting",
            "repo_path": str(tmp_path),
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
        }
    )

    cfg = out["environment_config"]
    assert cfg["environment_type"] == "dotnet"
    assert cfg["test_framework"] == "xunit"
    assert "dist/" in cfg["gitignore_patterns"]
    assert "node_modules" in cfg["ignore_dirs"]
