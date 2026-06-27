from __future__ import annotations
import json
import os
from pathlib import Path
import pytest
from issue_resolver.tools.view_engine import file_viewer
from issue_resolver.utils.code_mapper import CodeMapper
from issue_resolver.utils.patch_engine import parse_and_apply_blocks
from issue_resolver.config import SANDBOX_WORKSPACE_DIR

def test_file_viewer_basic(tmp_path):
    f = tmp_path / "test.py"
    lines = [f"line {i}" for i in range(1, 201)]
    f.write_text("\n".join(lines), encoding="utf-8")
    
    import issue_resolver.tools.view_engine
    orig_dir = issue_resolver.tools.view_engine.SANDBOX_WORKSPACE_DIR
    issue_resolver.tools.view_engine.SANDBOX_WORKSPACE_DIR = str(tmp_path)
    
    try:
        res = file_viewer.invoke({
            "command": "view_window",
            "file_path": "test.py",
            "window_size": 10
        })
        parsed = json.loads(res)
        assert parsed["new_view_file"] == "test.py"
        assert parsed["new_view_line"] == 1
        assert "LINES: 1-10 OF 200" in parsed["output"]
        assert "   1 | line 1" in parsed["output"]
        
        res_scroll = file_viewer.invoke({
            "command": "scroll_down",
            "current_view_file": "test.py",
            "current_view_line": 1,
            "window_size": 10
        })
        parsed_scroll = json.loads(res_scroll)
        assert parsed_scroll["new_view_line"] == 11
        assert "LINES: 11-20 OF 200" in parsed_scroll["output"]
        
        res_scroll_up = file_viewer.invoke({
            "command": "scroll_up",
            "current_view_file": "test.py",
            "current_view_line": 11,
            "window_size": 10
        })
        parsed_scroll_up = json.loads(res_scroll_up)
        assert parsed_scroll_up["new_view_line"] == 1
        
        res_goto = file_viewer.invoke({
            "command": "goto_line",
            "line_number": 50,
            "current_view_file": "test.py",
            "current_view_line": 1,
            "window_size": 10
        })
        parsed_goto = json.loads(res_goto)
        assert parsed_goto["new_view_line"] == 50
        assert "LINES: 50-59 OF 200" in parsed_goto["output"]
    finally:
        issue_resolver.tools.view_engine.SANDBOX_WORKSPACE_DIR = orig_dir

def test_code_mapper(tmp_path):
    f = tmp_path / "mod.py"
    code = (
        "import os\n"
        "from pathlib import Path\n"
        "class A:\n"
        "    def foo(self, x: int):\n"
        "        pass\n"
        "def bar(y: str) -> None:\n"
        "    pass\n"
    )
    f.write_text(code, encoding="utf-8")
    
    outline = CodeMapper.generate_outline(str(f))
    assert "Class A" in outline
    assert "Method: def foo" in outline
    assert "Function: def bar" in outline
    
    repo_map = CodeMapper.generate_repo_map(str(tmp_path))
    assert "MAP FOR mod.py" in repo_map

def test_linter_gate_fail(tmp_path, monkeypatch):
    monkeypatch.setattr("issue_resolver.tools.sandbox_tools.get_sandbox_container", lambda: None)
    f = tmp_path / "bad.py"
    f.write_text("def ok():\n    pass\n", encoding="utf-8")
    
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@local"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "bad.py"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(tmp_path), capture_output=True)
    
    orig_dir = os.environ.get("SANDBOX_WORKSPACE_DIR")
    os.environ["SANDBOX_WORKSPACE_DIR"] = str(tmp_path)
    
    llm_output = (
        "File: bad.py\n"
        "<<<<<<< SEARCH\n"
        "def ok():\n"
        "    pass\n"
        "=======\n"
        "def ok(\n"
        "    print('syntax error'\n"
        ">>>>>>> REPLACE"
    )
    
    try:
        res = parse_and_apply_blocks(str(f), llm_output)
        assert res["success"] is False
        assert "Linter validation failed" in res["hint"]
        
        reverted = f.read_text(encoding="utf-8")
        assert "print('syntax error'" not in reverted
        assert "def ok():" in reverted
    finally:
        if orig_dir:
            os.environ["SANDBOX_WORKSPACE_DIR"] = orig_dir
        else:
            del os.environ["SANDBOX_WORKSPACE_DIR"]
