from __future__ import annotations
from pathlib import Path
import json
from langchain_core.tools import tool
from issue_resolver.config import SANDBOX_WORKSPACE_DIR

class FileViewer:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def view_window(self, file_path: str, first_line: int, window_size: int = 100) -> str:
        full_path = self.base_dir / file_path
        if not full_path.exists():
            return f"ERROR: File {file_path} not found"
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)
        start = max(1, min(first_line, total_lines))
        end = min(start + window_size - 1, total_lines)
        output = [f"--- FILE: {file_path}, LINES: {start}-{end} OF {total_lines} ---"]
        for idx in range(start, end + 1):
            output.append(f"{idx:4d} | {lines[idx - 1]}")
        output.append("-" * len(output[0]))
        return "\n".join(output)

@tool(description="Stateful windowed file viewer. Commands: view_window, scroll_down, scroll_up, goto_line.")
def file_viewer(
    command: str,
    file_path: str | None = None,
    line_number: int | None = None,
    window_size: int = 100,
    current_view_file: str | None = None,
    current_view_line: int = 1
) -> str:
    viewer = FileViewer(SANDBOX_WORKSPACE_DIR)
    active_file = file_path if file_path is not None else current_view_file
    if not active_file:
        return json.dumps({
            "output": "ERROR: No active file set.",
            "new_view_file": None,
            "new_view_line": 1
        })
    full_path = viewer.base_dir / active_file
    if not full_path.exists():
        return json.dumps({
            "output": f"ERROR: File {active_file} not found",
            "new_view_file": current_view_file,
            "new_view_line": current_view_line
        })
    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return json.dumps({
            "output": f"ERROR: Failed to read file {active_file}: {e}",
            "new_view_file": current_view_file,
            "new_view_line": current_view_line
        })
    total_lines = len(lines)
    active_line = current_view_line
    if command == "view_window":
        if line_number is not None:
            active_line = line_number
        elif file_path is not None:
            active_line = 1
    elif command == "scroll_down":
        if active_line + window_size > total_lines:
            active_line = max(1, total_lines - window_size + 1)
        else:
            active_line += window_size
    elif command == "scroll_up":
        active_line = max(1, active_line - window_size)
    elif command == "goto_line":
        if line_number is not None:
            active_line = line_number
    active_line = max(1, min(active_line, total_lines))
    output = viewer.view_window(active_file, active_line, window_size)
    return json.dumps({
        "output": output,
        "new_view_file": active_file,
        "new_view_line": active_line
    })
