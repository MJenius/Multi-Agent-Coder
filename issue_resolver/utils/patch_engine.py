import re
import difflib

class FuzzyPatchEngine:
    def __init__(self, file_content: str):
        self.file_content = file_content
        self.lines = file_content.split('\n')

    def apply_block(self, search_text: str, replace_text: str) -> dict:
        search_lines = search_text.split('\n')
        replace_lines = replace_text.split('\n')
        if not search_lines:
            return {"success": False, "hint": "Search block is empty."}
        norm_search = [line.rstrip() for line in search_lines]
        norm_file = [line.rstrip() for line in self.lines]
        n = len(self.lines)
        m = len(search_lines)
        if m > n:
            best_i = 0
            best_ratio = difflib.SequenceMatcher(None, "\n".join(norm_search), "\n".join(norm_file)).ratio()
        else:
            best_ratio = -1.0
            best_i = -1
            search_str = "\n".join(norm_search)
            for i in range(n - m + 1):
                window_str = "\n".join(norm_file[i : i + m])
                ratio = difflib.SequenceMatcher(None, search_str, window_str).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_i = i
        if best_ratio == 1.0:
            self.lines[best_i : best_i + m] = replace_lines
            self.file_content = "\n".join(self.lines)
            return {"success": True}
        if best_ratio >= 0.92:
            self.lines[best_i : best_i + m] = replace_lines
            self.file_content = "\n".join(self.lines)
            return {"success": True}
        if best_ratio >= 0.82:
            self.lines[best_i : best_i + m] = replace_lines
            self.file_content = "\n".join(self.lines)
            return {"success": True}
        line_num = best_i + 1 if best_i != -1 else 1
        return {
            "success": False,
            "hint": f"Failed to apply patch. Closest match at line {line_num} with similarity {best_ratio:.2f}."
        }

def parse_and_apply_blocks(file_path: str, llm_output: str) -> dict:
    pattern = re.compile(
        r"^<<<<<<< SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>> REPLACE$",
        re.MULTILINE | re.DOTALL
    )
    blocks = pattern.findall(llm_output)
    if not blocks:
        return {"success": False, "hint": "No SEARCH/REPLACE blocks found in output."}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    engine = FuzzyPatchEngine(content)
    for search_text, replace_text in blocks:
        res = engine.apply_block(search_text, replace_text)
        if not res["success"]:
            return res
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        f.write(engine.file_content)

    import os
    import subprocess
    from issue_resolver.tools.sandbox_tools import get_sandbox_container

    lint_failed = False
    lint_output = ""
    sandbox = get_sandbox_container()

    if sandbox:
        from issue_resolver.config import SANDBOX_WORKSPACE_DIR
        sandbox_workspace_abs = os.path.abspath(SANDBOX_WORKSPACE_DIR)
        file_path_abs = os.path.abspath(file_path)
        try:
            rel_path = os.path.relpath(file_path_abs, sandbox_workspace_abs).replace("\\", "/")
        except ValueError:
            rel_path = os.path.basename(file_path)

        res_lint = sandbox.exec_run(f"ruff check {rel_path}", workdir="/workspace")
        if res_lint.exit_code != 0:
            output = res_lint.output.decode("utf-8", errors="ignore")
            if "not found" in output.lower() or "command not found" in output.lower() or res_lint.exit_code == 127:
                res_lint = sandbox.exec_run(f"python -m py_compile {rel_path}", workdir="/workspace")
                if res_lint.exit_code != 0:
                    lint_failed = True
                    lint_output = res_lint.output.decode("utf-8", errors="ignore")
            else:
                lint_failed = True
                lint_output = output

        if lint_failed:
            sandbox.exec_run(f"git checkout -- {rel_path}", workdir="/workspace")
            return {"success": False, "hint": f"Linter validation failed:\n{lint_output.strip()}"}
    else:
        try:
            res_lint = subprocess.run(
                ["ruff", "check", file_path],
                capture_output=True,
                text=True
            )
            if res_lint.returncode != 0:
                lint_failed = True
                lint_output = res_lint.stdout + "\n" + res_lint.stderr
        except FileNotFoundError:
            try:
                res_lint = subprocess.run(
                    ["python", "-m", "py_compile", file_path],
                    capture_output=True,
                    text=True
                )
                if res_lint.returncode != 0:
                    lint_failed = True
                    lint_output = res_lint.stdout + "\n" + res_lint.stderr
            except Exception:
                pass
        if lint_failed:
            curr_dir = os.path.dirname(os.path.abspath(file_path))
            git_root = None
            while curr_dir:
                if os.path.exists(os.path.join(curr_dir, ".git")):
                    git_root = curr_dir
                    break
                parent = os.path.dirname(curr_dir)
                if parent == curr_dir:
                    break
                curr_dir = parent
            cwd = git_root if git_root else os.path.dirname(os.path.abspath(file_path))
            subprocess.run(
                ["git", "checkout", "--", file_path],
                cwd=cwd,
                capture_output=True,
                text=True
            )
            return {"success": False, "hint": f"Linter validation failed:\n{lint_output.strip()}"}

    return {"success": True}
