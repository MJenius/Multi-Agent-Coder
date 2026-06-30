import re
import difflib

import ast

class FuzzyPatchEngine:
    def __init__(self, file_content: str):
        self.file_content = file_content.replace("\r\n", "\n")
        self.lines = self.file_content.split('\n')

    def apply_block(self, search_text: str, replace_text: str) -> dict:
        search_text = search_text.replace("\r\n", "\n")
        replace_text = replace_text.replace("\r\n", "\n")
        
        # 1. Exact substring replace
        if search_text in self.file_content:
            self.file_content = self.file_content.replace(search_text, replace_text, 1)
            self.lines = self.file_content.split('\n')
            return {"success": True, "method": "exact_substring"}
            
        search_lines = search_text.split('\n')
        replace_lines = replace_text.split('\n')
        m = len(search_lines)
        n = len(self.lines)
        
        if m == 0:
            return {"success": False, "hint": "Search block is empty."}
            
        norm_search = [line.rstrip() for line in search_lines]
        norm_file = [line.rstrip() for line in self.lines]
        
        # 2. Normalized line match (ignoring trailing whitespace)
        for i in range(n - m + 1):
            if norm_file[i : i + m] == norm_search:
                self.lines[i : i + m] = replace_lines
                self.file_content = "\n".join(self.lines)
                return {"success": True, "method": "normalized_exact_lines"}
                
        # 3. Indentation-flexible match
        stripped_search = [line.strip() for line in search_lines]
        stripped_file = [line.strip() for line in self.lines]
        for i in range(n - m + 1):
            if stripped_file[i : i + m] == stripped_search:
                # Calculate indentation offset
                file_indent = len(self.lines[i]) - len(self.lines[i].lstrip())
                search_indent = len(search_lines[0]) - len(search_lines[0].lstrip())
                indent_offset = file_indent - search_indent
                
                adjusted_replace = []
                for r_line in replace_lines:
                    if not r_line.strip():
                        adjusted_replace.append("")
                    else:
                        r_indent = len(r_line) - len(r_line.lstrip())
                        new_indent = max(0, r_indent + indent_offset)
                        adjusted_replace.append(" " * new_indent + r_line.lstrip())
                        
                self.lines[i : i + m] = adjusted_replace
                self.file_content = "\n".join(self.lines)
                return {"success": True, "method": "indentation_flexible_lines"}
                
        # 4. AST-Context Match (for Python)
        try:
            tree = ast.parse(self.file_content)
            func_nodes = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", start + 1)
                    func_nodes.append((node.name, start, end))
                    
            for func_name, start, end in func_nodes:
                func_file_lines = self.lines[start:end]
                func_n = len(func_file_lines)
                if func_n < m:
                    continue
                
                stripped_func = [line.strip() for line in func_file_lines]
                for offset in range(func_n - m + 1):
                    if stripped_func[offset : offset + m] == stripped_search:
                        match_idx = start + offset
                        file_indent = len(self.lines[match_idx]) - len(self.lines[match_idx].lstrip())
                        search_indent = len(search_lines[0]) - len(search_lines[0].lstrip())
                        indent_offset = file_indent - search_indent
                        
                        adjusted_replace = []
                        for r_line in replace_lines:
                            if not r_line.strip():
                                adjusted_replace.append("")
                            else:
                                r_indent = len(r_line) - len(r_line.lstrip())
                                new_indent = max(0, r_indent + indent_offset)
                                adjusted_replace.append(" " * new_indent + r_line.lstrip())
                                
                        self.lines[match_idx : match_idx + m] = adjusted_replace
                        self.file_content = "\n".join(self.lines)
                        return {"success": True, "method": "ast_context_flexible_lines", "function": func_name}
        except Exception:
            pass
            
        # 5. Fuzzy Sequence Match
        best_ratio = -1.0
        best_i = -1
        search_str = "\n".join(norm_search)
        for i in range(n - m + 1):
            window_str = "\n".join(norm_file[i : i + m])
            ratio = difflib.SequenceMatcher(None, search_str, window_str).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_i = i
                
        if best_ratio >= 0.8:
            file_indent = len(self.lines[best_i]) - len(self.lines[best_i].lstrip())
            search_indent = len(search_lines[0]) - len(search_lines[0].lstrip())
            indent_offset = file_indent - search_indent
            
            adjusted_replace = []
            for r_line in replace_lines:
                if not r_line.strip():
                    adjusted_replace.append("")
                else:
                    r_indent = len(r_line) - len(r_line.lstrip())
                    new_indent = max(0, r_indent + indent_offset)
                    adjusted_replace.append(" " * new_indent + r_line.lstrip())
                    
            self.lines[best_i : best_i + m] = adjusted_replace
            self.file_content = "\n".join(self.lines)
            return {"success": True, "method": "fuzzy_sequence_match", "ratio": round(best_ratio, 2)}
            
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
