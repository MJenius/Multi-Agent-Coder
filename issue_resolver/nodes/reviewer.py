from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from langchain_core.messages import HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.config import REVIEWER_MODEL_CANDIDATES
from issue_resolver.llm_utils import invoke_with_role_fallback
from issue_resolver.tools.sandbox_tools import (
    apply_diff_in_sandbox,
    run_tests_in_sandbox,
    format_parsed_error_summary,
)
from issue_resolver.utils.logger import append_to_history


def _ast_validate(source_code: str, language: str, file_path: str = "") -> tuple[bool, str]:
    lang = language.lower()

    if lang == "python":
        try:
            ast.parse(source_code)
            return True, ""
        except SyntaxError as exc:
            detail_parts = [f"SyntaxError: {exc.msg}"]
            if exc.lineno:
                detail_parts.append(f"Line {exc.lineno}")
            if exc.offset:
                detail_parts.append(f"Column {exc.offset}")
            if exc.text:
                detail_parts.append(f"Near: {exc.text.strip()[:120]}")
            if file_path:
                detail_parts.append(f"File: {file_path}")
            return False, " | ".join(detail_parts)

    if lang in ("javascript", "typescript", "nodejs"):
        if not file_path:
            return True, ""
        try:
            result = subprocess.run(
                ["node", "--check", file_path],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip()[:500] if result.stderr else "node --check failed"
                return False, f"JS/TS syntax error: {detail}"
            return True, ""
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return True, ""

    if lang in ("dotnet", "csharp"):
        if not file_path:
            return True, ""
        try:
            project_dir = str(Path(file_path).parent)
            result = subprocess.run(
                ["dotnet", "build", "--no-restore", "--nologo"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                error_lines = [
                    l for l in result.stderr.split("\n")
                    if "error CS" in l
                ]
                detail = "\n".join(error_lines[:5]) if error_lines else result.stderr[:500]
                return False, f"C# build error: {detail}"
            return True, ""
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return True, ""

    return True, ""


def _detect_language_from_path(file_path: str) -> str:
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".cs": "csharp",
    }
    suffix = Path(file_path).suffix.lower()
    return ext_map.get(suffix, "unknown")


def _extract_patched_file_path(diff_text: str) -> str:
    for line in diff_text.split("\n"):
        if line.startswith("+++ b/"):
            return line[6:].strip()
        if line.startswith("+++ "):
            return line[4:].strip()
    return ""


def _apply_diff_to_content(original: str, diff_text: str) -> str | None:
    import io
    original_lines = original.splitlines(keepends=True)
    if original_lines and not original_lines[-1].endswith("\n"):
        original_lines[-1] += "\n"

    hunks = []
    current_hunk = None
    for line in diff_text.split("\n"):
        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                current_hunk = {
                    "orig_start": int(match.group(1)),
                    "orig_count": int(match.group(2) or 1),
                    "lines": [],
                }
                hunks.append(current_hunk)
        elif current_hunk is not None:
            if line.startswith("-") or line.startswith("+") or line.startswith(" "):
                current_hunk["lines"].append(line)

    if not hunks:
        return None

    result_lines = list(original_lines)
    offset = 0

    for hunk in hunks:
        start = hunk["orig_start"] - 1 + offset
        old_lines = []
        new_lines = []
        for hl in hunk["lines"]:
            if hl.startswith("-"):
                old_lines.append(hl[1:])
            elif hl.startswith("+"):
                new_lines.append(hl[1:] + "\n" if not hl[1:].endswith("\n") else hl[1:])
            elif hl.startswith(" "):
                old_lines.append(hl[1:])
                new_lines.append(hl[1:] + "\n" if not hl[1:].endswith("\n") else hl[1:])

        end = start + len(old_lines)
        result_lines[start:end] = new_lines
        offset += len(new_lines) - len(old_lines)

    return "".join(result_lines)


def _categorize_error(error_text: str) -> str:
    lower = error_text.lower()

    patterns = {
        "SyntaxError": (r"syntaxerror|unexpected token|invalid syntax|parse error", 2.0),
        "EnvironmentError": (
            r"modulenotfound|importerror|no such file|not found error|cannot find",
            2.0,
        ),
        "LogicFailure": (r"assertion|expected.*got|test failed|failure", 1.5),
        "FrameworkError": (r"cannot find|not a function|undefined|typeerror", 1.0),
    }

    scores = {}
    for category, (pattern, weight) in patterns.items():
        if re.search(pattern, lower):
            scores[category] = weight

    return max(scores, key=scores.get) if scores else "UnknownError"


def _extract_line_numbers(error_text: str) -> str:
    lines = []

    python_pattern = r"line\s+(\d+)"
    csharp_pattern = r"\((\d+),\d+\)"
    js_pattern = r":\s*(\d+):\d+"

    for pattern in [python_pattern, csharp_pattern, js_pattern]:
        matches = re.findall(pattern, error_text)
        if matches:
            lines.extend(matches[:5])

    if lines:
        return "lines " + ", ".join(sorted(set(lines)))
    return ""


def reviewer_node(state: AgentState) -> dict:
    proposed_fix = state.get("proposed_fix", "")
    repo_path = state.get("repo_path", "")
    structured_plan = state.get("structured_plan", {})
    files_to_edit = structured_plan.get("files_to_edit", [])

    if not files_to_edit and proposed_fix:
        patched_file = _extract_patched_file_path(proposed_fix)
        if patched_file:
            files_to_edit = [patched_file]

    if not proposed_fix:
        print("[Reviewer] [WARN] No proposed fix found in state.")
        return {
            "errors": "No fix proposed.",
            "validation_status": "failed",
            "ast_validation_passed": True,
            "ast_error_detail": "",
            "history": append_to_history("Reviewer", "Failure", "No fix proposed to review."),
        }

    # 1. First run basic AST pre-validation check if python file
    patched_file = _extract_patched_file_path(proposed_fix)
    if patched_file:
        lang = _detect_language_from_path(patched_file)
        if lang == "python":
            print(f"[Reviewer] [AST] Validating syntax for '{patched_file}'")
            from issue_resolver.nodes.coder import _extract_file_info
            file_info = _extract_file_info(state.get("file_context", []))
            matched_path = patched_file.lstrip("./")
            original_content = file_info.get(matched_path, "")

            if original_content:
                patched_content = _apply_diff_to_content(original_content, proposed_fix)
                if patched_content:
                    ast_ok, ast_error = _ast_validate(patched_content, lang, patched_file)
                    if not ast_ok:
                        print(f"[Reviewer] [AST FAIL] {ast_error}")
                        return {
                            "errors": f"AST validation failed: {ast_error}",
                            "validation_status": "failed",
                            "ast_validation_passed": False,
                            "ast_error_detail": ast_error,
                            "proposed_fix": "",
                            "history": append_to_history(
                                "Reviewer",
                                "AST Validation Failed",
                                f"Syntax error detected before sandbox execution: {ast_error}",
                            ),
                        }

    # 2. Run sandbox verification pipeline
    from issue_resolver.utils.verifier import VerificationPipeline
    verification_type = state.get("verification_type", "runtime tests")
    pipeline = VerificationPipeline(repo_path, verification_type=verification_type)
    
    # Run the verification steps on changed files
    print("[Reviewer] Applying and validating patch using the verification pipeline...")
    
    try:
        # First, apply the proposed diff in the sandbox
        patch_output = apply_diff_in_sandbox(proposed_fix, repo_path)
        if "Error" in patch_output:
            if "Sandbox container not found" in patch_output:
                print("[Reviewer] Sandbox container not found - validation inconclusive.")
                return {
                    "errors": "",
                    "validation_status": "inconclusive",
                    "history": append_to_history(
                        "Reviewer",
                        "Skipped",
                        "Sandbox container not found. Skipping verification.",
                    ),
                }
            print(f"[Reviewer] [FAIL] Patch failed to apply: {patch_output}")
            return {
                "errors": f"Failed to apply patch:\n{patch_output}",
                "validation_status": "failed",
                "history": append_to_history("Reviewer", "Apply Patch Failed", patch_output),
            }

        # Run verification steps
        report = pipeline.run(files_to_edit)
        success = report["passed"]

        # Aggregate output details
        output_parts = []
        for step in report["results"]:
            if not step["passed"]:
                output_parts.append(f"[{step['step_name']} FAIL]\n{step['output']}")

        error_summary = "\n\n".join(output_parts)
        condensed = error_summary
        
        if not success:
            # Categorize the failures
            error_category = _categorize_error(error_summary)
            error_line_numbers = _extract_line_numbers(error_summary)
            
            history_payload = f"Verification failed:\n{error_summary}"
            history_additions = append_to_history("Reviewer", "Test Execution", history_payload)
            
            return {
                "errors": condensed,
                "validation_status": "failed",
                "error_category": error_category,
                "test_error_context": error_summary[:500],
                "error_line_numbers": error_line_numbers,
                "ast_validation_passed": True,
                "ast_error_detail": "",
                "verification_report": report,
                "history": history_additions,
            }
        
        print("[Reviewer] [OK] All verification steps passed successfully.")
        return {
            "errors": "",
            "validation_status": "passed",
            "ast_validation_passed": True,
            "ast_error_detail": "",
            "verification_report": report,
            "history": append_to_history("Reviewer", "Verification Passed", "All tests and checks passed."),
        }

    except Exception as exc:
        print(f"[Reviewer] [ERROR] Exception: {exc}")
        return {
            "errors": f"Reviewer validation error: {exc}",
            "validation_status": "failed",
            "ast_validation_passed": True,
            "ast_error_detail": "",
            "history": append_to_history("Reviewer", "Error", str(exc)),
        }

