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
    test_code = state.get("test_code", "")
    test_file_path = state.get("test_file_path", "")
    file_context = state.get("file_context", [])
    environment_config = state.get("environment_config", {})
    env_type = (
        environment_config.get("environment_type", "python")
        if isinstance(environment_config, dict)
        else "python"
    )

    if not proposed_fix:
        print("[Reviewer] [WARN] No proposed fix found in state.")
        return {
            "errors": "No fix proposed.",
            "validation_status": "failed",
            "ast_validation_passed": True,
            "ast_error_detail": "",
            "history": append_to_history("Reviewer", "Failure", "No fix proposed to review."),
        }

    patched_file = _extract_patched_file_path(proposed_fix)
    if patched_file:
        lang = _detect_language_from_path(patched_file)
        print(f"[Reviewer] [AST] Validating syntax for '{patched_file}' (lang={lang})")

        from issue_resolver.nodes.coder import _extract_file_info
        file_info = _extract_file_info(file_context)

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
                            f"Syntax error detected before test execution: {ast_error}",
                        ),
                    }
                print("[Reviewer] [AST OK] Syntax validation passed")
            else:
                print("[Reviewer] [AST SKIP] Could not apply diff for pre-validation")
        else:
            print(f"[Reviewer] [AST SKIP] Original content for '{matched_path}' not in context")

    try:
        print("[Reviewer] Applying proposed fix in the sandbox...")
        patch_output = apply_diff_in_sandbox(proposed_fix, repo_path)

        if test_code and test_file_path:
            print(
                f"[Reviewer] [TEST-DRIVEN] Test file '{test_file_path}' "
                f"will be validated after fix is applied."
            )

        if "Error" in patch_output:
            if "Sandbox container not found" in patch_output:
                print("[Reviewer] Docker sandbox unavailable -- validation inconclusive.")
                return {
                    "errors": "",
                    "validation_status": "inconclusive",
                    "ast_validation_passed": True,
                    "ast_error_detail": "",
                    "history": append_to_history(
                        "Reviewer",
                        "Skipped",
                        "Docker sandbox unavailable. Fix generated but not validated.",
                    ),
                }
            print("[Reviewer] [FAIL] Patch failed to apply.")
            return {
                "errors": f"Failed to apply patch:\n{patch_output}",
                "validation_status": "failed",
                "ast_validation_passed": True,
                "ast_error_detail": "",
                "history": append_to_history("Reviewer", "Apply Patch Failed", patch_output),
            }

        print("[Reviewer] Running validation in sandbox...")
        success, output = run_tests_in_sandbox(proposed_fix)

        parsed_summary = format_parsed_error_summary(env_type, output)

        condensed = parsed_summary
        if not success:
            prompt = (
                "Summarize this build/test failure for a coding agent in one line. "
                "Include the most actionable file/line/error token.\n\n"
                f"Environment: {env_type}\n"
                f"Parsed: {parsed_summary}\n"
                f"Raw:\n{output[:3000]}"
            )
            try:
                resp, _ = invoke_with_role_fallback(
                    role="Reviewer",
                    candidates=REVIEWER_MODEL_CANDIDATES,
                    messages=[HumanMessage(content=prompt)],
                    temperature=0,
                    max_tokens=180,
                )
                llm_summary = (getattr(resp, "content", "") or "").strip()
                if llm_summary:
                    condensed = llm_summary
            except Exception as exc:
                print(f"[Reviewer] [WARN] Reviewer summarizer unavailable: {exc}")

        history_payload = f"{condensed}\n\n[RAW]\n{output[:5000]}"
        history_additions = append_to_history("Reviewer", "Test Execution", history_payload)

        if success:
            print("[Reviewer] [OK] Code ran successfully.")
            return {
                "errors": "",
                "validation_status": "passed",
                "ast_validation_passed": True,
                "ast_error_detail": "",
                "history": history_additions,
            }

        print("[Reviewer] [FAIL] Code execution failed.")

        error_category = _categorize_error(output)
        error_line_numbers = _extract_line_numbers(output)

        print(f"[Reviewer] Error category: {error_category}, Lines: {error_line_numbers}")

        return {
            "errors": condensed,
            "validation_status": "failed",
            "error_category": error_category,
            "test_error_context": output[:500],
            "error_line_numbers": error_line_numbers,
            "ast_validation_passed": True,
            "ast_error_detail": "",
            "history": history_additions,
        }

    except Exception as exc:
        error_msg = str(exc)
        print(f"[Reviewer] [ERROR] Exception: {error_msg}")
        if "docker" in error_msg.lower() or "CreateFile" in error_msg:
            print("[Reviewer] Docker unavailable -- validation inconclusive.")
            return {
                "errors": "",
                "validation_status": "inconclusive",
                "ast_validation_passed": True,
                "ast_error_detail": "",
                "history": append_to_history(
                    "Reviewer",
                    "Skipped",
                    "Docker unavailable. Fix generated but not validated in sandbox.",
                ),
            }
        return {
            "errors": f"Reviewer error: {error_msg}",
            "validation_status": "failed",
            "ast_validation_passed": True,
            "ast_error_detail": "",
            "history": append_to_history("Reviewer", "Error", error_msg),
        }
