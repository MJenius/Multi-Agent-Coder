"""Incremental Patcher Node — applies and validates patch blocks incrementally.

Applies SEARCH/REPLACE blocks one by one, running compilation and linting checks
after each block.  If a block fails compilation or linting, it is immediately
reverted, and the error is recorded before continuing to apply the next block.
"""

from __future__ import annotations

import os
import re
from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.utils.patch_engine import FuzzyPatchEngine
from issue_resolver.tools.sandbox_tools import get_sandbox_container


def _parse_blocks(llm_output: str) -> list[tuple[str, str]]:
    """Parse SEARCH/REPLACE blocks from output."""
    pattern = re.compile(
        r"^<<<<<<< SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>> REPLACE$",
        re.MULTILINE | re.DOTALL
    )
    return pattern.findall(llm_output)


def incremental_patcher_node(state: AgentState) -> dict:
    """Apply proposed_fix blocks incrementally and verify each step."""
    proposed_fix = state.get("proposed_fix", "")
    repo_path = state.get("repo_path", ".")
    structured_plan = state.get("structured_plan", {})
    files_to_edit = structured_plan.get("files_to_edit", [])

    if not proposed_fix:
        print("[IncrementalPatcher] No proposed fix found to apply.")
        return {"validation_status": "failed", "errors": "No patch proposed."}

    blocks = _parse_blocks(proposed_fix)
    if not blocks:
        print("[IncrementalPatcher] [WARNING] No SEARCH/REPLACE blocks found in proposed fix.")
        return {"validation_status": "failed", "errors": "Patch parse failed: No SEARCH/REPLACE blocks found."}

    print(f"[IncrementalPatcher] Applying {len(blocks)} patch blocks incrementally...")

    applied_blocks = []
    failed_blocks = []

    for idx, (search_text, replace_text) in enumerate(blocks):
        block_applied = False

        # Attempt to apply the block to each file in files_to_edit
        for file_rel in files_to_edit:
            file_abs = os.path.abspath(os.path.join(repo_path, file_rel))
            if not os.path.exists(file_abs):
                continue

            with open(file_abs, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            engine = FuzzyPatchEngine(content)
            res = engine.apply_block(search_text, replace_text)

            if res["success"]:
                # Write changes back
                with open(file_abs, "w", encoding="utf-8", newline="") as f:
                    f.write(engine.file_content)

                # Verify linting/compilation on the file immediately
                lint_failed = False
                lint_output = ""
                sandbox = get_sandbox_container()

                if sandbox:
                    res_lint = sandbox.exec_run(f"ruff check {file_rel}", workdir="/workspace")
                    if res_lint.exit_code != 0:
                        output = res_lint.output.decode("utf-8", errors="ignore")
                        if "not found" in output.lower() or "command not found" in output.lower() or res_lint.exit_code == 127:
                            res_lint = sandbox.exec_run(f"python -m py_compile {file_rel}", workdir="/workspace")
                            if res_lint.exit_code != 0:
                                lint_failed = True
                                lint_output = res_lint.output.decode("utf-8", errors="ignore")
                        else:
                            lint_failed = True
                            lint_output = output
                else:
                    import subprocess
                    try:
                        res_lint = subprocess.run(["ruff", "check", file_abs], capture_output=True, text=True)
                        if res_lint.returncode != 0:
                            lint_failed = True
                            lint_output = res_lint.stdout + "\n" + res_lint.stderr
                    except FileNotFoundError:
                        try:
                            res_lint = subprocess.run(["python", "-m", "py_compile", file_abs], capture_output=True, text=True)
                            if res_lint.returncode != 0:
                                lint_failed = True
                                lint_output = res_lint.stdout + "\n" + res_lint.stderr
                        except Exception:
                            pass

                if lint_failed:
                    print(f"[IncrementalPatcher] Block {idx + 1} caused compilation/lint failure. Reverting block.")
                    # Revert file changes
                    if sandbox:
                        sandbox.exec_run(f"git checkout -- {file_rel}", workdir="/workspace")
                    else:
                        subprocess.run(["git", "checkout", "--", file_abs], cwd=repo_path)
                    failed_blocks.append({
                        "index": idx,
                        "file": file_rel,
                        "reason": f"Linter/Compilation validation failed: {lint_output.strip()}"
                    })
                else:
                    block_applied = True
                    applied_blocks.append({
                        "index": idx,
                        "file": file_rel
                    })
                    break

        if not block_applied and not any(fb["index"] == idx for fb in failed_blocks):
            # Reached here and no file accepted this block
            print(f"[IncrementalPatcher] Block {idx + 1} search content could not be matched in any source files.")
            failed_blocks.append({
                "index": idx,
                "file": "unknown",
                "reason": "Search block content not matched in target files."
            })

    # Summarize results
    errors_summary = ""
    if failed_blocks:
        errors_summary = "\n".join(
            f"Block {fb['index'] + 1} on {fb['file']} failed: {fb['reason']}"
            for fb in failed_blocks
        )

    if not applied_blocks:
        validation_status = "failed"
        status_msg = f"Failed to apply all {len(blocks)} blocks. Errors:\n{errors_summary}"
    elif failed_blocks:
        validation_status = "applied_with_errors"
        status_msg = f"Applied {len(applied_blocks)}/{len(blocks)} blocks successfully with errors:\n{errors_summary}"
    else:
        validation_status = "applied"
        status_msg = f"Successfully applied all {len(blocks)} blocks."

    print(f"[IncrementalPatcher] Result: {status_msg}")

    # Commit execution trace
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "patch_applied_incrementally",
            "IncrementalPatcher",
            f"Applied {len(applied_blocks)}/{len(blocks)} blocks",
            details={
                "validation_status": validation_status,
                "applied_blocks": applied_blocks,
                "failed_blocks": failed_blocks,
            },
        )

    return {
        "validation_status": validation_status,
        "errors": errors_summary,
        "history": append_to_history(
            "IncrementalPatcher",
            "Apply Patch Status",
            status_msg[:300],
        ),
    }
