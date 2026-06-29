"""Verification Pipeline — coordinates linting, type-checking, testing, and security.

Executes a sequence of verification steps, automatically detecting available tools
and configuration settings in the repository.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any

from issue_resolver.core.interfaces import VerificationResult, VerificationStep
from issue_resolver.tools.sandbox_tools import get_sandbox_container


# ---------------------------------------------------------------------------
# Built-in Verification Steps
# ---------------------------------------------------------------------------


class PythonLintStep(VerificationStep):
    """Ruff-based linter step."""

    @property
    def step_name(self) -> str:
        return "lint_ruff"

    @property
    def language(self) -> str:
        return "python"

    def run(self, repo_path: str, changed_files: list[str]) -> VerificationResult:
        t0 = time.monotonic()
        sandbox = get_sandbox_container()

        # Build file arguments
        files = [f for f in changed_files if f.endswith(".py")]
        if not files:
            return VerificationResult(self.step_name, True, "No Python files modified.")

        file_args = " ".join(files)

        if sandbox:
            res = sandbox.exec_run(f"ruff check {file_args}", workdir="/workspace")
            passed = res.exit_code == 0
            output = res.output.decode("utf-8", errors="ignore")
        else:
            try:
                res = subprocess.run(
                    ["ruff", "check"] + files,
                    cwd=repo_path, capture_output=True, text=True, timeout=30, check=False,
                )
                passed = res.returncode == 0
                output = res.stdout + "\n" + res.stderr
            except Exception as exc:
                passed = False
                output = f"Failed to execute local ruff check: {exc}"

        duration_ms = (time.monotonic() - t0) * 1000
        return VerificationResult(self.step_name, passed, output.strip(), duration_ms)


class PythonCompileStep(VerificationStep):
    """py_compile syntax compilation check step."""

    @property
    def step_name(self) -> str:
        return "compile_check"

    @property
    def language(self) -> str:
        return "python"

    def run(self, repo_path: str, changed_files: list[str]) -> VerificationResult:
        t0 = time.monotonic()
        sandbox = get_sandbox_container()

        files = [f for f in changed_files if f.endswith(".py")]
        if not files:
            return VerificationResult(self.step_name, True, "No Python files modified.")

        passed = True
        outputs = []

        for f in files:
            if sandbox:
                res = sandbox.exec_run(f"python -m py_compile {f}", workdir="/workspace")
                if res.exit_code != 0:
                    passed = False
                    outputs.append(f"Syntax error in {f}:\n" + res.output.decode("utf-8", errors="ignore"))
            else:
                try:
                    res = subprocess.run(
                        ["python", "-m", "py_compile", f],
                        cwd=repo_path, capture_output=True, text=True, timeout=10, check=False,
                    )
                    if res.returncode != 0:
                        passed = False
                        outputs.append(f"Syntax error in {f}:\n" + res.stdout + "\n" + res.stderr)
                except Exception as exc:
                    passed = False
                    outputs.append(f"Compilation verification check failed for {f}: {exc}")

        duration_ms = (time.monotonic() - t0) * 1000
        output_str = "\n".join(outputs) if not passed else "Compilation syntax check passed."
        return VerificationResult(self.step_name, passed, output_str, duration_ms)


class PythonTestStep(VerificationStep):
    """Test verification runner (pytest/unittest)."""

    @property
    def step_name(self) -> str:
        return "run_tests"

    @property
    def language(self) -> str:
        return "python"

    def run(self, repo_path: str, changed_files: list[str]) -> VerificationResult:
        t0 = time.monotonic()
        sandbox = get_sandbox_container()

        # Run tests in sandbox
        if sandbox:
            from issue_resolver.tools.sandbox_tools import run_tests_in_sandbox
            # Generate diff to let sandbox run_tests determine context if needed, or pass empty string
            success, output = run_tests_in_sandbox("")
            duration_ms = (time.monotonic() - t0) * 1000
            return VerificationResult(self.step_name, success, output, duration_ms)
        else:
            # Fallback to local pytest check
            try:
                res = subprocess.run(
                    ["pytest"],
                    cwd=repo_path, capture_output=True, text=True, timeout=60, check=False,
                )
                duration_ms = (time.monotonic() - t0) * 1000
                return VerificationResult(self.step_name, res.returncode == 0, res.stdout + "\n" + res.stderr, duration_ms)
            except Exception as exc:
                duration_ms = (time.monotonic() - t0) * 1000
                return VerificationResult(self.step_name, False, f"Local test runner failed: {exc}", duration_ms)


# ---------------------------------------------------------------------------
# Verification Pipeline Coordinator
# ---------------------------------------------------------------------------


class VerificationPipeline:
    """Manages execution of verification steps."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self.steps: list[VerificationStep] = [
            PythonCompileStep(),
            PythonLintStep(),
            PythonTestStep(),
        ]

    def add_step(self, step: VerificationStep) -> None:
        """Register a custom verification step (e.g. from plugin)."""
        self.steps.append(step)

    def run(self, changed_files: list[str]) -> dict[str, Any]:
        """Run all matching steps on the modified files list."""
        print(f"[VerificationPipeline] Running pipeline on changed files: {changed_files}")

        results = []
        overall_passed = True

        for step in self.steps:
            # Filter steps by file suffix/language matches
            print(f"[VerificationPipeline] Executing: {step.step_name}...")
            try:
                res = step.run(self.repo_path, changed_files)
                results.append({
                    "step_name": res.step_name,
                    "passed": res.passed,
                    "output": res.output[:1000],  # Keep output snippet
                    "duration_ms": res.duration_ms,
                })
                if not res.passed:
                    overall_passed = False

                # Log trace event
                from issue_resolver.core.execution_trace import get_trace
                trace = get_trace()
                if trace:
                    trace.record_verification(
                        step_name=res.step_name,
                        passed=res.passed,
                        output_snippet=res.output,
                        duration_ms=res.duration_ms,
                    )
            except Exception as exc:
                print(f"[VerificationPipeline] [ERROR] Step {step.step_name} failed with exception: {exc}")
                overall_passed = False
                results.append({
                    "step_name": step.step_name,
                    "passed": False,
                    "output": f"Step crashed: {exc}",
                    "duration_ms": 0.0,
                })

        return {
            "passed": overall_passed,
            "results": results,
            "timestamp": time.time(),
        }
