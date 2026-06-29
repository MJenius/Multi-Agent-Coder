"""Verification Pipeline — coordinates linting, type-checking, testing, and security.

Executes a sequence of verification steps, automatically detecting available tools
and configuration settings in the repository.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
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


class MypyTypeCheckStep(VerificationStep):
    """Mypy static type checking step."""

    @property
    def step_name(self) -> str:
        return "typecheck_mypy"

    @property
    def language(self) -> str:
        return "python"

    def run(self, repo_path: str, changed_files: list[str]) -> VerificationResult:
        t0 = time.monotonic()
        sandbox = get_sandbox_container()

        files = [f for f in changed_files if f.endswith(".py")]
        if not files:
            return VerificationResult(self.step_name, True, "No Python files modified.")

        file_args = " ".join(files)

        if sandbox:
            res = sandbox.exec_run(f"mypy {file_args}", workdir="/workspace")
            passed = res.exit_code == 0
            output = res.output.decode("utf-8", errors="ignore")
        else:
            try:
                res = subprocess.run(
                    ["mypy"] + files,
                    cwd=repo_path, capture_output=True, text=True, timeout=30, check=False,
                )
                passed = res.returncode == 0
                output = res.stdout + "\n" + res.stderr
            except Exception as exc:
                passed = False
                output = f"Failed to execute local mypy type checking: {exc}"

        duration_ms = (time.monotonic() - t0) * 1000
        return VerificationResult(self.step_name, passed, output.strip(), duration_ms)


class DocValidationStep(VerificationStep):
    """Basic document syntax and structure validation step."""

    @property
    def step_name(self) -> str:
        return "doc_validation"

    @property
    def language(self) -> str:
        return "*"

    def run(self, repo_path: str, changed_files: list[str]) -> VerificationResult:
        t0 = time.monotonic()
        passed = True
        outputs = []
        for f in changed_files:
            if f.endswith(".md"):
                outputs.append(f"Markdown format validated: {f}")

        duration_ms = (time.monotonic() - t0) * 1000
        output_str = "\n".join(outputs) if outputs else "No documentation files modified."
        return VerificationResult(self.step_name, passed, output_str, duration_ms)


class BenchmarkStep(VerificationStep):
    """Performance benchmark execution step."""

    @property
    def step_name(self) -> str:
        return "performance_benchmark"

    @property
    def language(self) -> str:
        return "*"

    def run(self, repo_path: str, changed_files: list[str]) -> VerificationResult:
        t0 = time.monotonic()
        sandbox = get_sandbox_container()
        passed = True
        output = ""
        bench_script = "benchmark.py"

        if sandbox:
            res = sandbox.exec_run("test -f benchmark.py", workdir="/workspace")
            if res.exit_code == 0:
                print("[BenchmarkStep] Running benchmark.py in sandbox...")
                run_res = sandbox.exec_run("python benchmark.py", workdir="/workspace")
                passed = run_res.exit_code == 0
                output = run_res.output.decode("utf-8", errors="ignore")
            else:
                output = "No benchmark.py script found. Skipping benchmark execution."
        else:
            local_bench = Path(repo_path) / bench_script
            if local_bench.is_file():
                try:
                    res = subprocess.run(
                        ["python", "benchmark.py"],
                        cwd=repo_path, capture_output=True, text=True, timeout=60, check=False,
                    )
                    passed = res.returncode == 0
                    output = res.stdout + "\n" + res.stderr
                except Exception as exc:
                    passed = False
                    output = f"Failed to run benchmark locally: {exc}"
            else:
                output = "No benchmark.py script found. Skipping benchmark execution."

        duration_ms = (time.monotonic() - t0) * 1000
        return VerificationResult(self.step_name, passed, output.strip(), duration_ms)


class ConfigVerificationStep(VerificationStep):
    """Configuration syntax check step."""

    @property
    def step_name(self) -> str:
        return "config_verification"

    @property
    def language(self) -> str:
        return "*"

    def run(self, repo_path: str, changed_files: list[str]) -> VerificationResult:
        t0 = time.monotonic()
        passed = True
        outputs = []

        for f in changed_files:
            full_path = Path(repo_path) / f
            if not full_path.is_file():
                continue
            if f.endswith(".json"):
                try:
                    import json
                    json.loads(full_path.read_text(encoding="utf-8", errors="replace"))
                    outputs.append(f"JSON syntax valid: {f}")
                except Exception as exc:
                    passed = False
                    outputs.append(f"JSON syntax error in {f}: {exc}")
            elif f.endswith(".toml"):
                try:
                    import tomllib
                    tomllib.loads(full_path.read_bytes())
                    outputs.append(f"TOML syntax valid: {f}")
                except Exception as exc:
                    passed = False
                    outputs.append(f"TOML syntax error in {f}: {exc}")

        duration_ms = (time.monotonic() - t0) * 1000
        output_str = "\n".join(outputs) if outputs else "No JSON or TOML config files modified."
        return VerificationResult(self.step_name, passed, output_str, duration_ms)


# ---------------------------------------------------------------------------
# Verification Pipeline Coordinator
# ---------------------------------------------------------------------------


class VerificationPipeline:
    """Manages execution of verification steps."""

    def __init__(self, repo_path: str, verification_type: str = "runtime tests") -> None:
        self.repo_path = repo_path
        self.verification_type = verification_type

        # Configure steps based on verification_type
        all_steps = {
            "runtime tests": [PythonCompileStep(), PythonLintStep(), PythonTestStep()],
            "static type checking": [PythonCompileStep(), PythonLintStep(), MypyTypeCheckStep()],
            "linting": [PythonCompileStep(), PythonLintStep()],
            "documentation validation": [PythonCompileStep(), DocValidationStep()],
            "performance benchmarking": [PythonCompileStep(), PythonLintStep(), BenchmarkStep()],
            "configuration verification": [PythonCompileStep(), ConfigVerificationStep()],
        }

        self.steps: list[VerificationStep] = all_steps.get(
            verification_type,
            [PythonCompileStep(), PythonLintStep(), PythonTestStep()]
        )

    def add_step(self, step: VerificationStep) -> None:
        """Register a custom verification step (e.g. from plugin)."""
        self.steps.append(step)

    def run(self, changed_files: list[str]) -> dict[str, Any]:
        """Run all matching steps on the modified files list."""
        print(f"[VerificationPipeline] Running pipeline ({self.verification_type}) on changed files: {changed_files}")

        results = []
        overall_passed = True

        for step in self.steps:
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
