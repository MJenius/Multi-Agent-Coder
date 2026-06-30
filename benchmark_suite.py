"""Continuous Evaluation Benchmark Suite.

Runs the multi-agent issue resolver machine against a fixed set of real GitHub
issues to track pass@1, localization accuracy, verification success, retries,
and solve rate, reporting objective comparative metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import State Machine and Graph
from issue_resolver.graph import app
from issue_resolver.state import AgentState


def expected_to_cat(expected_fix_type: str) -> str:
    """Map expected fix type to issue category format."""
    mapping = {
        "bug_fix": "Bug",
        "typing": "Typing",
        "configuration": "Configuration",
        "refactor": "Refactor",
        "testing": "Testing",
        "documentation": "Documentation",
        "performance": "Performance",
        "security": "Security",
        "feature": "Feature",
        "api_change": "API Change",
        "dependency_update": "Dependency Update",
    }
    return mapping.get(expected_fix_type.lower(), "Bug")


@dataclass
class BenchmarkIssue:
    repo_url: str
    issue_id: str
    issue_text: str
    expected_files: list[str]
    expected_symbols: list[str]
    expected_fix_type: str
    difficulty: str
    language: str


@dataclass
class BenchmarkResult:
    issue_id: str
    difficulty: str
    # Localization metrics
    localization_precision: float = 0.0
    localization_recall: float = 0.0
    symbols_precision: float = 0.0
    symbols_recall: float = 0.0
    # Execution metrics
    pass_at_1: bool = False
    verification_success: bool = False
    retry_count: int = 0
    solved: bool = False
    files_modified: int = 0
    # Resource metrics
    total_runtime_ms: float = 0.0
    llm_calls: int = 0
    total_tokens: int = 0
    
    # Extended reliability metrics
    classification_correct: bool = False
    localization_confidence: float = 0.0
    patch_applied_successfully: bool = False


class BenchmarkSuite:
    """Benchmark runner that computes comparative statistics."""

    def __init__(self, issues: list[BenchmarkIssue]) -> None:
        self.issues = issues
        self.results: list[BenchmarkResult] = []

    def run_issue(self, issue: BenchmarkIssue, sandbox_dir: str) -> BenchmarkResult:
        """Run the LangGraph machine on a single benchmark issue and evaluate results."""
        print(f"\n[Benchmark] Running issue {issue.issue_id} ({issue.difficulty})...")
        
        t0 = time.monotonic()
        
        # Initialize LangGraph state
        initial_state = {
            "issue": issue.issue_text,
            "repo_path": sandbox_dir,
            "iterations": 0,
            "is_resolved": False,
            "file_context": [],
            "history": [],
            "coder_retry_budget": 3,
        }

        # Clear active trace and context
        from issue_resolver.core.execution_trace import start_trace
        trace = start_trace(run_id=f"bench_{issue.issue_id}")

        # Run State Machine
        final_state = app.invoke(initial_state)
        
        duration_ms = (time.monotonic() - t0) * 1000
        
        # Calculate localization metrics
        localization = final_state.get("localization_result", {})
        primary_files = [f["path"] for f in localization.get("primary_files", [])]
        found_symbols = [s["name"] for s in localization.get("symbols", [])]

        # File precision/recall
        expected_f = set(issue.expected_files)
        found_f = set(primary_files[:len(expected_f) + 1]) # look at top files
        matched_f = expected_f & found_f
        
        loc_precision = len(matched_f) / max(len(found_f), 1)
        loc_recall = len(matched_f) / max(len(expected_f), 1)

        # Symbol precision/recall
        expected_s = set(issue.expected_symbols)
        found_s = set(found_symbols[:len(expected_s) + 2])
        matched_s = expected_s & found_s
        
        sym_precision = len(matched_s) / max(len(found_s), 1)
        sym_recall = len(matched_s) / max(len(expected_s), 1)

        # Execution metrics
        verification = final_state.get("verification_report", {})
        pass_at_1 = final_state.get("validation_status") == "applied"
        verification_success = verification.get("passed", False)
        retry_count = final_state.get("iterations", 1) - 1
        solved = final_state.get("is_resolved", False)
        
        # Count files modified
        files_modified = 0
        if solved and final_state.get("proposed_fix"):
            files_modified = len(final_state.get("structured_plan", {}).get("files_to_edit", []))

        # Resource usage
        llm_calls = sum(1 for e in trace.events if e.event_type in ("plan_generated", "candidates_generated", "debugging_completed"))
        total_tokens = trace.total_tokens()

        # Calculate extended metrics
        classified_cat = final_state.get("issue_category", "Bug")
        expected_cat = expected_to_cat(issue.expected_fix_type)
        classification_correct = (classified_cat.lower() == expected_cat.lower())
        
        loc_confidence = final_state.get("localization_confidence", 0.0)
        
        validation_status = final_state.get("validation_status", "")
        patch_applied_successfully = validation_status in ("applied", "applied_with_errors", "passed")

        result = BenchmarkResult(
            issue_id=issue.issue_id,
            difficulty=issue.difficulty,
            localization_precision=loc_precision,
            localization_recall=loc_recall,
            symbols_precision=sym_precision,
            symbols_recall=sym_recall,
            pass_at_1=pass_at_1,
            verification_success=verification_success,
            retry_count=retry_count,
            solved=solved,
            files_modified=files_modified,
            total_runtime_ms=duration_ms,
            llm_calls=llm_calls,
            total_tokens=total_tokens,
            classification_correct=classification_correct,
            localization_confidence=loc_confidence,
            patch_applied_successfully=patch_applied_successfully,
        )
        
        self.results.append(result)
        return result

    def run_all(self, sandbox_dir: str) -> dict[str, Any]:
        """Execute all loaded issues and aggregate results."""
        self.results.clear()
        
        for issue in self.issues:
            try:
                self.run_issue(issue, sandbox_dir)
            except Exception as e:
                print(f"[Benchmark] [ERROR] Issue {issue.issue_id} crashed: {e}")
                self.results.append(BenchmarkResult(
                    issue_id=issue.issue_id,
                    difficulty=issue.difficulty,
                ))

        return self.get_summary()

    def get_summary(self) -> dict[str, Any]:
        """Aggregate stats over all completed run results."""
        n = len(self.results)
        if n == 0:
            return {}

        solved_count = sum(1 for r in self.results if r.solved)
        pass1_count = sum(1 for r in self.results if r.pass_at_1)
        verif_count = sum(1 for r in self.results if r.verification_success)
        class_correct_count = sum(1 for r in self.results if r.classification_correct)
        patch_applied_count = sum(1 for r in self.results if r.patch_applied_successfully)

        return {
            "total_issues": n,
            "solve_rate": solved_count / n,
            "pass_at_1": pass1_count / n,
            "verification_success_rate": verif_count / n,
            "avg_localization_precision": sum(r.localization_precision for r in self.results) / n,
            "avg_localization_recall": sum(r.localization_recall for r in self.results) / n,
            "avg_symbols_precision": sum(r.symbols_precision for r in self.results) / n,
            "avg_symbols_recall": sum(r.symbols_recall for r in self.results) / n,
            "avg_retries": sum(r.retry_count for r in self.results) / n,
            "avg_runtime_sec": sum(r.total_runtime_ms for r in self.results) / (n * 1000),
            "avg_llm_calls": sum(r.llm_calls for r in self.results) / n,
            "avg_tokens": sum(r.total_tokens for r in self.results) / n,
            
            # New metrics
            "classification_accuracy": class_correct_count / n,
            "avg_localization_confidence": sum(r.localization_confidence for r in self.results) / n,
            "patch_application_success_rate": patch_applied_count / n,
        }

    def compare(self, baseline: dict[str, Any], current: dict[str, Any]) -> str:
        """Generate a comparison report showing delta changes."""
        lines = [
            "=" * 60,
            "                    BENCHMARK COMPARISON REPORT                 ",
            "=" * 60,
        ]
        
        metrics = [
            ("Solve Rate", "solve_rate", "{:.2%}"),
            ("Pass @ 1 Rate", "pass_at_1", "{:.2%}"),
            ("Verification Success", "verification_success_rate", "{:.2%}"),
            ("Classification Accuracy", "classification_accuracy", "{:.2%}"),
            ("Avg Loc Confidence", "avg_localization_confidence", "{:.2%}"),
            ("Patch Apply Success", "patch_application_success_rate", "{:.2%}"),
            ("Loc Precision", "avg_localization_precision", "{:.2%}"),
            ("Loc Recall", "avg_localization_recall", "{:.2%}"),
            ("Symbols Precision", "avg_symbols_precision", "{:.2%}"),
            ("Symbols Recall", "avg_symbols_recall", "{:.2%}"),
            ("Avg Retries", "avg_retries", "{:.2f}"),
            ("Avg Runtime (sec)", "avg_runtime_sec", "{:.2f}"),
            ("Avg LLM Calls", "avg_llm_calls", "{:.2f}"),
            ("Avg Tokens Used", "avg_tokens", "{:.0f}"),
        ]
        
        for label, key, fmt in metrics:
            base_val = baseline.get(key, 0.0)
            curr_val = current.get(key, 0.0)
            diff = curr_val - base_val
            
            fmt_base = fmt.format(base_val)
            fmt_curr = fmt.format(curr_val)
            
            sign = "+" if diff > 0 else ""
            fmt_diff = fmt.format(diff) if "%" in fmt else f"{sign}{diff:.2f}"
            if "%" in fmt:
                fmt_diff = f"{sign}{diff*100:.1f}%"
                
            lines.append(f"{label:<25} Baseline: {fmt_base:<10} Current: {fmt_curr:<10} Delta: {fmt_diff}")
            
        lines.append("=" * 60)
        return "\n".join(lines)


def load_issues(path: str) -> list[BenchmarkIssue]:
    """Load benchmark issues from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        BenchmarkIssue(
            repo_url=item["repo_url"],
            issue_id=item["issue_id"],
            issue_text=item["issue_text"],
            expected_files=item["expected_files"],
            expected_symbols=item["expected_symbols"],
            expected_fix_type=item["expected_fix_type"],
            difficulty=item["difficulty"],
            language=item["language"],
        )
        for item in data
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Coder Benchmark Suite")
    parser.add_argument("--issues", default="benchmark_issues.json", help="Path to issues file")
    parser.add_argument("--sandbox", default="sandbox_workspace", help="Path to workspace directory")
    parser.add_argument("--baseline", help="Baseline performance result json")
    parser.add_argument("--compare-only", action="store_true", help="Compare baseline and current without running")
    parser.add_argument("--baseline-compare", help="JSON to compare current run against")
    parser.add_argument("--output", default="benchmark_report.json", help="Output report json file")
    
    args = parser.parse_args()
    
    if args.compare_only:
        if not args.baseline or not args.output:
            print("Must specify --baseline and --output (current results) to compare.")
            exit(1)
        with open(args.baseline, "r") as f:
            base = json.load(f)
        with open(args.output, "r") as f:
            curr = json.load(f)
        suite = BenchmarkSuite([])
        print(suite.compare(base, curr))
        exit(0)

    # Run Benchmark
    issues = load_issues(args.issues)
    suite = BenchmarkSuite(issues)
    
    print(f"Loaded {len(issues)} benchmark issues.")
    summary = suite.run_all(args.sandbox)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {args.output}")

    # Output run summary
    print("\nSummary Results:")
    for k, v in summary.items():
        if "rate" in k or "@" in k or "precision" in k or "recall" in k:
            print(f"  {k:<30} {v:.2%}")
        else:
            print(f"  {k:<30} {v:.2f}")

    # Perform comparison
    if args.baseline_compare and os.path.exists(args.baseline_compare):
        with open(args.baseline_compare, "r") as f:
            base = json.load(f)
        print("\nComparison against baseline:")
        print(suite.compare(base, summary))
