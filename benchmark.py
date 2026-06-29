"""Multi-Agent-Coder Benchmark CLI.

Runs evaluation suites on issues and collects metrics such as pass@1, pass@N,
latencies, api costs, and regression failures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# Mock issues for running benchmarks
_MOCK_ISSUES = [
    {
        "id": "issue-001",
        "title": "Fix division by zero in calculate_average",
        "issue": "ZeroDivisionError: division by zero in utils.py inside calculate_average when total is empty.",
    },
    {
        "id": "issue-002",
        "title": "Add default parameters to User class initialization",
        "issue": "TypeError: __init__() missing 1 required positional argument: 'role' in models.py when loading session.",
    }
]


def run_benchmark(
    suite_path: str | None = None,
    output_path: str = "benchmark_report.json",
    limit: int = 10,
) -> None:
    print("=" * 60)
    print("           Multi-Agent-Coder v2 Benchmark Suite")
    print("=" * 60)

    issues = _MOCK_ISSUES
    if suite_path:
        p = Path(suite_path)
        if p.is_file():
            try:
                issues = json.loads(p.read_text(encoding="utf-8"))
                print(f"Loaded {len(issues)} issues from benchmark suite '{suite_path}'")
            except Exception as exc:
                print(f"Failed to load suite: {exc}. Using mock issues.")

    issues = issues[:limit]

    results = []
    total_time = 0.0
    passed_count = 0
    total_tokens = 0
    total_cost = 0.0

    from issue_resolver.graph import app
    from issue_resolver.core.execution_trace import start_trace

    for idx, item in enumerate(issues, 1):
        print(f"\n[{idx}/{len(issues)}] Running: {item.get('title', 'Untitled')}")
        
        # Initialize execution trace
        trace = start_trace(run_id=item["id"])

        state_input = {
            "issue": item["issue"],
            "repo_path": "./sandbox_workspace",
            "iterations": 0,
            "coder_retry_budget": 3,
            "history": [],
        }

        t0 = time.monotonic()
        try:
            res = app.invoke(state_input)
            duration = time.monotonic() - t0
            passed = res.get("validation_status") == "passed"
        except Exception as exc:
            duration = time.monotonic() - t0
            passed = False
            res = {"errors": str(exc)}
            print(f"Execution failed with exception: {exc}")

        # Finalize trace and metrics
        trace_data = trace.finalize()
        tokens = trace_data["summary"]["total_tokens"]
        cost = tokens * 0.000002  # Mock key calculation logic ($2 per M tokens)

        total_time += duration
        total_tokens += tokens
        total_cost += cost

        if passed:
            passed_count += 1

        results.append({
            "id": item["id"],
            "title": item.get("title"),
            "passed": passed,
            "duration_s": duration,
            "tokens_used": tokens,
            "cost_est": cost,
            "errors": res.get("errors", ""),
        })

        status = "PASSED" if passed else "FAILED"
        print(f"Result: {status} in {duration:.1f}s (Tokens: {tokens}, Cost: ${cost:.4f})")

    # Generate aggregate metrics report
    success_rate = (passed_count / len(issues)) * 100 if issues else 0.0
    avg_latency = total_time / len(issues) if issues else 0.0

    report = {
        "benchmark_summary": {
            "total_issues": len(issues),
            "passed": passed_count,
            "failed": len(issues) - passed_count,
            "success_rate": success_rate,
            "average_latency_s": avg_latency,
            "total_tokens_consumed": total_tokens,
            "total_estimated_cost_usd": total_cost,
            "wall_clock_time_s": total_time,
        },
        "results": results,
    }

    # Write report file
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("                     BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Success Rate:      {success_rate:.1f}% ({passed_count}/{len(issues)})")
    print(f"Average Latency:   {avg_latency:.1f} seconds")
    print(f"Total Tokens:      {total_tokens}")
    print(f"Total API Cost:    ${total_cost:.4f}")
    print(f"Report Written to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent-Coder v2 Benchmark Suite CLI")
    parser.add_argument("--suite", type=str, help="Path to benchmark issues json file")
    parser.add_argument("--output", type=str, default="benchmark_report.json", help="Path to output report json file")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of issues to run")

    args = parser.parse_args()
    run_benchmark(suite_path=args.suite, output_path=args.output, limit=args.limit)
