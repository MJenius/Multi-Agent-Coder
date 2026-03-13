"""Unit tests for sandbox error parsing utilities."""

from __future__ import annotations

from issue_resolver.tools.sandbox_tools import (
    format_parsed_error_summary,
    parse_dotnet_error_trace,
    parse_eslint_error_trace,
    parse_node_error_trace,
)


def test_parse_dotnet_error_trace_extracts_code_and_line():
    output = "src/Program.cs(45,13): error CS1002: ; expected [src/App.csproj]"
    parsed = parse_dotnet_error_trace(output)

    assert parsed["count"] == 1
    assert parsed["primary"]["code"] == "CS1002"
    assert parsed["primary"]["line"] == 45
    assert "expected" in parsed["primary"]["message"]


def test_parse_node_error_trace_extracts_culprit_file():
    output = "TypeError: boom\n    at run (src/server/hang.js:87:22)\n    at processTicksAndRejections (node:internal/process/task_queues:95:5)"
    parsed = parse_node_error_trace(output)

    assert parsed["culprit"] is not None
    assert parsed["culprit"]["file"].endswith("src/server/hang.js")
    assert parsed["culprit"]["line"] == 87


def test_parse_eslint_error_trace_extracts_rule_and_location():
    output = "src/ui/button.tsx:45:13: Unexpected console statement  no-console"
    parsed = parse_eslint_error_trace(output)

    assert parsed["count"] == 1
    assert parsed["primary"]["rule"] == "no-console"
    assert parsed["primary"]["line"] == 45


def test_format_parsed_error_summary_for_dotnet():
    output = "src/Util.cs(12,7): error CS0103: The name 'missing' does not exist in the current context"
    summary = format_parsed_error_summary("dotnet", output)

    assert "CS0103" in summary
    assert "src/Util.cs:12:7" in summary


def test_format_parsed_error_summary_for_eslint():
    output = "src/ui/button.tsx:45:13: Unexpected console statement  no-console"
    summary = format_parsed_error_summary("nodejs", output)

    assert "ESLint error" in summary
    assert "no-console" in summary
