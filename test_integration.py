"""Integration test: Full researcher node with QRCoder hint - should complete in <10s."""
import time
from issue_resolver.nodes.researcher import researcher_node

ISSUE = (
    "Title: ASCII 'small' renderer prints inverted by default\n\n"
    "Body: By default, the ASCII 'small' renderer prints inverted compared to "
    "the ASCII standard renderer or any other renderer.\n\n"
    "CRITICAL INSTRUCTION: The repository code is located strictly inside the "
    "'./sandbox_workspace' directory.\n\n"
    "🎯 HINT: QRCoder/AsciiQRCode.cs"
)

state = {
    "issue": ISSUE,
    "repo_path": "./sandbox_workspace",
    "file_context": [],
    "proposed_fix": "",
    "errors": "",
    "next_step": "",
    "iterations": 0,
    "history": [],
}

print("=" * 60)
print("Integration Test: Researcher Node with Hints")
print("=" * 60)

t0 = time.time()
result = researcher_node(state)
elapsed = time.time() - t0

print(f"\n{'=' * 60}")
print(f"RESULT:")
print(f"  Time: {elapsed:.1f}s")
print(f"  Snippets: {len(result.get('file_context', []))}")
print(f"  History entries: {len(result.get('history', []))}")

# Show first snippet summary
for i, snippet in enumerate(result.get("file_context", [])):
    first_line = snippet.split("\n")[0]
    line_count = snippet.count("\n")
    print(f"  Snippet {i+1}: {first_line} ({line_count} lines)")

# Assertions
assert len(result["file_context"]) >= 1, "Should have at least 1 snippet"
assert elapsed < 10.0, f"Should complete in <10s, took {elapsed:.1f}s"
assert any("AsciiQRCode" in s for s in result["file_context"]), "Should contain AsciiQRCode content"

print(f"\nPASS - Researcher completed in {elapsed:.1f}s with {len(result['file_context'])} snippet(s)")
