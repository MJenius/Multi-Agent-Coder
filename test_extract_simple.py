"""Test the diff extraction logic directly."""

# Inline the extraction function to avoid imports
def _extract_diff(llm_output: str) -> str:
    """Return the content between ```diff and ``` markers."""
    if not llm_output:
        return ""
    
    # Start marker
    start_marker = "```diff"
    start_pos = llm_output.find(start_marker)
    
    if start_pos == -1:
        # Fallback: maybe they just did ``` without diff?
        start_marker = "```"
        start_pos = llm_output.find(start_marker)

    if start_pos != -1:
        content_start = start_pos + len(start_marker)
        # Look for end marker
        end_pos = llm_output.find("```", content_start)
        if end_pos != -1:
            return llm_output[content_start:end_pos].strip()
        else:
            # Not closed? Take the rest of the output
            return llm_output[content_start:].strip()

    # No markers found -- return raw output (best effort)
    if "---" in llm_output and "+++" in llm_output:
        return llm_output.strip()
    
    return ""

# Test 1: Standard format with closing marker
test1 = """
1. Plan step 1
2. Plan step 2

```diff
diff --git a/file.cs b/file.cs
--- a/file.cs
+++ b/file.cs
@@ -10,5 +10,5 @@
 context line
-old line
+new line
 context line
```
"""

result1 = _extract_diff(test1)
print(f"Test 1 - Standard format:")
print(f"  Result length: {len(result1)}")
print(f"  Has ---: {'---' in result1}")
print(f"  Has +++: {'+++' in result1}")
assert "---" in result1 and "+++" in result1, "FAIL - missing markers"
assert "diff --git" in result1, "FAIL - missing git header"
assert "old line" in result1, "FAIL - missing content"
print("  PASS")

# Test 2: No preceding plan, just the diff
test2 = """```diff
diff --git a/AsciiQRCode.cs b/AsciiQRCode.cs
--- a/AsciiQRCode.cs
+++ b/AsciiQRCode.cs
@@ -140,7 +140,7 @@ public string GetGraphicSmall(bool drawQuietZones = true, bool invert = false)
     else if (current == WHITE)
```
"""

result2 = _extract_diff(test2)
print(f"\nTest 2 - Just diff block:")
print(f"  Result length: {len(result2)}")
assert "AsciiQRCode.cs" in result2, "FAIL - missing filename"
assert "---" in result2, "FAIL - missing ---"
print("  PASS")

# Test 3: Unclosed diff block (fallback to --- +++)
test3 = """Plan steps go here
Then we have the diff:

```diff
diff --git a/file.cs b/file.cs
--- a/file.cs
+++ b/file.cs
-old content
+new content
"""

result3 = _extract_diff(test3)
print(f"\nTest 3 - Unclosed block (no closing ```):")
print(f"  Result length: {len(result3)}")
if len(result3) > 0:
    print(f"  First 80 chars: {result3[:80]}")
    assert "---" in result3, f"FAIL - missing ---, got: {result3[:100]}"
    assert "+++" in result3, "FAIL - missing +++"
    print("  PASS")
else:
    print(f"  FAIL - got empty result for unclosed block")

# Test 4: Already-formatted diff without markers (fallback)
test4 = """Let me fix the issue.

diff --git a/AsciiQRCode.cs b/AsciiQRCode.cs
--- a/AsciiQRCode.cs
+++ b/AsciiQRCode.cs
@@ -1 +1 @@
-BLACK = true, WHITE = false
+BLACK = false, WHITE = true
"""

result4 = _extract_diff(test4)
print(f"\nTest 4 - No markers (fallback to --- +++):")
print(f"  Result length: {len(result4)}")
if len(result4) > 0:
    assert "diff --git" in result4, "FAIL - missing diff header"
    assert "---" in result4, "FAIL - missing ---"
    print("  PASS")
else:
    print(f"  FAIL - got empty result for fallback case")

print("\nALL TESTS PASSED")
