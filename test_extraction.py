"""Test the diff extraction logic."""
import sys
sys.path.insert(0, ".")

from issue_resolver.nodes.coder import _extract_diff

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
assert "---" in result1 and "+++" in result1, "FAIL"
assert "diff --git" in result1, "FAIL"
assert "old line" in result1, "FAIL"
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
assert "AsciiQRCode.cs" in result2, "FAIL"
assert "---" in result2, "FAIL"
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
[TRUNCATED]
"""

result3 = _extract_diff(test3)
print(f"\nTest 3 - Unclosed block:")
print(f"  Result length: {len(result3)}")
assert "---" in result3, f"FAIL - expected--- in result, got: {result3[:100]}"
assert "+++" in result3, "FAIL"
print("  PASS")

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
print(f"\nTest 4 - No markers (fallback):")
print(f"  Result length: {len(result4)}")
assert "diff --git" in result4, "FAIL"
assert "---" in result4, "FAIL"
print("  PASS")

print("\nALL TESTS PASSED")
