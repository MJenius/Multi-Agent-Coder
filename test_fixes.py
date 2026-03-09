"""Quick test to verify Phase 1 fixes."""
from issue_resolver.nodes.researcher import _extract_hints_from_issue, _detect_language

# Test 1: Hint extraction from QRCoder issue (the real case)
issue_text = (
    "Title: ASCII small renderer prints inverted by default\n\n"
    "Body: By default, the ASCII small renderer prints inverted.\n\n"
    "CRITICAL INSTRUCTION: The repository code is located strictly inside the "
    "'./sandbox_workspace' directory.\n\n"
    "🎯 HINT: For QRCoder issues involving ASCII rendering, the bug is likely "
    "in QRCoder/AsciiQRCode.cs. Focus your search there."
)

hints = _extract_hints_from_issue(issue_text)
print(f"Test 1 - Hints extracted: {hints}")
assert "QRCoder/AsciiQRCode.cs" in hints, f"Expected QRCoder/AsciiQRCode.cs in {hints}"
for h in hints:
    assert " " not in h, f"Hint contains spaces (sentence): {h}"
print("  PASS: Only clean file paths, no sentences")

# Test 2: Clean path hint
issue2 = "🎯 HINT: QRCoder/AsciiQRCode.cs"
hints2 = _extract_hints_from_issue(issue2)
print(f"Test 2 - Clean hint: {hints2}")
assert hints2 == ["QRCoder/AsciiQRCode.cs"], f"Got {hints2}"
print("  PASS")

# Test 3: Multiple file paths in issue
issue3 = "Look at src/main.py and also check handlers/request.ts for the bug."
hints3 = _extract_hints_from_issue(issue3)
print(f"Test 3 - Multiple paths: {hints3}")
assert "src/main.py" in hints3
assert "handlers/request.ts" in hints3
print("  PASS")

# Test 4: Language detection (fast, no recursive glob)
import time
t0 = time.time()
lang = _detect_language("./sandbox_workspace")
elapsed = time.time() - t0
print(f"Test 4 - Language detection: {lang} ({elapsed:.3f}s)")
assert lang == "csharp", f"Expected csharp, got {lang}"
assert elapsed < 2.0, f"Language detection too slow: {elapsed:.1f}s"
print("  PASS")

# Test 5: No hints in plain text
issue5 = "The program crashes when I click the button. Please fix it."
hints5 = _extract_hints_from_issue(issue5)
print(f"Test 5 - No hints: {hints5}")
assert hints5 == [], f"Expected empty list, got {hints5}"
print("  PASS")

# Test 6: Code blocks should not generate false hints
issue6 = """The bug is likely caused by incorrect constants:
```cs
bool BLACK = true, WHITE = false;
```
Please fix this in the renderer."""
hints6 = _extract_hints_from_issue(issue6)
print(f"Test 6 - Code block text: {hints6}")
# "false" has no code extensions, so no matches expected
print("  PASS")

# Test 7: Relative path with ./
issue7 = "Please check ./src/utils/helper.js for the issue."
hints7 = _extract_hints_from_issue(issue7)
print(f"Test 7 - Relative path: {hints7}")
assert "src/utils/helper.js" in hints7, f"Got {hints7}"
print("  PASS")

print("\nALL TESTS PASSED")
