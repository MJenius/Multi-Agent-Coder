"""Tests for the coder node's parsing logic."""

from __future__ import annotations

from issue_resolver.nodes.coder import (
    _parse_unified_diff,
    _parse_json_patch,
    _extract_issue_identifiers,
)


def test_parse_unified_diff():
    diff_text = """
Some explanation here.
```diff
--- a/stripe/_encode.py
+++ b/stripe/_encode.py
@@ -160,3 +160,5 @@
-        elif isinstance(value, list) or isinstance(value, tuple):
+        elif isinstance(value, (list, tuple)):
+            if len(value) == 0:
+                yield (key, "")
```
"""
    result = _parse_unified_diff(diff_text)
    assert "--- a/stripe/_encode.py" in result
    assert "+++" in result


def test_parse_json_patch():
    file_info = {"stripe/_encode.py": "original content line 1\noriginal content line 2"}
    known_paths = ["stripe/_encode.py"]
    patch_text = """
```json
{
  "file": "stripe/_encode.py",
  "hunks": [
    {
      "start_line": 2,
      "delete_lines": ["original content line 2"],
      "insert_lines": ["new content line 2"]
    }
  ]
}
```
"""
    diff = _parse_json_patch(patch_text, file_info, known_paths)
    assert diff != ""
    assert "stripe/_encode.py" in diff
    assert "+new content line 2" in diff


def test_extract_issue_identifiers():
    issue_text = "Setting a List Field to `[]` on `blocked_categories` spending_controls doesn't clear it."
    ids = _extract_issue_identifiers(issue_text)
    assert "[]" in ids["high"]
    assert "blocked_categories" in ids["medium"]
