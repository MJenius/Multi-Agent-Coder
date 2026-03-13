"""Tests for the coder node's parsing and fix-application logic.

Uses real LLM output samples (markdown with headers and code blocks)
to verify the parser handles them correctly.
"""

from typing import cast

from issue_resolver.nodes.coder import (
    _parse_fix,
    _strip_markdown,
    _find_and_replace,
    _extract_plan,
  _attempt_fix,
  _extract_issue_identifiers,
  coder_node,
)
from issue_resolver.state import AgentState


# ── Test data: actual LLM output samples ─────────────────────────────────

XML_OUTPUT = """\
<plan>Fix regex for uz-UZ to include carrier codes 33 and 55.</plan>
<fix>
FILE: src/lib/isMobilePhone.js
SEARCH:
  'uz-UZ': /^(\\+?998)?(6[125-79]|7[1-69]|88|9\\d)\\d{7}$/,
REPLACE:
  'uz-UZ': /^(\\+?998)?(33|55|6[125-79]|7[1-69]|88|9\\d)\\d{7}$/,
</fix>
"""

MARKDOWN_SIMPLE = """\
## Fix for isMobilePhone function

### FILE: src/lib/isMobilePhone.js

### SEARCH:
```javascript
  'uz-UZ': /^(\\+?998)?(6[125-79]|7[1-69]|88|9\\d)\\d{7}$/,
```

### REPLACE:
```javascript
  'uz-UZ': /^(\\+?998)?(33|55|6[125-79]|7[1-69]|88|9\\d)\\d{7}$/,
```
"""

MARKDOWN_LARGE_REFACTOR = """\
## Fix for isMobilePhone function

### FILE: src/lib/isMobilePhone.js

### SEARCH:
```javascript
if (locale in phones) {
  return phones[locale].test(str);
} else if (!locale || locale === 'any') {
  for (const key in phones) {
    if (phones.hasOwnProperty(key)) {
      const phone = phones[key];
      if (phone.test(str)) {
        return true;
      }
    }
  }
  return false;
}
```

### REPLACE:
```javascript
if (locale in phones) {
  return phones[locale].test(str);
} else if (!locale || locale === 'any') {
  for (const key in phones) {
    if (phones.hasOwnProperty(key)) {
      const phone = phones[key];
      if (phone.test(str)) {
        return true;
      }
    }
  }
  return false;
} else {
  throw new Error('Invalid locale');
}
```
"""

SOURCE_FILE = """\
import assertString from './util/assertString';

const phones = {
  'uz-UZ': /^(\\+?998)?(6[125-79]|7[1-69]|88|9\\d)\\d{7}$/,
  'vi-VN': /^(\\+?84|0)((3([2-9]))|(5([25689]))|(7([0|6-9]))|(8([1-9]))|(9([0-9])))\\d{7}$/,
};

export default function isMobilePhone(str, locale, options) {
  assertString(str);
  if (locale in phones) {
    return phones[locale].test(str);
  } else if (!locale || locale === 'any') {
    for (const key in phones) {
      if (phones.hasOwnProperty(key)) {
        const phone = phones[key];
        if (phone.test(str)) {
          return true;
        }
      }
    }
    return false;
  }
  return false;
}

export const locales = Object.keys(phones);
"""


# ── Tests ─────────────────────────────────────────────────────────────────

class TestStripMarkdown:
    def test_removes_headers(self):
        text = "### FILE: test.js\n## SEARCH:\n# REPLACE:"
        result = _strip_markdown(text)
        assert "###" not in result
        assert "##" not in result
        assert "FILE: test.js" in result

    def test_removes_code_fences(self):
        text = "```javascript\nconst x = 1;\n```"
        result = _strip_markdown(text)
        assert "```" not in result
        assert "const x = 1;" in result

    def test_preserves_code_content(self):
        text = "### SEARCH:\n```js\n  'uz-UZ': /regex/,\n```"
        result = _strip_markdown(text)
        assert "'uz-UZ': /regex/," in result


class TestParseFix:
    def test_xml_format(self):
        f, s, r = _parse_fix(XML_OUTPUT)
        assert f == "src/lib/isMobilePhone.js"
        assert "uz-UZ" in s
        assert "33|55" in r

    def test_markdown_simple(self):
        f, s, r = _parse_fix(MARKDOWN_SIMPLE)
        assert f == "src/lib/isMobilePhone.js", f"Got file: {f!r}"
        assert "uz-UZ" in s, f"Got search: {s!r}"
        assert "33|55" in r, f"Got replace: {r!r}"

    def test_markdown_large_refactor(self):
        f, s, r = _parse_fix(MARKDOWN_LARGE_REFACTOR)
        assert f == "src/lib/isMobilePhone.js"
        assert "locale in phones" in s
        assert "throw new Error" in r

    def test_empty_input(self):
        f, s, r = _parse_fix("")
        assert f == ""
        assert s == ""
        assert r == ""

    def test_garbage_input(self):
        f, s, r = _parse_fix("This is just a random explanation with no fix.")
        assert f == ""


class TestExtractPlan:
    def test_extracts_plan(self):
        assert "Fix regex" in _extract_plan(XML_OUTPUT)

    def test_no_plan(self):
        assert _extract_plan("no plan here") == ""


class TestFindAndReplace:
    def test_exact_match(self):
        original = "  'uz-UZ': /old/,\n  'vi-VN': /other/,"
        result = _find_and_replace(original, "  'uz-UZ': /old/,", "  'uz-UZ': /new/,")
        assert result is not None
        assert "'uz-UZ': /new/," in result

    def test_whitespace_normalized(self):
        original = "    const x = 1;\n    const y = 2;"
        result = _find_and_replace(original, "const x = 1;", "const x = 42;")
        assert result is not None
        assert "42" in result

    def test_level4_large_block_diff(self):
        """Test that Level 4 extracts the minimal diff from a large SEARCH/REPLACE."""
        _, search, replace = _parse_fix(MARKDOWN_LARGE_REFACTOR)
        result = _find_and_replace(SOURCE_FILE, search, replace)
        assert result is not None, "Level 4 should find and apply the diff"
        assert "throw new Error" in result

    def test_exact_small_regex_fix(self):
        """Test the actual fix case: small regex change."""
        _, search, replace = _parse_fix(MARKDOWN_SIMPLE)
        result = _find_and_replace(SOURCE_FILE, search, replace)
        assert result is not None, "Small regex fix should match"
        assert "33|55" in result

    def test_no_match(self):
        result = _find_and_replace("completely different code", "not here", "nope")
        assert result is None


class TestAttemptFixRelevance:
    def test_rejects_unrelated_refactor_candidate(self):
        """Large unrelated refactor candidate should be rejected for this issue."""
        issue = (
            "Title: isMobilePhone('uz-UZ') returns false for valid Uzbekistan carrier codes\n"
            "Body: +998770179999 should pass"
        )
        literals = _extract_issue_identifiers(issue)
        file_info = {"src/lib/isMobilePhone.js": SOURCE_FILE}
        known_paths = ["src/lib/isMobilePhone.js"]

        diff, _, reason = _attempt_fix(
            MARKDOWN_LARGE_REFACTOR,
            file_info,
            known_paths,
            literals,
        )
        assert diff == ""
        assert "not relevant" in reason.lower() or "no usable" in reason.lower()

    def test_accepts_relevant_literal_fix_candidate(self):
        """Literal-targeted candidate should be accepted and produce a diff."""
        issue = "Fix uz-UZ regex for +99877 numbers"
        literals = _extract_issue_identifiers(issue)
        file_info = {"src/lib/isMobilePhone.js": SOURCE_FILE}
        known_paths = ["src/lib/isMobilePhone.js"]

        diff, _, reason = _attempt_fix(
            MARKDOWN_SIMPLE,
            file_info,
            known_paths,
            literals,
        )
        assert reason == ""
        assert diff.startswith("--- a/src/lib/isMobilePhone.js")
        assert "+  'uz-UZ': /^(\\+?998)?(33|55|6[125-79]|7[1-69]|88|9\\d)\\d{7}$/," in diff


class TestCoderNodeRetry:
    def test_retries_after_unrelated_candidate(self, monkeypatch):
        """coder_node should retry when first candidate is irrelevant, then recover."""

        class _Resp:
            def __init__(self, content):
                self.content = content

        call_counter = {"count": 0}

        def fake_invoke_with_role_fallback(**_kwargs):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                return _Resp(MARKDOWN_LARGE_REFACTOR), "qwen-2.5-coder-32b"
            return _Resp(MARKDOWN_SIMPLE), "qwen-2.5-coder-32b"

        monkeypatch.setattr("issue_resolver.nodes.coder.invoke_with_role_fallback", fake_invoke_with_role_fallback)

        file_context = [
            "# --- file: src/lib/isMobilePhone.js ---\n" + SOURCE_FILE
        ]
        state = cast(AgentState, {
            "issue": "isMobilePhone('uz-UZ') fails for +99877 numbers",
            "repo_path": "./sandbox_workspace",
            "file_context": file_context,
            "plan": "",
            "proposed_fix": "",
            "errors": "",
            "validation_status": "",
            "next_step": "",
            "iterations": 0,
            "is_resolved": False,
            "environment_config": {},
            "contribution_guidelines": "",
            "history": [],
        })

        out = coder_node(state)
        assert "proposed_fix" in out
        assert out["proposed_fix"].startswith("--- a/src/lib/isMobilePhone.js")
        assert call_counter["count"] >= 2
