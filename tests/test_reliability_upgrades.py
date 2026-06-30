import os
from issue_resolver.utils.patch_engine import FuzzyPatchEngine, parse_and_apply_blocks
from issue_resolver.nodes.researcher import _manage_context_budget
from issue_resolver.nodes.issue_classifier import clean_issue_for_classification
from issue_resolver.nodes.candidate_generator import _validate_patch_blocks
from benchmark_suite import expected_to_cat

def test_fuzzy_patch_engine_exact():
    content = "line 1\nline 2\nline 3\n"
    engine = FuzzyPatchEngine(content)
    res = engine.apply_block("line 2", "line 2 modified")
    assert res["success"] is True
    assert engine.file_content == "line 1\nline 2 modified\nline 3\n"

def test_fuzzy_patch_engine_indentation_shift():
    content = "class MyClass:\n    def run(self):\n        print('foo')\n"
    engine = FuzzyPatchEngine(content)
    # Search block has different indentation level
    search_block = "def run(self):\n    print('foo')"
    replace_block = "def run(self):\n    print('bar')\n    print('baz')"
    res = engine.apply_block(search_block, replace_block)
    assert res["success"] is True
    assert "        print('bar')" in engine.file_content
    assert "        print('baz')" in engine.file_content

def test_fuzzy_patch_engine_ast_context():
    content = "def test_func():\n    val = 1\n    return val\n"
    engine = FuzzyPatchEngine(content)
    search_block = "val = 1\nreturn val"
    replace_block = "val = 2\nreturn val"
    res = engine.apply_block(search_block, replace_block)
    assert res["success"] is True
    assert "val = 2" in engine.file_content

def test_clean_issue_for_classification():
    issue = """
<!-- this is a comment -->
### Checklist
- [ ] I have read the contribution guidelines
- [x] This is a runtime bug with calculate_total
Some custom description [read the docs](http://example.com/readme.md)
"""
    cleaned = clean_issue_for_classification(issue)
    assert "comment" not in cleaned
    assert "Checklist" not in cleaned
    assert "contribution guidelines" not in cleaned
    assert "readme.md" not in cleaned
    assert "calculate_total" in cleaned
    assert "Some custom description" in cleaned

def test_validate_patch_blocks(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def hello():\n    print('hi')\n", encoding="utf-8")
    
    # Valid block
    patch_valid = "<<<<<<< SEARCH\ndef hello():\n    print('hi')\n=======\ndef hello():\n    print('hello')\n>>>>>>> REPLACE"
    is_valid, _ = _validate_patch_blocks(patch_valid, str(tmp_path), ["test.py"])
    assert is_valid is True
    
    # Invalid block
    patch_invalid = "<<<<<<< SEARCH\ndef hello():\n    print('invalid')\n=======\ndef hello():\n    print('hello')\n>>>>>>> REPLACE"
    is_valid, err = _validate_patch_blocks(patch_invalid, str(tmp_path), ["test.py"])
    assert is_valid is False
    assert "was not found" in err

def test_expected_to_cat():
    assert expected_to_cat("bug_fix") == "Bug"
    assert expected_to_cat("typing") == "Typing"
    assert expected_to_cat("configuration") == "Configuration"

def test_context_budget_manager():
    snippets = [
        "# --- file: a.py ---\n" + "A" * 5000,
        "# --- file: b.py ---\n" + "B" * 5000,
        "# --- file: a.py ---\n" + "A" * 6000,
        "# --- file: c.py ---\n" + "C" * 3000
    ]
    processed = _manage_context_budget(snippets)
    assert len(processed) == 3
    b_snip = [s for s in processed if "b.py" in s][0]
    assert "[...truncated]" in b_snip
