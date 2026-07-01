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


def test_clean_issue_text():
    from issue_resolver.utils.issue_utils import clean_issue_text
    raw = """
    Title: Fix type crash
    <!-- HTML Comment -->
    - [ ] Checklist item 1
    - [x] Checklist item 2
    Please check the following before submitting.
    Follow the [instructions](http://example.com/instruct.py) and read [docs](http://github.com/user/docs).
    Let's see: `calculate_total` in `calculate.py` crashes.
    """
    cleaned = clean_issue_text(raw)
    assert "HTML Comment" not in cleaned
    assert "Checklist item 1" not in cleaned
    assert "Checklist item 2" not in cleaned
    assert "Please check" not in cleaned
    assert "instruct.py" not in cleaned
    assert "http://" not in cleaned
    assert "calculate_total" in cleaned
    assert "calculate.py" in cleaned


def test_selective_reset_runtime_context():
    import issue_resolver.runtime_context as rc
    # Mock plugins and router
    rc._MODEL_ROUTER = "router_instance"
    rc._PLUGIN_REGISTRY = "plugin_registry_instance"
    rc._KNOWLEDGE_GRAPH = "knowledge_graph_instance"
    rc._ENVIRONMENT_CONFIG = {"repo_root": "/path"}
    
    rc.reset_runtime_context()
    
    assert rc.get_model_router() == "router_instance"
    assert rc.get_plugin_registry() == "plugin_registry_instance"
    assert rc.get_knowledge_graph() is None
    assert rc.get_environment_config() == {}


def test_detect_environment_metadata(tmp_path):
    # Create mock python manifest files
    (tmp_path / "pyproject.toml").write_text("""
    [project]
    dependencies = ["django", "fastapi"]
    
    [tool.pytest]
    test_option = true
    """, encoding="utf-8")
    (tmp_path / "uv.lock").write_text("""
    [[package]]
    name = "django"
    version = "5.0"
    """, encoding="utf-8")
    
    from issue_resolver.utils.metadata_detector import detect_environment_metadata
    metadata = detect_environment_metadata(tmp_path)
    
    assert metadata["primary_language"] == "python"
    assert metadata["framework"] == "Django"
    assert metadata["test_framework"] == "pytest"
    assert metadata["package_manager"] == "uv"


def test_compute_localization_quality_metrics():
    from issue_resolver.core.metrics import compute_localization_quality_metrics
    state = {
        "localization_result": {
            "primary_files": [
                {"path": "src/main.py", "score": 1.0, "confidence": "high"},
                {"path": "src/utils.py", "score": 0.8, "confidence": "medium"}
            ],
            "graph_hits": 8,
            "graph_misses": 2,
        },
        "localization_confidence": 0.90,
        "structured_plan": {
            "files_to_edit": ["src/main.py"]
        }
    }
    
    metrics = compute_localization_quality_metrics(state, is_resolved=True)
    quality = metrics["localization_quality"]
    
    assert quality["precision"] == 0.5  # 1 matched / 2 retrieved
    assert quality["recall"] == 1.0     # 1 matched / 1 expected
    assert quality["all_edited_in_initial"] is True
    assert quality["graph_hit_rate"] == 0.8
    assert abs(quality["confidence_calibration_error"] - 0.10) < 1e-6

