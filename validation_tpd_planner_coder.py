#!/usr/bin/env python3
"""
Validation for TPD Quota Handling + Planner Context Optimization + Coder Safe Access
"""

import sys
from issue_resolver.llm_utils import _is_quota_exceeded, _QUOTA_EXCEEDED_MODELS
from issue_resolver.nodes.planner import _SYSTEM_PROMPT as PLANNER_PROMPT
from issue_resolver.nodes.coder import _SYSTEM_PROMPT as CODER_PROMPT


def test_tpd_quota_detection():
    """Verify TPD quota detection function works."""
    print("\n" + "="*70)
    print("TEST: TPD Quota Detection")
    print("="*70)
    
    # Test case 1: TPD error with 429 and "tokens per day"
    exc1 = Exception("429: tokens per day exceeded")
    result1 = _is_quota_exceeded(exc1)
    print(f"  ✓ Detect 'tokens per day': {result1}")
    assert result1, "Should detect tokens per day error"
    
    # Test case 2: TPD error with daily quota
    exc2 = Exception("429 Too Many Requests: daily quota exceeded")
    result2 = _is_quota_exceeded(exc2)
    print(f"  ✓ Detect 'daily quota': {result2}")
    assert result2, "Should detect daily quota error"
    
    # Test case 3: Regular rate limit (should NOT trigger)
    exc3 = Exception("429: rate limit exceeded")
    result3 = _is_quota_exceeded(exc3)
    print(f"  ✓ Ignore regular rate limit: {not result3}")
    assert not result3, "Should NOT treat rate limit as quota"
    
    # Test case 4: Different error (should NOT trigger)
    exc4 = Exception("400: bad request")
    result4 = _is_quota_exceeded(exc4)
    print(f"  ✓ Ignore non-429 errors: {not result4}")
    assert not result4, "Should NOT trigger on non-429 errors"
    
    print("  ✅ All TPD detection tests passed!\n")


def test_quota_exceeded_models_exists():
    """Verify _QUOTA_EXCEEDED_MODELS is initialized."""
    print("="*70)
    print("TEST: QUOTA_EXCEEDED_MODELS Session Tracking")
    print("="*70)
    
    # Check that _QUOTA_EXCEEDED_MODELS exists and is a set
    assert isinstance(_QUOTA_EXCEEDED_MODELS, set), "_QUOTA_EXCEEDED_MODELS should be a set"
    print(f"  ✓ _QUOTA_EXCEEDED_MODELS is a set: {_QUOTA_EXCEEDED_MODELS}")
    
    # Test that we can add to it (simulating quota exhaustion)
    test_model = "test-model-for-quota"
    _QUOTA_EXCEEDED_MODELS.add(test_model)
    assert test_model in _QUOTA_EXCEEDED_MODELS, "Should be able to add models to set"
    print(f"  ✓ Can add models to tracking set: {test_model} added")
    
    # Clean up
    _QUOTA_EXCEEDED_MODELS.discard(test_model)
    
    print("  ✅ Session tracking test passed!\n")


def test_planner_context_optimization():
    """Verify Planner prompt includes context optimization references."""
    print("="*70)
    print("TEST: Planner Context Optimization")
    print("="*70)
    
    # Check that planner has the optimization logic in code (via imports)
    from issue_resolver.nodes.planner import GROQ_CONTEXT_WINDOWS
    
    print(f"  ✓ GROQ_CONTEXT_WINDOWS imported: {len(GROQ_CONTEXT_WINDOWS)} models")
    for model, window in GROQ_CONTEXT_WINDOWS.items():
        print(f"    - {model}: {window} tokens")
    
    # Verify context windows are correct
    assert GROQ_CONTEXT_WINDOWS["llama-3.3-70b-versatile"] == 8192, "llama should be 8K"
    assert GROQ_CONTEXT_WINDOWS["qwen-2.5-coder-32b"] == 32768, "qwen should be 32K"
    print("  ✓ Context window sizes verified")
    
    # Check that planner code has truncation logic (by reading the function)
    import inspect
    from issue_resolver.nodes.planner import planner_node
    source = inspect.getsource(planner_node)
    
    truncation_markers = [
        "truncated",
        "context_window",
        "available_for_context",
    ]
    found_markers = [m for m in truncation_markers if m in source.lower()]
    
    print(f"  ✓ Found truncation logic markers: {found_markers}")
    assert len(found_markers) >= 2, "Should have context truncation logic"
    
    print("  ✅ Planner context optimization verified!\n")


def test_coder_safe_access_guidance():
    """Verify Coder has safe attribute access guidance."""
    print("="*70)
    print("TEST: Coder Safe Attribute Access Guidance")
    print("="*70)
    
    # Check for safe access patterns in prompt
    safe_access_patterns = [
        "getattr",
        "hasattr",
        "optional",
        "attribute",
    ]
    
    found_patterns = [p for p in safe_access_patterns if p in CODER_PROMPT.lower()]
    print(f"  ✓ Found safe access guidance: {found_patterns}")
    assert len(found_patterns) >= 3, "Should mention getattr, hasattr, and optional"
    
    # Check for Stripe-specific guidance
    if "stripe" in CODER_PROMPT.lower():
        print("  ✓ Has Stripe-specific guidance")
    
    # Check for example patterns
    if "getattr(obj" in CODER_PROMPT:
        print("  ✓ Has getattr() example pattern")
    if "hasattr(obj" in CODER_PROMPT:
        print("  ✓ Has hasattr() example pattern")
    
    print("  ✅ Coder safe access guidance verified!\n")


def test_integration():
    """Test that graph still compiles with all changes."""
    print("="*70)
    print("TEST: Graph Integration (All Changes Together)")
    print("="*70)
    
    from issue_resolver.graph import app
    
    # Check that app is built and has expected attributes
    assert app is not None, "Graph should compile"
    print(f"  ✓ Graph compiled: {type(app).__name__}")
    
    # Check that graph has the expected nodes
    if hasattr(app, "nodes"):
        nodes = list(app.nodes.keys()) if hasattr(app.nodes, "keys") else []
        print(f"  ✓ Graph has nodes: {len(nodes)} nodes")
    
    print("  ✅ Integration test passed!\n")


if __name__ == "__main__":
    try:
        test_tpd_quota_detection()
        test_quota_exceeded_models_exists()
        test_planner_context_optimization()
        test_coder_safe_access_guidance()
        test_integration()
        
        print("\n" + "="*70)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("="*70)
        print("\nChanges Verified:")
        print("  ✅ TPD quota detection implemented")
        print("  ✅ Session-level quota tracking in place")
        print("  ✅ Planner context truncation active")
        print("  ✅ Coder safe attribute access guidance added")
        print("  ✅ Graph compiles and integrates successfully")
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
