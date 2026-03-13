"""
TestGen Node -- Generates a framework-aware failing test that reproduces the issue.

The TestGen node receives the plan + issue context + detected test framework,
and writes a test that FAILS initially (before fix) and PASSES after the fix.

Supports: pytest (Python), Jest (JavaScript), xUnit/NUnit (C#), unittest (Python).

Note: TestGen generates the test code but does not write it yet. The actual
test file writing and execution will happen during review/validation.
"""

from __future__ import annotations

import re
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.config import TESTGEN_MODEL_CANDIDATES
from issue_resolver.llm_utils import invoke_with_role_fallback, calculate_max_tokens


def _get_test_framework_prompt(framework: str, language: str, env_type: str) -> str:
    """Return framework-specific instructions for TestGen."""
    
    templates = {
        ("pytest", "python"): """Write a pytest test file that reproduces the issue.

Framework: pytest (Python)
- Import: `import pytest`
- Test function: `def test_<name>():`
- Assertions: `assert <condition>`
- Test should FAIL before fix, PASS after

Output ONLY the Python test code in a ```python code block. No explanation.""",
        
        ("unittest", "python"): """Write a unittest test file that reproduces the issue.

Framework: unittest (Python)
- Inherit: `class TestCase(unittest.TestCase):`
- Test method: `def test_<name>(self):`
- Assertions: `self.assertEqual(...)`
- Test should FAIL before fix, PASS after

Output ONLY the Python test code in a ```python code block. No explanation.""",
        
        ("jest", "javascript"): """Write a Jest test file that reproduces the issue.

Framework: Jest (JavaScript)
- Describe block: `describe('<name>', () => {`
- Test function: `test('<name>', () => {`
- Assertions: `expect(...).toBe(...);`
- Test should FAIL before fix, PASS after

Output ONLY the JavaScript test code in a ```javascript code block. No explanation.""",
        
        ("vitest", "javascript"): """Write a Vitest test file that reproduces the issue.

Framework: Vitest (JavaScript)
- Import: `import { test, expect } from 'vitest'`
- Test function: `test('<name>', () => {`
- Assertions: `expect(...).toBe(...);`
- Test should FAIL before fix, PASS after

Output ONLY the JavaScript test code in a ```javascript code block. No explanation.""",
        
        ("xunit", "dotnet"): """Write an xUnit test that reproduces the issue.

Framework: xUnit (C#)
- Using: `using Xunit;`
- Test class: `public class TestFixture`
- Test method: `[Fact] public void TestName() {`
- Assertions: `Assert.Equal(...)`
- Test should FAIL before fix, PASS after

Output ONLY the C# test code in a ```csharp code block. No explanation.""",
        
        ("nunit", "dotnet"): """Write an NUnit test that reproduces the issue.

Framework: NUnit (C#)
- Using: `using NUnit.Framework;`
- Test class: `[TestFixture] public class TestFixture`
- Test method: `[Test] public void TestName() {`
- Assertions: `Assert.AreEqual(...)`
- Test should FAIL before fix, PASS after

Output ONLY the C# test code in a ```csharp code block. No explanation.""",
    }
    
    # Try specific (framework, language) combo first
    key = (framework, language)
    if key in templates:
        return templates[key]
    
    # Fallback to pytest (most common)
    return templates[("pytest", "python")]


def testgen_node(state: AgentState) -> dict:
    """Generate a failing test that reproduces the issue."""
    print("[TestGen] Generating reproduction test...")

    issue_text = state.get("issue", "(no issue)")
    file_context = state.get("file_context", [])
    plan = state.get("plan", "")
    environment_config = state.get("environment_config", {})
    iterations = state.get("iterations", 0)
    
    # Extract framework info
    env_type = environment_config.get("environment_type", "python") if isinstance(environment_config, dict) else "python"
    test_framework = environment_config.get("test_framework", "pytest") if isinstance(environment_config, dict) else "pytest"
    repo_root = environment_config.get("repo_root", "./") if isinstance(environment_config, dict) else "./"
    
    # Map environment type to language
    language_map = {
        "python": "python",
        "nodejs": "javascript",
        "dotnet": "dotnet",
        "unknown": "python",
    }
    language = language_map.get(env_type, "python")
    
    print(f"[TestGen] Framework: {test_framework}, Language: {language}")
    
    # Build framework-specific prompt
    framework_instructions = _get_test_framework_prompt(test_framework, language, env_type)
    
    # Build context
    context_parts = []
    if plan:
        context_parts.append(f"## Fix Strategy\n{plan}")
    if file_context:
        context_parts.append(f"## Code Context\n" + "\n\n".join(file_context))
    
    context_str = "\n\n".join(context_parts) if context_parts else "(no context)"
    
    # Build prompt with emphasis on test-driven validation
    user_prompt = f"""GitHub Issue to Reproduce:
{issue_text}

## Repository Context
{context_str}

{framework_instructions}

## Critical: Test-Driven Validation
This test is CRITICAL to the fix process:
1. It MUST fail with the current code (reproduce the issue)
2. It will be run BEFORE the Coder attempts any fixes
3. After the fix is applied, it MUST pass
4. This ensures the fix actually solves the bug

For attribute access errors (e.g., AttributeError on optional fields):
- Create a mock/test case that has MISSING attributes
- This will trigger the exact error from the issue
- Example: Create a Stripe LineItem WITHOUT a subscription_item field

STRIPE-SPECIFIC MOCKING (using unittest.mock.MagicMock):
──────────────────────────────────────────────────────
```python
from unittest.mock import MagicMock

# Create a LineItem mock WITHOUT subscription_item attribute
line_item = MagicMock(spec=['id', 'amount', 'description'])  # subscription_item NOT included
# Using spec ensures ONLY listed attributes exist

# Or use side_effect to explicitly raise AttributeError:
line_item = MagicMock()
del line_item.subscription_item  # Ensure attribute doesn't exist

# Test should try to access the missing attribute:
with pytest.raises(AttributeError):
    value = line_item.subscription_item  # This is what fails in the bug
```

Remember: The test MUST fail with the current code and PASS after the fix is applied."""
    
    messages = [
        SystemMessage(content="You are a test generation expert. Write concise, focused reproduction tests that expose the bug described in the issue. The test drives the entire fix process. For Stripe/API objects with optional attributes, use unittest.mock.MagicMock with explicit spec to create objects missing optional fields—this triggers AttributeError bugs."),
        HumanMessage(content=user_prompt),
    ]
    
    # Estimate tokens and calculate max_tokens
    estimated_input_tokens = (len(user_prompt) + 500) // 4  # 500 chars buffer for system prompt
    first_model = TESTGEN_MODEL_CANDIDATES[0] if TESTGEN_MODEL_CANDIDATES else "llama-3.3-70b-versatile"
    max_tokens = calculate_max_tokens(first_model, estimated_input_tokens)
    
    try:
        resp, chosen_model = invoke_with_role_fallback(
            role="TestGen",
            candidates=TESTGEN_MODEL_CANDIDATES,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        print(f"[TestGen] Using model: {chosen_model}")
    except Exception as exc:
        print(f"[TestGen] [ERROR] LLM failed: {exc}")
        return {
            "errors": f"TestGen failed: {exc}",
            "iterations": iterations + 1,
            "history": append_to_history("TestGen", "Error", str(exc)),
        }
    
    raw = getattr(resp, "content", "") or ""
    if not raw:
        return {
            "errors": "TestGen returned empty response",
            "iterations": iterations + 1,
            "history": append_to_history("TestGen", "Error", "Empty response"),
        }
    
    # Extract code from ```language ... ``` blocks
    test_code = ""
    patterns = [
        (r"```python\n(.*?)\n```", "python"),
        (r"```javascript\n(.*?)\n```", "javascript"),
        (r"```csharp\n(.*?)\n```", "csharp"),
        (r"```cs\n(.*?)\n```", "csharp"),
    ]
    
    for pattern, found_lang in patterns:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            test_code = match.group(1).strip()
            print(f"[TestGen] Extracted test code ({len(test_code)} chars)")
            break
    
    if not test_code:
        print("[TestGen] [WARNING] Could not extract code from fenced blocks, using full response")
        test_code = raw.strip()
    
    if not test_code:
        return {
            "errors": "Could not extract test code from LLM output",
            "iterations": iterations + 1,
            "history": append_to_history("TestGen", "Parse Failed", raw[:300]),
        }
    
    # Determine test file path based on language
    test_file_map = {
        "python": "test_issue_fix.py",
        "javascript": "test_issue_fix.js",
        "dotnet": "TestIssueFix.cs",
    }
    test_filename = test_file_map.get(language, "test_issue_fix.py")
    test_file_path = f"{repo_root}/{test_filename}"
    
    print(f"[TestGen] [TEST-DRIVEN] Test file: {test_filename}")  
    print(f"[TestGen] [TEST-DRIVEN] This test will determine if the fix works.")
    
    return {
        "test_code": test_code,
        "test_file_path": test_file_path,
        "test_framework_used": test_framework,
        "test_runs_initially": False,  # Will be verified during review
        "iterations": iterations + 1,
        "history": append_to_history(
            "TestGen",
            "Test Generated",
            f"Test file: {test_filename}\n{test_code[:300]}...",
        ),
    }
