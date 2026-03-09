#!/usr/bin/env python
"""Quick test of hint extraction"""

from issue_resolver.nodes.researcher import _extract_hints_from_issue

# Test case 1: Actual QRCoder issue 646
issue_646 = """
Title: ASCII 'small' renderer prints inverted by default

HINT: For QRCoder issues involving ASCII rendering, the bug is likely in QRCoder/AsciiQRCode.cs. Focus your search there.
"""

result = _extract_hints_from_issue(issue_646)
print(f"Test 1 - QRCoder issue 646:")
print(f"  Result: {result}")
print(f"  Pass: {'QRCoder' in str(result)}")
print()

# Test case 2: Simple hint
issue_simple = "Bug in file src/main.py needs fixing"
result = _extract_hints_from_issue(issue_simple)
print(f"Test 2 - Simple hint:")
print(f"  Result: {result}")
print(f"  Pass: {'src/main.py' in str(result)}")
