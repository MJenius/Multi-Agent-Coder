#!/usr/bin/env python
"""Test improved hint extraction"""

from issue_resolver.nodes.researcher import _extract_hints_from_issue

# Test with actual QRCoder issue 646 text
issue_646 = """Title: ASCII 'small' renderer prints inverted by default

Body: By default, the ASCII 'small' renderer prints inverted compared to the ASCII standard renderer or any other renderer. 

It has a boolean 'invert' argument, which defaults to 'false'; setting this to 'true' corrects the output. 

This is is likely caused by incorrect constants used in the code:

bool BLACK = true, WHITE = false;

As seen above, the 'black' text constant consists of a space, while the the 'white' text constants are 'filled in'.

This should be fixed in v2, as it would be a breaking change for v1.

CRITICAL INSTRUCTION: The repository code is located strictly inside the './sandbox_workspace' directory. Do not search the root directory '.'

HNT: For QRCoder issues involving ASCII rendering, the bug is likely in QRCoder/AsciiQRCode.cs. Focus your search there."""

# Test the extraction
hints = _extract_hints_from_issue(issue_646)

print("Hint Extraction Test")
print("=" * 60)
print(f"Total hints found: {len(hints)}")
print()

for i, hint in enumerate(hints, 1):
    is_valid = '/' in hint or '.' in hint
    status = "VALID" if is_valid else "INVALID"
    print(f"Hint {i} [{status}]: {hint[:70]}")

print()
print("Expected: Just 'QRCoder/AsciiQRCode.cs' (or close variants)")
print()

# Check if correct file path is included
if any('AsciiQRCode' in h for h in hints):
    print("Result: PASS - Found AsciiQRCode reference")
else:
    print("Result: FAIL - Did not find AsciiQRCode reference")
