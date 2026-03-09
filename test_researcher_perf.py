 #!/usr/bin/env python
"""Performance test for Researcher node with QRCoder issue."""

import time
from issue_resolver.state import AgentState
from issue_resolver.nodes.researcher import researcher_node

# The exact issue that was hanging for 4+ minutes
qrcoder_issue = """Title: ASCII 'small' renderer prints inverted by default

Body: By default, the ASCII 'small' renderer prints inverted compared to the ASCII standard renderer or any other renderer. It has a boolean 'invert' argument, which defaults to 'false'; setting this to 'true' corrects the output. This is likely caused by incorrect constants used in the code:

bool BLACK = true, WHITE = false;

var palette = new
{
    WHITE_ALL = "█",
    WHITE_BLACK = "▀",
    BLACK_WHITE = "▄",
    BLACK_ALL = " "
};

🎯 HINT: The bug is likely in QRCoder/ASCIIQRCode.cs"""

# Create initial state
state = AgentState(
    issue=qrcoder_issue,
    repo_path='./sandbox_workspace',
    file_context=[],
    errors=''
)

# Run researcher with timer
print("[TEST] Running Researcher Node on QRCoder issue...")
print(f"Issue length: {len(qrcoder_issue)} chars")
print("Expected behavior: Complete in <1 minute, populate file_context")
print()

start = time.time()
result = researcher_node(state)
duration = time.time() - start

print()
print(f"Duration: {duration:.2f}s")
print(f"Snippets collected: {len(result.get('file_context', []))}")
print(f"Files read: {sum(1 for s in result.get('file_context', []) if '# --- file:' in s)}")

if result.get('file_context'):
    print(f"\nFirst snippet (first 400 chars):")
    snippet = result['file_context'][0][:400]
    print(snippet + ("..." if len(result['file_context'][0]) > 400 else ""))
    
    if duration < 60:
        print(f"\n✅ SUCCESS: Completed in {duration:.2f}s (target: <60s)")
    else:
        print(f"\n⚠️  SLOW: Completed in {duration:.2f}s (target: <60s)")
else:
    print("\n⚠️  FAILED: No snippets collected!")
    print("This indicates the researcher did not find the target file.")
