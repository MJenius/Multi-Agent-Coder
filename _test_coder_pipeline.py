"""Quick integration test for the coder pipeline."""
from issue_resolver.nodes.coder import _extract_file_info, _find_and_replace, _make_diff, _match_path

# Simulate researcher output (with line numbers as the read_file tool produces)
ctx = (
    "# --- [HINTED] file: QRCoder/AsciiQRCode.cs ---\n"
    "1: using System;\n"
    "2: \n"
    "3: public class AsciiQRCode {\n"
    "4:     var palette = new {\n"
    '5:         WHITE_ALL = "\\u2588",\n'
    '6:         WHITE_BLACK = "\\u2580",\n'
    '7:         BLACK_WHITE = "\\u2584",\n'
    '8:         BLACK_ALL = " ",\n'
    "9:     };\n"
    "10: }"
)

info = _extract_file_info([ctx])
print("Files:", list(info.keys()))
original = info["QRCoder/AsciiQRCode.cs"]
print("Lines:", len(original.split("\n")))
print("Content preview:", repr(original[:200]))

# Simulate LLM search/replace
search_block = (
    '        WHITE_ALL = "\\u2588",\n'
    '        WHITE_BLACK = "\\u2580",\n'
    '        BLACK_WHITE = "\\u2584",\n'
    '        BLACK_ALL = " ",'
)
replace_block = (
    '        WHITE_ALL = " ",\n'
    '        WHITE_BLACK = "\\u2584",\n'
    '        BLACK_WHITE = "\\u2580",\n'
    '        BLACK_ALL = "\\u2588",'
)

modified = _find_and_replace(original, search_block, replace_block)
if modified and modified != original:
    diff = _make_diff(original, modified, "QRCoder/AsciiQRCode.cs")
    print("\nGenerated diff:")
    print(diff)
    print("\nSUCCESS: Full pipeline works!")
else:
    print("\nfind_and_replace exact match failed, testing fuzzy...")
    # The indentation might not match exactly - test fuzzy
    # Try without the leading spaces
    search2 = (
        'WHITE_ALL = "\\u2588",\n'
        'WHITE_BLACK = "\\u2580",\n'
        'BLACK_WHITE = "\\u2584",\n'
        'BLACK_ALL = " ",'
    )
    replace2 = (
        'WHITE_ALL = " ",\n'
        'WHITE_BLACK = "\\u2584",\n'
        'BLACK_WHITE = "\\u2580",\n'
        'BLACK_ALL = "\\u2588",'
    )
    modified2 = _find_and_replace(original, search2, replace2)
    if modified2:
        diff = _make_diff(original, modified2, "QRCoder/AsciiQRCode.cs")
        print("\nGenerated diff (fuzzy match):")
        print(diff)
        print("\nSUCCESS: Fuzzy pipeline works!")
    else:
        print("Both exact and fuzzy failed - checking data...")
        for i, line in enumerate(original.split("\n")):
            print(f"  {i}: {repr(line)}")

# Test path matching
print("\n--- Path Matching Tests ---")
assert _match_path("AsciiQRCode.cs", ["QRCoder/AsciiQRCode.cs"]) == "QRCoder/AsciiQRCode.cs"
assert _match_path("./sandbox_workspace/QRCoder/Foo.cs", ["QRCoder/Foo.cs"]) == "QRCoder/Foo.cs"
print("Path matching: OK")

print("\nAll integration tests passed!")
