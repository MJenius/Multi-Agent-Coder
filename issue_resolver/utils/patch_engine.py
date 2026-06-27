import re
import difflib

class FuzzyPatchEngine:
    def __init__(self, file_content: str):
        self.file_content = file_content
        self.lines = file_content.split('\n')

    def apply_block(self, search_text: str, replace_text: str) -> dict:
        search_lines = search_text.split('\n')
        replace_lines = replace_text.split('\n')
        if not search_lines:
            return {"success": False, "hint": "Search block is empty."}
        norm_search = [line.rstrip() for line in search_lines]
        norm_file = [line.rstrip() for line in self.lines]
        n = len(self.lines)
        m = len(search_lines)
        if m > n:
            best_i = 0
            best_ratio = difflib.SequenceMatcher(None, "\n".join(norm_search), "\n".join(norm_file)).ratio()
        else:
            best_ratio = -1.0
            best_i = -1
            search_str = "\n".join(norm_search)
            for i in range(n - m + 1):
                window_str = "\n".join(norm_file[i : i + m])
                ratio = difflib.SequenceMatcher(None, search_str, window_str).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_i = i
        if best_ratio == 1.0:
            self.lines[best_i : best_i + m] = replace_lines
            self.file_content = "\n".join(self.lines)
            return {"success": True}
        if best_ratio >= 0.92:
            self.lines[best_i : best_i + m] = replace_lines
            self.file_content = "\n".join(self.lines)
            return {"success": True}
        if best_ratio >= 0.82:
            self.lines[best_i : best_i + m] = replace_lines
            self.file_content = "\n".join(self.lines)
            return {"success": True}
        line_num = best_i + 1 if best_i != -1 else 1
        return {
            "success": False,
            "hint": f"Failed to apply patch. Closest match at line {line_num} with similarity {best_ratio:.2f}."
        }

def parse_and_apply_blocks(file_path: str, llm_output: str) -> dict:
    pattern = re.compile(
        r"^<<<<<<< SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>> REPLACE$",
        re.MULTILINE | re.DOTALL
    )
    blocks = pattern.findall(llm_output)
    if not blocks:
        return {"success": False, "hint": "No SEARCH/REPLACE blocks found in output."}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    engine = FuzzyPatchEngine(content)
    for search_text, replace_text in blocks:
        res = engine.apply_block(search_text, replace_text)
        if not res["success"]:
            return res
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        f.write(engine.file_content)
    return {"success": True}
