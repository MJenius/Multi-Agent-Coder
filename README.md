# Multi-Agent GitHub Issue Resolution System 🤖🛠️

An autonomous, multi-agent AI system built using **LangGraph** and **Ollama** to analyze, resolve, and validate GitHub issues locally. It safely reads code context, generates surgical `Unified Diff` patches, and rigorously tests fixes within an isolated Docker sandbox.

---

## 🏗️ Architecture

The system leverages a StateGraph architecture driven by specialized agents. A **Supervisor** node manages the workflow routing between agents based on analytical condition checks and fallback LLM-reasoning (handling ambiguity or complex execution loops).

```mermaid
graph TD
    Supervisor{Supervisor (llama3.2:3b)}
    Researcher[Researcher (llama3.2:3b)]
    Coder[Coder (qwen2.5-coder:7b)]
    Reviewer((Reviewer (Docker Sandbox)))
    End([Resolution Report])

    Supervisor -- "Needs Context" --> Researcher
    Researcher -- "Context Gathered" --> Supervisor
    Supervisor -- "Ready for Fix" --> Coder
    Coder -- "Unified Diff Generated" --> Reviewer
    Reviewer -- "Patch Applied & Tested" --> Supervisor
    Supervisor -- "Tests Passed / Iteration Max" --> End
```

### 🧑‍💻 The Agents:
- **Supervisor (`llama3.2:3b`)**: The intelligent router. Monitors the `AgentState` with **Hard Guards**—deterministic logic that overrides the LLM to prevent infinite loops or premature exits. It ensures a graceful exit with a failure summary if a fix is not found within limits.
- **Researcher (`llama3.2:3b`)**: Explores the local codebase utilizing custom tool binding (`list_files`, `search_code`, `read_file`). Keeps strict token & RAM constraints on file reads.
- **Coder (`qwen2.5-coder:7b`)**: Extremely focused code generation model. Analyzes the file context against the GitHub issue and generates surgical string replace operations. **Enforced Logic Rule:** It is strictly prohibited from generating "no-op" diffs (only comments/docs/whitespace).
- **Reviewer (Deterministic)**: The Judge. Mounts the target code into an isolated, network-disabled **Docker Container**. It uses **Git-based baselines** to restore the sandbox environment between test runs, ensuring no side effects from failed patches.

---

## 🛠️ Technology Stack
- **Framework:** [LangGraph](https://python.langchain.com/docs/langgraph) (Stateful multi-actor orchestration)
- **Local Inference:** [Ollama](https://ollama.com/) (`llama3.2:3b` & `qwen2.5-coder:7b`)
- **Isolation/Sandbox:** Docker Engine (python-slim container with Git initialized)
- **Analytics & Logging:** Structured JSON-based history & token estimation.

---

## ✨ Key Features
- **Deterministic Flow Control**: Hard-coded guards in the Supervisor ensure the system terminates as soon as tests pass or limits are hit.
- **Surgical Patching**: Coder produces standard Unified Diffs that are easily reviewable and applied via `patch`.
- **Sandbox Isolation**: Code execution happens in a dedicated Docker sandbox with restricted permissions.
- **Observability**: Every agent action, tool call, and reasoning block is captured in a chronological `history` log.
- **Local-First**: All LLM inference happens on your machine via Ollama—no data leaves the local environment.

---

## 🚀 Getting Started

### Prerequisites
1. **Python 3.12+**
2. **Docker Engine / Desktop** installed and running.
3. **Ollama** installed with models pulled locally (approx. 16GB RAM recommended):
   ```bash
   ollama run llama3.2:latest
   ollama run qwen2.5-coder:7b
   ```

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MJenius/Multi-Agent-Coder.git
   cd Multi-Agent-Coder
   ```

2. Install graph dependencies via `pyproject.toml` or `pip`:
   ```bash
   pip install langgraph langchain-ollama psycopg[binary] fastapi uvicorn
   ```

3. Initialize the Docker Sandbox Environment:
   ```bash
   docker-compose up -d --build sandbox
   ```

---

## 🎮 How to Run

A sample "smoke test" driving the agents through a dummy repository sandbox is provided in `main.py`.

Run the resolution pipeline:
```bash
python -m issue_resolver.main
```

### Observability & Console Outputs
While the graph executes, comprehensive terminal logs will echo exactly what the agents are thinking, reasoning, and executing:
```text
[Supervisor] [ROUTE] Decision -> researcher  (iteration 1)
[Researcher] Starting codebase exploration...
[Researcher]    --> search_code({'query': 'calculate_total', 'directory': './sandbox_workspace'})
[Researcher]    --> read_file({'file_path': './sandbox_workspace/utils.py'})
[Coder] LLM returned 540 chars.
[Reviewer] Applying proposed fix in the sandbox...
[Reviewer] [OK] Code ran successfully.
...
```

### 📊 Resolution Report
Upon completion, the system generates a `resolution_report.json` summarizing the entire run. This includes a `history` of every step:
```json
{
    "is_resolved": true,
    "total_iterations": 3,
    "total_character_estimate": 1502,
    "files_read_summary": [...],
    "failed_diffs_and_tracebacks": [],
    "final_successful_diff": "--- a/utils.py\n+++ b/utils.py...",
    "history": [
        {
            "agent": "Supervisor",
            "action": "Routing Decision",
            "output": "researcher",
            "timestamp": "2026-03-07T..."
        },
        ...
    ]
}
```

---

## 🔧 Recent Infrastructure Improvements (March 2026)

### **Phase 4: Adaptive Model Downscaling**
Implements automatic fallback when LLM models are decommissioned by providers.

**Problem Solved:** When Groq decommissions a model (e.g., `mixtral-8x7b-32768`), the system no longer crashes with repeated retries.

**How it Works:**
- Session-level tracking of decommissioned models via `_DECOMMISSIONED_MODELS` set
- Smart error detection: When a 400/bad request error occurs, the model is flagged as permanently unavailable
- Automatic failover: Immediately tries the next fallback candidate in the chain
- No manual intervention required

**Model Fallback Hierarchy:**
- Supervisor: llama-3.3-70b → qwen-2.5-coder-32b
- Researcher: llama-3.3-70b → qwen-2.5-coder-32b
- Coder: qwen-2.5-coder-32b → llama-3.3-70b
- Each role has 1-2 backups to prevent single points of failure

**Files Modified:** `issue_resolver/llm_utils.py`, `issue_resolver/config.py`

### **Phase 3B: Smart Context Extraction**
Preserves critical debugging information when issue descriptions exceed token limits.

**Problem Solved:** Large GitHub issues with stack traces were getting truncated, losing essential debugging context.

**How it Works:**
- Priority-based extraction identifies and preserves critical sections:
  1. **Stack traces / error messages** (HIGHEST priority)
  2. "To Reproduce" steps
  3. "Expected vs Actual" behavior
  4. Environment/config details
  5. Remaining text (truncated last)
- Uses regex pattern detection to find section headers
- Sliding window approach captures complete sections
- Result: Preserves 95%+ of critical content within 4000 character limit

**Key Functions:**
- `extract_critical_sections(text, max_length=4000)` - Smart truncation
- `find_stack_trace(text)` - Extract stack traces from issue text
- `prioritize_issue_context(text)` - Separate title from intelligently truncated body

**Files Modified:** `issue_resolver/utils/issue_utils.py` (new), `app.py`

### **Future Enhancements (Phase 2A & 3A)**

**Phase 2A: Fuzzy Matching** (3-5 hours effort, 10% ROI)
- Problem: Indentation differences cause exact-match failures in code patches
- Solution: RapidFuzz with fallback chain (Exact → Fuzzy 0.85 → Fuzzy 0.75)
- Expected Result: Reduce fix failure rate from ~15% to ~5%

**Phase 3A: Ripgrep Integration** (3-4 hours effort, 3-5x ROI)
- Problem: Grep searches find irrelevant plugin code instead of core library definitions
- Solution: Ripgrep with camelCase/snake_case variant detection + scope-aware searching
- Expected Result: 3-5x better search relevance, reduce dead-ends from ~20% to ~5%

---

## ⚖️ License
This project is licensed under the MIT License.

