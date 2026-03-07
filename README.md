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
- **Supervisor (`llama3.2:3b`)**: The intelligent router. Monitors the `AgentState`, preventing infinite loops and deciding whether it needs more research, a coding fix, or if it can gracefully exit and finalize the report.
- **Researcher (`llama3.2:3b`)**: Explores the local codebase utilizing custom tool binding (`list_files`, `search_code`, `read_file`). Keeps strict token & RAM constraints on file reads.
- **Coder (`qwen2.5-coder:7b`)**: Extremely focused code generation model. Analyzes the file context against the GitHub issue and generates surgical string replace operations packaged as a `Unified Diff`.
- **Reviewer (Deterministic)**: The Judge. Mounts the target code into an isolated, network-disabled **Docker Container**. Applies the Coder's patches and executes unit tests. Returns raw tracebacks if it fails.

---

## 🛠️ Technology Stack
- **Framework:** [LangGraph](https://python.langchain.com/docs/langgraph) (Stateful multi-actor orchestration)
- **Local Inference:** [Ollama](https://ollama.com/) (`llama3.2:3b` & `qwen2.5-coder:7b`)
- **Isolation/Sandbox:** Docker Engine (python-slim container)
- **Analytics & Logging:** Native Python traces & token estimations.

---

## 🚀 Getting Started

### Prerequisites
1. **Python 3.12+**
2. **Docker Engine / Desktop** installed and running.
3. **Ollama** installed with models pulled locally (approx. 16GB RAM constraints):
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
Upon completion, the Supervisor generates a `resolution_report.json` showcasing the metrics of the run, mapping out iterations, token consumption, and trace logs:
```json
{
    "is_resolved": true,
    "total_iterations": 3,
    "total_character_estimate": 1502,
    "total_token_estimate": 375,
    "files_read_summary": [
        "Tool: read_file\nArgs: {\"file_path\": \"./sandbox_workspace/utils.py\"}"
    ],
    "failed_diffs_and_tracebacks": [],
    "final_successful_diff": "```diff\n--- a/utils.py\n+++ b/utils.py...",
    "final_errors": null,
    "full_history_trace": [...]
}
```

---

## ⚖️ License
This project is licensed under the MIT License.
