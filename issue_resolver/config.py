"""
Centralized configuration -- all tunables in one place.

Values are read from environment variables with sensible defaults
(matching the original hardcoded values). Create a `.env` file in
the project root to override any setting.
"""

from __future__ import annotations

import os

# Load .env file if python-dotenv is installed (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

SUPERVISOR_MODEL: str = os.environ.get("OLLAMA_SUPERVISOR_MODEL", "llama3.2:latest")
CODER_MODEL: str = os.environ.get("OLLAMA_CODER_MODEL", "qwen2.5-coder:7b")
RESEARCHER_MODEL: str = os.environ.get("OLLAMA_RESEARCHER_MODEL", "llama3.2:latest")

# Performance-optimized defaults (reduce for faster operation, increase for complex fixes)
# Reduced from 1024 to 400 for faster generation (most fixes are ~100-200 tokens)
CODER_NUM_PREDICT: int = int(os.environ.get("OLLAMA_CODER_NUM_PREDICT", "400"))
# Reduced from 2 to 1 retry (2 total attempts) - smarter targeting makes retries less needed  
CODER_MAX_RETRIES: int = int(os.environ.get("CODER_MAX_RETRIES", "1"))

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
MAX_ITERATIONS: int = int(os.environ.get("MAX_ITERATIONS", "5"))

# ---------------------------------------------------------------------------
# Sandbox / workspace
# ---------------------------------------------------------------------------
SANDBOX_WORKSPACE_DIR: str = os.environ.get(
    "SANDBOX_WORKSPACE_DIR",
    os.path.abspath("sandbox_workspace"),
)

# ---------------------------------------------------------------------------
# PostgreSQL (used in docker-compose, referenced here for documentation)
# ---------------------------------------------------------------------------
POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "user")
POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "password")
POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "langgraph_state")
