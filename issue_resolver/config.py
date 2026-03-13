"""Centralized configuration for model, graph, and sandbox settings."""

from __future__ import annotations

import os


def _parse_model_list(env_key: str, fallback: list[str]) -> list[str]:
    """Parse comma-separated model names from environment with stable fallback order."""
    raw = os.environ.get(env_key, "")
    if not raw.strip():
        return fallback
    parsed = [part.strip() for part in raw.split(",") if part.strip()]
    return parsed or fallback

# Load .env file if python-dotenv is installed (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Groq / LLM
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

SUPERVISOR_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "GROQ_SUPERVISOR_MODELS",
    [
        "llama-3.3-70b-versatile",
        "qwen-2.5-coder-32b",  # fallback for reasoning tasks
    ],
)
RESEARCHER_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "GROQ_RESEARCHER_MODELS",
    [
        "llama-3.3-70b-versatile",
        "qwen-2.5-coder-32b",  # fallback with tool binding support
    ],
)
CODER_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "GROQ_CODER_MODELS",
    [
        "qwen-2.5-coder-32b",  # best for code generation
        "llama-3.3-70b-versatile",  # strong fallback
    ],
)
REVIEWER_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "GROQ_REVIEWER_MODELS",
    [
        "llama-3.1-8b-instant",  # fast, sufficient for validation
        "llama-3.3-70b-versatile",  # stronger fallback
    ],
)

SUPERVISOR_MODEL: str = SUPERVISOR_MODEL_CANDIDATES[0]
RESEARCHER_MODEL: str = RESEARCHER_MODEL_CANDIDATES[0]
CODER_MODEL: str = CODER_MODEL_CANDIDATES[0]
REVIEWER_MODEL: str = REVIEWER_MODEL_CANDIDATES[0]

# Performance-optimized defaults
CODER_NUM_PREDICT: int = int(os.environ.get("GROQ_CODER_MAX_TOKENS", "400"))
CODER_MAX_RETRIES: int = int(os.environ.get("CODER_MAX_RETRIES", "1"))

# Dynamic token allocation for Coder
# Maps model name to context window size (in tokens)
GROQ_CONTEXT_WINDOWS: dict[str, int] = {
    "qwen-2.5-coder-32b": 32768,
    "llama-3.3-70b-versatile": 8192,
    "llama-3.1-8b-instant": 8192,
}

CODER_MAX_OUTPUT_RATIO: float = float(os.environ.get("CODER_MAX_OUTPUT_RATIO", "0.3"))
CODER_MIN_OUTPUT_TOKENS: int = int(os.environ.get("CODER_MIN_OUTPUT_TOKENS", "500"))
CODER_TARGET_OUTPUT_TOKENS: int = int(os.environ.get("CODER_TARGET_OUTPUT_TOKENS", "3000"))

# Retry controls for transient API failures and rate limits.
LLM_MAX_ATTEMPTS: int = int(os.environ.get("LLM_MAX_ATTEMPTS", "4"))
LLM_BACKOFF_INITIAL_SECONDS: float = float(os.environ.get("LLM_BACKOFF_INITIAL_SECONDS", "1.0"))
LLM_BACKOFF_MULTIPLIER: float = float(os.environ.get("LLM_BACKOFF_MULTIPLIER", "2.0"))
LLM_BACKOFF_MAX_SECONDS: float = float(os.environ.get("LLM_BACKOFF_MAX_SECONDS", "12.0"))

# Rate limiting configuration (Groq API)
GROQ_RPM_LIMIT: int = int(os.environ.get("GROQ_RPM_LIMIT", "30"))
GROQ_TPM_LIMIT: int = int(os.environ.get("GROQ_TPM_LIMIT", "6000"))

# Planner and TestGen model candidates
PLANNER_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "GROQ_PLANNER_MODELS",
    [
        "llama-3.3-70b-versatile",  # strong reasoning for strategy
        "qwen-2.5-coder-32b",  # fallback for planning
    ],
)
TESTGEN_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "GROQ_TESTGEN_MODELS",
    [
        "qwen-2.5-coder-32b",  # best for test generation
        "llama-3.3-70b-versatile",  # fallback
    ],
)

PLANNER_MODEL: str = PLANNER_MODEL_CANDIDATES[0]
TESTGEN_MODEL: str = TESTGEN_MODEL_CANDIDATES[0]

# Planner refinement limits
PLANNER_MAX_ITERATIONS: int = int(os.environ.get("PLANNER_MAX_ITERATIONS", "2"))

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
