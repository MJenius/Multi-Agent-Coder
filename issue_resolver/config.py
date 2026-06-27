from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _parse_model_list(env_key: str, fallback: list[str]) -> list[str]:
    raw = os.environ.get(env_key, "")
    if not raw.strip():
        return fallback
    parsed = [part.strip() for part in raw.split(",") if part.strip()]
    return parsed or fallback


NVIDIA_BASE_URL: str = os.environ.get(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)

NVIDIA_API_KEY_TIER1: str = os.environ.get("NVIDIA_API_KEY_TIER1", "")
NVIDIA_API_KEY_TIER2: str = os.environ.get("NVIDIA_API_KEY_TIER2", "")
NVIDIA_API_KEY_TIER3: str = os.environ.get("NVIDIA_API_KEY_TIER3", "")
NVIDIA_API_KEY_TIER4: str = os.environ.get("NVIDIA_API_KEY_TIER4", "")

MODEL_API_KEY_MAP: dict[str, str] = {
    "nvidia/nemotron-3-super-120b-a12b": NVIDIA_API_KEY_TIER1,
    "meta/llama-3.3-70b-instruct": NVIDIA_API_KEY_TIER2,
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": NVIDIA_API_KEY_TIER3,
    "meta/llama-3.1-8b-instruct": NVIDIA_API_KEY_TIER4,
    "nvidia/llama-3.1-nemotron-nano-8b-v1": NVIDIA_API_KEY_TIER3,
}

SUPERVISOR_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "NVIDIA_SUPERVISOR_MODELS",
    [
        "nvidia/nemotron-3-super-120b-a12b",
        "meta/llama-3.3-70b-instruct",
    ],
)

PLANNER_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "NVIDIA_PLANNER_MODELS",
    [
        "nvidia/nemotron-3-super-120b-a12b",
        "meta/llama-3.3-70b-instruct",
    ],
)

CODER_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "NVIDIA_CODER_MODELS",
    [
        "meta/llama-3.3-70b-instruct",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    ],
)

TESTGEN_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "NVIDIA_TESTGEN_MODELS",
    [
        "meta/llama-3.3-70b-instruct",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    ],
)

RESEARCHER_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "NVIDIA_RESEARCHER_MODELS",
    [
        "meta/llama-3.1-8b-instruct",
        "nvidia/llama-3.1-nemotron-nano-8b-v1",
    ],
)

REVIEWER_MODEL_CANDIDATES: list[str] = _parse_model_list(
    "NVIDIA_REVIEWER_MODELS",
    [
        "meta/llama-3.1-8b-instruct",
        "nvidia/llama-3.1-nemotron-nano-8b-v1",
    ],
)

SUPERVISOR_MODEL: str = SUPERVISOR_MODEL_CANDIDATES[0]
PLANNER_MODEL: str = PLANNER_MODEL_CANDIDATES[0]
CODER_MODEL: str = CODER_MODEL_CANDIDATES[0]
TESTGEN_MODEL: str = TESTGEN_MODEL_CANDIDATES[0]
RESEARCHER_MODEL: str = RESEARCHER_MODEL_CANDIDATES[0]
REVIEWER_MODEL: str = REVIEWER_MODEL_CANDIDATES[0]

NVIDIA_CONTEXT_WINDOWS: dict[str, int] = {
    "nvidia/nemotron-3-super-120b-a12b": 1_048_576,
    "meta/llama-3.3-70b-instruct": 131_072,
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": 131_072,
    "meta/llama-3.1-8b-instruct": 131_072,
    "nvidia/llama-3.1-nemotron-nano-8b-v1": 131_072,
}

CODER_NUM_PREDICT: int = int(os.environ.get("CODER_NUM_PREDICT", "4096"))
CODER_MAX_RETRIES: int = int(os.environ.get("CODER_MAX_RETRIES", "2"))
CODER_RETRY_BUDGET: int = int(os.environ.get("CODER_RETRY_BUDGET", "3"))

CODER_MAX_OUTPUT_RATIO: float = float(os.environ.get("CODER_MAX_OUTPUT_RATIO", "0.3"))
CODER_MIN_OUTPUT_TOKENS: int = int(os.environ.get("CODER_MIN_OUTPUT_TOKENS", "1000"))
CODER_TARGET_OUTPUT_TOKENS: int = int(os.environ.get("CODER_TARGET_OUTPUT_TOKENS", "16384"))

LLM_MAX_ATTEMPTS: int = int(os.environ.get("LLM_MAX_ATTEMPTS", "4"))
LLM_BACKOFF_INITIAL_SECONDS: float = float(os.environ.get("LLM_BACKOFF_INITIAL_SECONDS", "1.0"))
LLM_BACKOFF_MULTIPLIER: float = float(os.environ.get("LLM_BACKOFF_MULTIPLIER", "2.0"))
LLM_BACKOFF_MAX_SECONDS: float = float(os.environ.get("LLM_BACKOFF_MAX_SECONDS", "30.0"))

NVIDIA_RPM_LIMIT: int = int(os.environ.get("NVIDIA_RPM_LIMIT", "60"))
NVIDIA_TPM_LIMIT: int = int(os.environ.get("NVIDIA_TPM_LIMIT", "100000"))

PLANNER_MAX_ITERATIONS: int = int(os.environ.get("PLANNER_MAX_ITERATIONS", "2"))

MAX_ITERATIONS: int = int(os.environ.get("MAX_ITERATIONS", "5"))

SANDBOX_WORKSPACE_DIR: str = os.environ.get(
    "SANDBOX_WORKSPACE_DIR",
    os.path.abspath("sandbox_workspace"),
)

POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "user")
POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "password")
POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "langgraph_state")
