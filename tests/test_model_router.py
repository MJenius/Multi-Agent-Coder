"""Tests for the Dynamic Model Router.

Verifies role configuration resolution, condition matching,
overrides, and fallback handling.
"""

from __future__ import annotations

from issue_resolver.core.model_router import ModelRouter, ModelConfig


def test_model_router_resolution() -> None:
    config_data = {
        "roles": {
            "planner": {
                "default": {
                    "model": "z-ai/glm-5.1",
                    "api_key_env": "NVIDIA_API_KEY_PLANNER",
                    "temperature": 1.0,
                    "max_tokens": 16384,
                },
                "overrides": [
                    {
                        "condition": {"retry_count": ">2"},
                        "model": "qwen/qwen3.5-122b-a10b",
                    }
                ],
                "fallback_models": ["meta/llama-3.3-70b-instruct"],
            }
        }
    }

    router = ModelRouter.from_dict(config_data)

    # 1. Resolve default
    cfg = router.resolve("planner", context={"retry_count": 0})
    assert cfg.model == "z-ai/glm-5.1"
    assert cfg.temperature == 1.0
    assert cfg.max_tokens == 16384

    # 2. Resolve with override
    cfg_override = router.resolve("planner", context={"retry_count": 3})
    assert cfg_override.model == "qwen/qwen3.5-122b-a10b"

    # 3. Resolve with model failure (should use fallback)
    router.mark_failed("z-ai/glm-5.1")
    cfg_fail = router.resolve("planner", context={"retry_count": 0})
    assert cfg_fail.model == "meta/llama-3.3-70b-instruct"
